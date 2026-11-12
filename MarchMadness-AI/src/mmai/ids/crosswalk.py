"""Learn team-ID crosswalks from games, not names.

**ESPN → Kaggle.** Join ESPN's completed games to Kaggle's results on (date ±1 day, winner's
score, loser's score). Each unambiguous match is one vote that ESPN's winner is Kaggle's
``WTeamID`` and its loser is ``LTeamID``. A team plays ~30 games a season, so the votes are
overwhelming and names never come into it, which is exactly what went wrong when the legacy
fuzzy matcher turned Miami (OH) into Miami (FL).

**Odds API name → Kaggle.** Match each Odds API event to the ESPN game that tips within a
window of its ``commence_time`` and whose team names fit best, then inherit ESPN's learned id.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

XW_COLUMNS = ["espn_id", "kaggle_id", "votes", "share", "espn_name"]


@dataclass
class CrosswalkReport:
    games: int = 0
    matched: int = 0
    ambiguous: int = 0
    unmatched: int = 0
    mapped: int = 0
    low_confidence: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def kaggle_results_with_dates(results: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """Add a calendar ``date`` column to Kaggle results (DayZero + DayNum)."""
    zero = {int(s): datetime.strptime(str(d), "%m/%d/%Y").date() for s, d in zip(seasons["Season"], seasons["DayZero"])}
    out = results.copy()
    out["date"] = [zero[int(s)] + timedelta(days=int(n)) for s, n in zip(out["Season"], out["DayNum"])]
    return out


def learn_espn_crosswalk(
    espn_games: pd.DataFrame,
    kaggle_results: pd.DataFrame,
    *,
    slack_days: int = 1,
    min_votes: int = 3,
    min_share: float = 0.8,
) -> tuple[pd.DataFrame, CrosswalkReport]:
    """``espn_games``: date_et, home/away_id, home/away_score, home/away_name, completed.
    ``kaggle_results``: date, WTeamID, LTeamID, WScore, LScore."""
    rep = CrosswalkReport()
    index: dict[tuple[int, int], list[tuple[date, int, int]]] = defaultdict(list)
    for r in kaggle_results.itertuples(index=False):
        index[(int(r.WScore), int(r.LScore))].append((r.date, int(r.WTeamID), int(r.LTeamID)))

    votes: dict[str, Counter] = defaultdict(Counter)
    names: dict[str, str] = {}
    games = espn_games[espn_games["completed"] & espn_games["home_score"].notna() & espn_games["away_score"].notna()]
    for r in games.itertuples(index=False):
        hs, aws = int(r.home_score), int(r.away_score)
        if hs == aws:
            continue
        names[str(r.home_id)], names[str(r.away_id)] = r.home_name, r.away_name
        if hs > aws:
            (wid, ws), (lid, ls) = (str(r.home_id), hs), (str(r.away_id), aws)
        else:
            (wid, ws), (lid, ls) = (str(r.away_id), aws), (str(r.home_id), hs)
        d = date.fromisoformat(str(r.date_et))
        rep.games += 1
        candidates = index.get((ws, ls), [])
        pick = [c for c in candidates if c[0] == d] or [c for c in candidates if abs((c[0] - d).days) <= slack_days]
        if len(pick) == 1:
            _, kw, kl = pick[0]
            votes[wid][kw] += 1
            votes[lid][kl] += 1
            rep.matched += 1
        elif pick:
            rep.ambiguous += 1
        else:
            rep.unmatched += 1  # typically a non-Division I opponent, absent from Kaggle

    rows = []
    for espn_id, counts in votes.items():
        (kid, n), total = counts.most_common(1)[0], sum(counts.values())
        row = {"espn_id": espn_id, "kaggle_id": int(kid), "votes": n, "share": round(n / total, 3),
               "espn_name": names.get(espn_id)}
        (rows if n >= min_votes and n / total >= min_share else rep.low_confidence).append(row)
    xw = pd.DataFrame(rows, columns=XW_COLUMNS)
    if len(xw):
        xw = xw.sort_values("votes", ascending=False)
        dup = xw.duplicated("kaggle_id", keep="first")  # enforce one-to-one
        rep.conflicts = xw[dup].to_dict("records")
        xw = xw[~dup].sort_values("espn_name").reset_index(drop=True)
    rep.mapped = len(xw)
    return xw, rep


# --- names ------------------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def normalize_name(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    return " ".join(_NON_ALNUM.sub(" ", s).split())


def name_similarity(a: str | None, b: str | None) -> float:
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def _side_sim(name: str, game: pd.Series, side: str) -> float:
    return max(name_similarity(name, game[f"{side}_name"]), name_similarity(name, game.get(f"{side}_location")))


def learn_name_crosswalk(
    events: pd.DataFrame,
    espn_games: pd.DataFrame,
    espn_to_kaggle: dict[str, int],
    *,
    window_minutes: int = 90,
    min_score: float = 1.3,
    min_margin: float = 0.15,
) -> tuple[pd.DataFrame, dict]:
    """``events``: commence_utc, home_team, away_team (Odds API). ``espn_games``: start_utc,
    home/away_id, home/away_name, home/away_location. Returns (name crosswalk, report)."""
    eg = espn_games.dropna(subset=["start_utc"]).copy()
    eg["t"] = pd.to_datetime(eg["start_utc"], utc=True).dt.tz_localize(None)  # naive UTC for searchsorted
    eg = eg.sort_values("t").reset_index(drop=True)
    times = eg["t"].to_numpy(dtype="datetime64[ns]")
    window = np.timedelta64(window_minutes, "m")

    votes: dict[str, Counter] = defaultdict(Counter)
    matched = unmatched = 0
    uniq = events.drop_duplicates(["commence_utc", "home_team", "away_team"])
    for e in uniq.itertuples(index=False):
        t = pd.Timestamp(e.commence_utc)
        t = (t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t).to_datetime64()
        lo, hi = np.searchsorted(times, t - window), np.searchsorted(times, t + window, side="right")
        scored = []
        for j in range(lo, hi):
            g = eg.iloc[j]
            same = _side_sim(e.home_team, g, "home") + _side_sim(e.away_team, g, "away")
            swap = _side_sim(e.home_team, g, "away") + _side_sim(e.away_team, g, "home")
            scored.append((max(same, swap), swap > same, j))
        scored.sort(reverse=True)
        if not scored or scored[0][0] < min_score or (len(scored) > 1 and scored[0][0] - scored[1][0] < min_margin):
            unmatched += 1
            continue
        _, swapped, j = scored[0]
        g = eg.iloc[j]
        home_id, away_id = (g["away_id"], g["home_id"]) if swapped else (g["home_id"], g["away_id"])
        votes[e.home_team][str(home_id)] += 1
        votes[e.away_team][str(away_id)] += 1
        matched += 1

    rows = []
    for name, counts in votes.items():
        (espn_id, n), total = counts.most_common(1)[0], sum(counts.values())
        rows.append({"name": name, "espn_id": espn_id, "kaggle_id": espn_to_kaggle.get(espn_id),
                     "votes": n, "share": round(n / total, 3)})
    out = pd.DataFrame(rows, columns=["name", "espn_id", "kaggle_id", "votes", "share"])
    return out.sort_values("name").reset_index(drop=True), {"events": matched + unmatched, "matched": matched,
                                                           "unmatched": unmatched}
