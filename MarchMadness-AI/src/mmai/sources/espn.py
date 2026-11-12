"""ESPN's public JSON endpoints for college basketball (unofficial and undocumented).

Verified from this machine on 2026-11-11:

- **scoreboard**: one call per date returns every Division I game (``groups=50``) with scores,
  tip time, neutral-site flag and ESPN team ids.
- **summary**: one call per game returns team box scores covering every column of Kaggle's
  detailed results (FGM/FGA, 3s, FTs, OR/DR, Ast, TO, Stl, Blk, PF).
- **core odds**: one call per game returns lines by book. Men's coverage goes back to at least
  2017; items from 2025 on carry ``open``/``close``/``current`` blocks. ``spread`` is the
  *home* team's line.

Raw responses are cached forever under ``workspace/raw/espn`` and the tidy tables are rebuilt
from that cache, so an endpoint change can't take history away.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from mmai.calendar import et_date, to_utc
from mmai.sources.http import CachedJSONClient

LEAGUE = {"M": "mens-college-basketball", "W": "womens-college-basketball"}
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/{league}"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/{league}"
GROUP_DIVISION_I = 50
#: Regulation periods: men play halves; women have played quarters since 2015-16.
REGULATION_PERIODS = {"M": 2, "W": 4}


class ESPN:
    def __init__(self, client: CachedJSONClient, *, today: date | None = None):
        self.client = client
        self._today = today

    @property
    def today(self) -> date:
        return self._today or datetime.now(timezone.utc).date()

    def _ttl_for(self, d: date) -> float | None:
        """Dates more than two days old are final (cache forever); recent ones refresh after 20 min."""
        return None if d < self.today - timedelta(days=2) else 1200.0

    def scoreboard(self, gender: str, d: date) -> dict:
        r = self.client.get(
            f"{SITE.format(league=LEAGUE[gender])}/scoreboard",
            {"dates": d.strftime("%Y%m%d"), "groups": GROUP_DIVISION_I, "limit": 500},
            ttl_s=self._ttl_for(d),
        )
        return r.body or {}

    def summary(self, gender: str, event_id: str, *, final: bool = True) -> dict:
        r = self.client.get(
            f"{SITE.format(league=LEAGUE[gender])}/summary", {"event": str(event_id)},
            ttl_s=None if final else 600.0,
        )
        return r.body or {}

    def odds(self, gender: str, event_id: str, *, final: bool = True) -> dict:
        base = CORE.format(league=LEAGUE[gender])
        r = self.client.get(
            f"{base}/events/{event_id}/competitions/{event_id}/odds", {"limit": 100},
            ttl_s=None if final else 600.0,
        )
        return r.body or {}


# --- parsing helpers ---------------------------------------------------------------------

def _int(x: Any) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _num(x: Any) -> float | None:
    """Numbers from ESPN odds strings: '+25.5', '-110', '137.5', 'EVEN' (=+100), 'PK' (=0)."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().upper()
    if s in ("EVEN", "EV"):
        return 100.0
    if s in ("PK", "PICK", "PICK'EM"):
        return 0.0
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return None


def _split_made_att(s: Any) -> tuple[int | None, int | None]:
    if not isinstance(s, str) or "-" not in s:
        return None, None
    made, att = s.split("-", 1)
    return _int(made), _int(att)


def spread_from_details(details: str | None, home_abbr: str | None, away_abbr: str | None) -> float | None:
    """Home-team spread from ESPN's ``details`` string, e.g. 'DUKE -28.5' with DUKE at home -> -28.5."""
    if not details:
        return None
    s = details.strip()
    if s.upper() in ("EVEN", "PK", "PICK"):
        return 0.0
    parts = s.rsplit(" ", 1)
    if len(parts) != 2:
        return None
    abbr, line = parts[0].strip(), _num(parts[1])
    if line is None:
        return None
    if home_abbr and abbr == home_abbr:
        return line
    if away_abbr and abbr == away_abbr:
        return -line
    return None


# --- parsers -------------------------------------------------------------------------------

def parse_scoreboard(sb: dict, gender: str) -> list[dict]:
    """One row per game on a scoreboard response."""
    rows = []
    for e in sb.get("events") or []:
        comps = e.get("competitions") or []
        if not comps:
            continue
        c = comps[0]
        sides = {x.get("homeAway"): x for x in c.get("competitors") or []}
        if set(sides) != {"home", "away"}:
            continue
        status = ((c.get("status") or e.get("status")) or {}).get("type") or {}
        notes = c.get("notes") or []
        venue = c.get("venue") or {}
        address = venue.get("address") or {}
        start = c.get("date") or e.get("date")
        season = e.get("season") or {}
        row: dict[str, Any] = {
            "gender": gender,
            "event_id": str(e.get("id")),
            "start_utc": to_utc(start) if start else None,
            "date_et": et_date(start).isoformat() if start else None,
            "espn_season": season.get("year"),
            "season_type": season.get("type"),  # 2 = regular season, 3 = postseason
            "completed": bool(status.get("completed")),
            "status": status.get("name"),
            "neutral": bool(c.get("neutralSite")),
            "conference_game": bool(c.get("conferenceCompetition")),
            "tournament_id": c.get("tournamentId"),
            "note": (notes[0] or {}).get("headline") if notes else None,
            "venue": venue.get("fullName"),
            "city": address.get("city"),
            "state": address.get("state"),
            "periods": len(sides["home"].get("linescores") or []) or None,
        }
        for side in ("home", "away"):
            x = sides[side]
            t = x.get("team") or {}
            row[f"{side}_id"] = str(t.get("id") or x.get("id"))
            row[f"{side}_name"] = t.get("displayName")
            row[f"{side}_short"] = t.get("shortDisplayName")
            row[f"{side}_abbr"] = t.get("abbreviation")
            row[f"{side}_location"] = t.get("location")
            row[f"{side}_conf_id"] = str(t.get("conferenceId")) if t.get("conferenceId") is not None else None
            row[f"{side}_score"] = _int(x.get("score"))
        rows.append(row)
    return rows


BOX_PAIRS = {
    "fieldGoalsMade-fieldGoalsAttempted": ("FGM", "FGA"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("FGM3", "FGA3"),
    "freeThrowsMade-freeThrowsAttempted": ("FTM", "FTA"),
}
BOX_SINGLE = {
    "offensiveRebounds": "OR",
    "defensiveRebounds": "DR",
    "assists": "Ast",
    "steals": "Stl",
    "blocks": "Blk",
    "fouls": "PF",
}
BOX_COLUMNS = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]


def parse_box(summary: dict, gender: str, event_id: str) -> list[dict]:
    """Two rows (one per team) in Kaggle's detailed-results vocabulary, or [] if unavailable."""
    rows = []
    for t in (summary.get("boxscore") or {}).get("teams") or []:
        stats = {s.get("name"): s.get("displayValue") for s in t.get("statistics") or []}
        row: dict[str, Any] = {
            "gender": gender,
            "event_id": str(event_id),
            "team_id": str((t.get("team") or {}).get("id")),
            "home_away": t.get("homeAway"),
        }
        for name, (made, att) in BOX_PAIRS.items():
            row[made], row[att] = _split_made_att(stats.get(name))
        for name, col in BOX_SINGLE.items():
            row[col] = _int(stats.get(name))
        # Kaggle's TO counts team turnovers too; ESPN reports them separately.
        row["TO"] = _int(stats.get("totalTurnovers", stats.get("turnovers")))
        rows.append(row)
    if len(rows) != 2 or any(r["FGA"] is None for r in rows):
        return []
    return rows


def parse_odds(odds: dict, gender: str, event_id: str, home_abbr: str | None = None,
               away_abbr: str | None = None) -> list[dict]:
    """One row per book. ``spread_home`` is the home team's line (negative = home favored)."""
    rows = []
    for it in odds.get("items") or []:
        provider = it.get("provider") or {}
        home = it.get("homeTeamOdds") or {}
        away = it.get("awayTeamOdds") or {}
        row: dict[str, Any] = {
            "gender": gender,
            "event_id": str(event_id),
            "provider_id": str(provider.get("id")),
            "provider": provider.get("name"),
            "provider_priority": provider.get("priority"),
            "details": it.get("details"),
            "spread_home": _num(it.get("spread")),
            "spread_home_from_details": spread_from_details(it.get("details"), home_abbr, away_abbr),
            "total": _num(it.get("overUnder")),
            "over_price": _num(it.get("overOdds")),
            "under_price": _num(it.get("underOdds")),
            "ml_home": _num(home.get("moneyLine")),
            "ml_away": _num(away.get("moneyLine")),
            "spread_price_home": _num(home.get("spreadOdds")),
            "spread_price_away": _num(away.get("spreadOdds")),
        }
        for phase in ("open", "close"):
            hp, ap, tp = home.get(phase) or {}, away.get(phase) or {}, it.get(phase) or {}
            row[f"{phase}_spread_home"], row[f"{phase}_total"] = _phase_lines(hp, tp)
            row[f"{phase}_ml_home"] = _num((hp.get("moneyLine") or {}).get("american"))
            row[f"{phase}_ml_away"] = _num((ap.get("moneyLine") or {}).get("american"))
        rows.append(row)
    return rows


MAX_ABS_SPREAD = 75.0  # no college spread is this big, and American prices never fall inside (-100, +100)
TOTAL_RANGE = (60.0, 300.0)


def _phase_lines(home_phase: dict, total_phase: dict) -> tuple[float | None, float | None]:
    """(home spread, total) from an ``open``/``close`` block; None where the field holds something
    other than a line. ESPN BET's 2023 close blocks carry the price there (pointSpread "-115",
    total "-115"), while the real line sits in ``current`` and the item's top-level fields."""
    spread = _num((home_phase.get("pointSpread") or {}).get("american"))
    total = _num((total_phase.get("total") or {}).get("american"))
    prices = {_num((total_phase.get(k) or {}).get("american")) for k in ("over", "under")}
    if spread is not None and abs(spread) > MAX_ABS_SPREAD:
        spread = None
    if total is not None and (not TOTAL_RANGE[0] <= total <= TOTAL_RANGE[1] or total in prices):
        total = None
    return spread, total


# --- collection ------------------------------------------------------------------------------

def collect(
    espn: ESPN,
    gender: str,
    dates: Iterable[date],
    *,
    box: bool = False,
    odds: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Scoreboards for ``dates``; optionally box scores and lines for every completed game."""
    games: list[dict] = []
    for d in dates:
        games.extend(parse_scoreboard(espn.scoreboard(gender, d), gender))
    gdf = pd.DataFrame(games)
    if gdf.empty:
        return {"games": gdf, "box": pd.DataFrame(), "odds": pd.DataFrame()}
    gdf = gdf.drop_duplicates("event_id", keep="last").reset_index(drop=True)
    box_rows: list[dict] = []
    odds_rows: list[dict] = []
    done = gdf[gdf["completed"]]
    for i, r in enumerate(done.itertuples(index=False), 1):
        if box:
            box_rows.extend(parse_box(espn.summary(gender, r.event_id), gender, r.event_id))
        if odds:
            odds_rows.extend(parse_odds(espn.odds(gender, r.event_id), gender, r.event_id, r.home_abbr, r.away_abbr))
        if log and (box or odds) and i % 250 == 0:
            log(f"  {gender}: {i}/{len(done)} games")
    return {"games": gdf, "box": pd.DataFrame(box_rows), "odds": pd.DataFrame(odds_rows)}
