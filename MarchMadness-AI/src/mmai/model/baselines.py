"""Baselines the ratings engine has to beat (prereg G1/G3). All are strictly as-of: a game is
predicted from games on earlier days only.

- raw net rating: season-to-date offensive/defensive points per 100 possessions and pace, with
  no opponent adjustment
- Elo: margin-of-victory Elo carried across seasons with regression to the mean
- Massey consensus (men): median over a fixed set of long-running systems of each system's
  rank mapped to a normal score, joined on ``RankingDayNum <= DayNum`` (Kaggle's documented meaning)
- seed: NCAA-tournament seed difference
The ridge floor is ``RatingsConfig(target="points", carryover=0, prior_games=0)`` in ratings.py.
"""

from __future__ import annotations

import bisect
import math

import pandas as pd
from scipy.stats import norm

MASSEY_SYSTEMS = ("POM", "MOR", "WLK", "DOL", "COL", "MAS", "WOL", "RTH", "DOK", "WIL", "PGH", "KPK")


def _rates(t: int, acc: dict, lg_ppp: float, lg_pace: float) -> tuple[float, float, float]:
    """Season-to-date (ORtg, DRtg, pace) for team t; league averages before its first game."""
    n = acc["n"].get(t)
    if not n:
        return lg_ppp, lg_ppp, lg_pace
    return 100.0 * acc["pf"][t] / acc["poss"][t], 100.0 * acc["pa"][t] / acc["poss"][t], acc["poss40"][t] / n


def raw_net_rating(games: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date ORtg/DRtg/pace per team -> expected regulation points (pts_a, pts_b)."""
    out = []
    for _, gs in games.groupby("season", sort=True):
        acc: dict[str, dict] = {"pf": {}, "pa": {}, "poss": {}, "poss40": {}, "n": {}}
        lg = {"pts": 0.0, "poss": 0.0, "poss40": 0.0, "games": 0, "home_edge": 0.0, "home_n": 0}
        for _, dg in gs.groupby("daynum", sort=True):
            lg_ppp = 100.0 * lg["pts"] / lg["poss"] if lg["poss"] else 100.0
            lg_pace = lg["poss40"] / lg["games"] if lg["games"] else 68.0
            hca = lg["home_edge"] / lg["home_n"] if lg["home_n"] else 3.0
            for r in dg.itertuples(index=False):
                oa, da, pace_a = _rates(r.team_a, acc, lg_ppp, lg_pace)
                ob, db, pace_b = _rates(r.team_b, acc, lg_ppp, lg_pace)
                p = (pace_a + pace_b) / 2.0
                ppp_a = (oa + db) / 2.0 + hca * r.venue_a
                ppp_b = (ob + da) / 2.0 - hca * r.venue_a
                out.append((r.game_id, p * ppp_a / 100.0, p * ppp_b / 100.0, p))
            for r in dg.itertuples(index=False):  # update after predicting the whole day
                for t, s_for, s_ag in ((r.team_a, r.score_a, r.score_b), (r.team_b, r.score_b, r.score_a)):
                    acc["pf"][t] = acc["pf"].get(t, 0.0) + s_for
                    acc["pa"][t] = acc["pa"].get(t, 0.0) + s_ag
                    acc["poss"][t] = acc["poss"].get(t, 0.0) + r.poss
                    acc["poss40"][t] = acc["poss40"].get(t, 0.0) + r.poss40
                    acc["n"][t] = acc["n"].get(t, 0) + 1
                lg["pts"] += r.score_a + r.score_b
                lg["poss"] += 2 * r.poss
                lg["poss40"] += r.poss40
                lg["games"] += 1
                if r.venue_a:
                    lg["home_edge"] += 100.0 * (r.score_a - r.score_b) / r.poss / 2.0
                    lg["home_n"] += 1
    return pd.DataFrame(out, columns=["game_id", "raw_pts_a", "raw_pts_b", "raw_poss40"])


def elo(games: pd.DataFrame, *, k: float = 20.0, hca: float = 65.0, carry: float = 0.75) -> pd.DataFrame:
    """Pre-game Elo difference (A minus B, home advantage included) for every game."""
    ratings: dict[int, float] = {}
    out = []
    season_prev = None
    for (season, _), dg in games.groupby(["season", "daynum"], sort=True):
        if season != season_prev:
            ratings = {t: 1500.0 + carry * (r - 1500.0) for t, r in ratings.items()}
            season_prev = season
        diffs = [ratings.get(r.team_a, 1500.0) - ratings.get(r.team_b, 1500.0) + hca * r.venue_a
                 for r in dg.itertuples(index=False)]
        out.extend(zip(dg["game_id"], diffs))
        for r, diff in zip(dg.itertuples(index=False), diffs):
            expected_a = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
            a_won = r.score_a > r.score_b
            winner_edge = diff if a_won else -diff
            mult = math.log(abs(r.score_a - r.score_b) + 1.0) * 2.2 / (0.001 * winner_edge + 2.2)
            delta = k * mult * ((1.0 if a_won else 0.0) - expected_a)
            ratings[r.team_a] = ratings.get(r.team_a, 1500.0) + delta
            ratings[r.team_b] = ratings.get(r.team_b, 1500.0) - delta
    return pd.DataFrame(out, columns=["game_id", "elo_diff_a"])


def massey_consensus(massey: pd.DataFrame, games: pd.DataFrame, systems=MASSEY_SYSTEMS) -> pd.DataFrame:
    """Consensus normal-score difference (A minus B) as of each game's day; NaN if unranked."""
    m = massey[massey["SystemName"].isin(systems)]
    out = []
    for season, gs in games.groupby("season", sort=True):
        ms = m[m["Season"] == season]
        if ms.empty:
            continue
        by_sys = {}
        for sysname, s in ms.groupby("SystemName"):
            days = sorted(s["RankingDayNum"].unique())
            by_sys[sysname] = (days, {d: g.set_index("TeamID")["OrdinalRank"] for d, g in s.groupby("RankingDayNum")})
        for day, dg in gs.groupby("daynum", sort=True):
            cols = []
            for days, snaps in by_sys.values():
                i = bisect.bisect_right(days, day) - 1  # latest RankingDayNum <= day
                if i < 0:
                    continue
                ranks = snaps[days[i]]
                cols.append(pd.Series(norm.ppf(1.0 - (ranks.to_numpy() - 0.5) / len(ranks)), index=ranks.index))
            if not cols:
                continue
            z = pd.concat(cols, axis=1).median(axis=1)
            out.append(pd.DataFrame({"game_id": dg["game_id"].to_numpy(),
                                     "massey_diff_a": dg["team_a"].map(z).to_numpy() - dg["team_b"].map(z).to_numpy()}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["game_id", "massey_diff_a"])


def seed_numbers(seeds: pd.DataFrame) -> pd.DataFrame:
    """Kaggle seeds ('W01', 'X16a') -> (season, team, seed number)."""
    return pd.DataFrame({"season": seeds["Season"].astype(int), "team": seeds["TeamID"].astype(int),
                         "seed": seeds["Seed"].str[1:3].astype(int)})
