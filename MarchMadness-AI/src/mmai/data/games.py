"""One row per game, from Kaggle's detailed results (box scores), oriented for prediction.

Columns:
- identity: season, daynum, date, phase (regular | conf_tourney | ncaa), w_*/l_* winner/loser fields
- tempo: possessions (``poss``, game average of both teams' estimates) and per 40 minutes (``poss40``)
- orientation: ``team_a``/``team_b`` with ``venue_a`` (+1 home, -1 away, 0 neutral). Team A is
  the home team when there is one, else the lower TeamID. The winner never decides
  orientation, which would leak the result into every metric.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from mmai.sources.kaggle import KaggleData

BOX = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]
FT_POSSESSION_COEF = 0.475


def possessions(fga, oreb, to, fta):
    return fga - oreb + to + FT_POSSESSION_COEF * fta


def kaggle_games(kd: KaggleData, gender: str) -> pd.DataFrame:
    """Regular season (incl. conference tournaments) + NCAA tournament, detailed results only."""
    parts = []
    for phase, name in (("regular", "RegularSeason"), ("ncaa", "NCAATourney")):
        if kd.has(f"{gender}{name}DetailedResults.csv"):
            d = kd.detailed(gender, name).copy()
            d["phase"] = phase
            parts.append(d)
    raw = pd.concat(parts, ignore_index=True)
    ct_name = f"{gender}ConferenceTourneyGames.csv"
    if kd.has(ct_name):
        ct = kd.read(ct_name)[["Season", "DayNum", "WTeamID", "LTeamID"]].drop_duplicates()
        ct["_ct"] = True
        raw = raw.merge(ct, on=["Season", "DayNum", "WTeamID", "LTeamID"], how="left")
        raw.loc[raw["_ct"].eq(True) & raw["phase"].eq("regular"), "phase"] = "conf_tourney"
        raw = raw.drop(columns="_ct")
    games = finalize_games(raw, kd.seasons(gender))
    conf_name = f"{gender}TeamConferences.csv"
    return add_context(games, kd.read(conf_name) if kd.has(conf_name) else None)


def add_context(g: pd.DataFrame, conferences: pd.DataFrame | None) -> pd.DataFrame:
    """Pre-game context known from the schedule alone: conferences, conference-game flag,
    days of rest since each team's previous game (capped at 7; season opener = 7)."""
    g = g.copy()
    if conferences is not None and len(conferences):
        conf = conferences.set_index(["Season", "TeamID"])["ConfAbbrev"]
        g["conf_a"] = pd.Series(list(zip(g["season"], g["team_a"]))).map(conf).to_numpy()
        g["conf_b"] = pd.Series(list(zip(g["season"], g["team_b"]))).map(conf).to_numpy()
        g["conf_game"] = (g["conf_a"] == g["conf_b"]) & g["phase"].eq("regular")
    else:
        g["conf_a"], g["conf_b"], g["conf_game"] = None, None, False
    # Keyed by (game, side), not (team, day): Kaggle has the odd team with two games on one day
    # (e.g. M 2026 DayNum 131, team 1361); the second of those gets 0 days of rest.
    long = pd.concat([g[["game_id", "season", "daynum", "team_a"]].rename(columns={"team_a": "team"}).assign(side="a"),
                      g[["game_id", "season", "daynum", "team_b"]].rename(columns={"team_b": "team"}).assign(side="b")],
                     ignore_index=True)
    long = long.sort_values(["season", "team", "daynum", "game_id"])
    long["rest"] = long.groupby(["season", "team"])["daynum"].diff().fillna(7).clip(upper=7)
    rest = long.set_index(["game_id", "side"])["rest"]
    g["rest_a"] = pd.Series(list(zip(g["game_id"], ["a"] * len(g)))).map(rest).to_numpy()
    g["rest_b"] = pd.Series(list(zip(g["game_id"], ["b"] * len(g)))).map(rest).to_numpy()
    return g


def finalize_games(raw: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """Kaggle-schema rows (Season, DayNum, W*/L* ...) -> the unified, oriented game table."""
    g = pd.DataFrame({
        "season": raw["Season"].astype(int),
        "daynum": raw["DayNum"].astype(int),
        "phase": raw["phase"].to_numpy(),
        "w_team": raw["WTeamID"].astype(int),
        "l_team": raw["LTeamID"].astype(int),
        "w_score": raw["WScore"].astype(int),
        "l_score": raw["LScore"].astype(int),
        "w_loc": raw["WLoc"].to_numpy(),
        "numot": raw["NumOT"].fillna(0).astype(int),
    })
    for side in ("W", "L"):
        for c in BOX:
            g[f"{side.lower()}_{c}"] = raw[f"{side}{c}"].to_numpy()
    poss_w = possessions(g["w_FGA"], g["w_OR"], g["w_TO"], g["w_FTA"])
    poss_l = possessions(g["l_FGA"], g["l_OR"], g["l_TO"], g["l_FTA"])
    g["poss"] = (poss_w + poss_l) / 2.0
    g["poss40"] = g["poss"] * 40.0 / (40.0 + 5.0 * g["numot"])
    zero = {int(s): datetime.strptime(str(d), "%m/%d/%Y").date() for s, d in zip(seasons["Season"], seasons["DayZero"])}
    g["date"] = [zero[s] + timedelta(days=int(n)) for s, n in zip(g["season"], g["daynum"])]
    g["neutral"] = g["w_loc"].eq("N")

    home_w = g["w_loc"].eq("H")
    home_l = g["w_loc"].eq("A")
    lower_w = g["w_team"] < g["l_team"]
    a_is_w = home_w | (~home_l & lower_w)
    g["team_a"] = np.where(a_is_w, g["w_team"], g["l_team"])
    g["team_b"] = np.where(a_is_w, g["l_team"], g["w_team"])
    g["score_a"] = np.where(a_is_w, g["w_score"], g["l_score"])
    g["score_b"] = np.where(a_is_w, g["l_score"], g["w_score"])
    g["venue_a"] = np.where(g["neutral"], 0, 1).astype(int)  # A is home whenever there is a home team
    g["margin_a"] = g["score_a"] - g["score_b"]
    g["total"] = g["score_a"] + g["score_b"]
    g = g.sort_values(["season", "daynum", "team_a"]).reset_index(drop=True)
    g.insert(0, "game_id", [f"{s}_{d}_{a}_{b}" for s, d, a, b in zip(g["season"], g["daynum"], g["team_a"], g["team_b"])])
    return g
