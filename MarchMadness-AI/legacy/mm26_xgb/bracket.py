from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .modeling import ModelBundle, predict_matchups


ROUND_NAMES = {
    0: "First Four / Play-In",
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "National Championship",
}

ROUND_RE = re.compile(r"^R(\d+)")



def slot_round(slot: str) -> int:
    match = ROUND_RE.match(str(slot))
    if match:
        return int(match.group(1))
    return 0



def simulate_bracket(
    season: int,
    seeds: pd.DataFrame,
    slots: pd.DataFrame,
    team_features: pd.DataFrame,
    bundle: ModelBundle,
) -> pd.DataFrame:
    season_seeds = seeds.loc[seeds["Season"] == season].copy()
    season_slots = slots.loc[slots["Season"] == season].copy()
    if season_seeds.empty:
        raise ValueError(f"No seeds found for season {season}.")
    if season_slots.empty:
        raise ValueError(f"No slot data found for season {season}.")

    winners: dict[str, int] = dict(zip(season_seeds["Seed"].astype(str), season_seeds["TeamID"].astype(int)))
    unresolved = season_slots.copy()
    results: list[dict[str, object]] = []

    for _ in range(len(season_slots) + 10):
        progressed = False
        next_unresolved_rows = []

        for row in unresolved.itertuples(index=False):
            strong_key = str(row.StrongSeed)
            weak_key = str(row.WeakSeed)
            if strong_key in winners and weak_key in winners:
                team_a = winners[strong_key]
                team_b = winners[weak_key]
                pred = predict_matchups(
                    matchups=pd.DataFrame(
                        [{"Season": season, "TeamAID": team_a, "TeamBID": team_b}]
                    ),
                    team_features=team_features,
                    bundle=bundle,
                ).iloc[0]
                winner = int(pred["PredWinnerTeamID"])
                winners[str(row.Slot)] = winner
                results.append(
                    {
                        "Season": season,
                        "RoundNum": slot_round(str(row.Slot)),
                        "RoundName": ROUND_NAMES.get(slot_round(str(row.Slot)), f"Round {slot_round(str(row.Slot))}"),
                        "Slot": str(row.Slot),
                        "StrongSource": strong_key,
                        "WeakSource": weak_key,
                        "TeamAID": int(team_a),
                        "TeamBID": int(team_b),
                        "PredWinnerTeamID": winner,
                        "WinProbTeamA": float(pred["WinProbTeamA"]),
                        "WinProbTeamB": float(pred["WinProbTeamB"]),
                        "PredScoreTeamA": int(pred["PredScoreTeamA"]),
                        "PredScoreTeamB": int(pred["PredScoreTeamB"]),
                        "PredMargin": float(pred["PredMargin"]),
                        "PredTotal": float(pred["PredTotal"]),
                    }
                )
                progressed = True
            else:
                next_unresolved_rows.append(row)

        if not next_unresolved_rows:
            break
        unresolved = pd.DataFrame(next_unresolved_rows)
        if not progressed:
            missing_dependencies = unresolved[["Slot", "StrongSeed", "WeakSeed"]].copy()
            raise RuntimeError(
                "Could not resolve all bracket slots. Check seeds/slots files, especially play-in seeds.\n"
                + missing_dependencies.to_string(index=False)
            )

    result_df = pd.DataFrame(results).sort_values(["RoundNum", "Slot"]).reset_index(drop=True)
    return result_df
