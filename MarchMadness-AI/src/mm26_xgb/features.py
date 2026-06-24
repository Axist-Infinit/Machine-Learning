from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TEAM_GAME_COLUMNS = [
    "Season",
    "DayNum",
    "TeamID",
    "OppTeamID",
    "Site",
    "IsHome",
    "IsAway",
    "IsNeutral",
    "NumOT",
    "IsWin",
    "Score",
    "OppScore",
    "Margin",
    "FGM",
    "FGA",
    "FGM3",
    "FGA3",
    "FTM",
    "FTA",
    "OR",
    "DR",
    "Ast",
    "TO",
    "Stl",
    "Blk",
    "PF",
    "OppFGM",
    "OppFGA",
    "OppFGM3",
    "OppFGA3",
    "OppFTM",
    "OppFTA",
    "OppOR",
    "OppDR",
    "OppAst",
    "OppTO",
    "OppStl",
    "OppBlk",
    "OppPF",
]


@dataclass(slots=True)
class FeatureArtifacts:
    team_features: pd.DataFrame
    feature_columns: list[str]

MATCHUP_CONTEXT_EXCLUDE = {
    "Season",
    "TeamAID",
    "TeamBID",
    "TeamAScore",
    "TeamBScore",
    "TeamAWin",
    "Margin",
    "Total",
    "EventID",
    "CommenceTime",
    "TeamA",
    "TeamB",
    "TeamAName",
    "TeamBName",
    "MarketProbTeamA",
    "MarketProbTeamB",
    "MarketMoneylineTeamA",
    "MarketMoneylineTeamB",
    "MarketSpreadTeamA",
    "MarketSpreadPriceA",
    "MarketSpreadPriceB",
    "MarketTotal",
    "MarketOverPrice",
    "MarketUnderPrice",
    "MarketBookCount",
    "MoneylineHold",
    "SpreadHold",
    "TotalHold",
    "HasMarketProb",
    "HasMarketSpread",
    "HasMarketTotal",
    "WinBaseMargin",
    "MarginBase",
    "TotalBase",
}



def _safe_divide(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    den = np.where(np.asarray(den) == 0, np.nan, den)
    return num / den



def _flip_site(site: str) -> str:
    if site == "H":
        return "A"
    if site == "A":
        return "H"
    return "N"



def make_team_game_rows(detailed_results: pd.DataFrame) -> pd.DataFrame:
    winners = pd.DataFrame(
        {
            "Season": detailed_results["Season"],
            "DayNum": detailed_results["DayNum"],
            "TeamID": detailed_results["WTeamID"],
            "OppTeamID": detailed_results["LTeamID"],
            "Site": detailed_results["WLoc"].fillna("N"),
            "IsHome": (detailed_results["WLoc"] == "H").astype(int),
            "IsAway": (detailed_results["WLoc"] == "A").astype(int),
            "IsNeutral": (detailed_results["WLoc"] == "N").astype(int),
            "NumOT": detailed_results["NumOT"],
            "IsWin": 1,
            "Score": detailed_results["WScore"],
            "OppScore": detailed_results["LScore"],
            "Margin": detailed_results["WScore"] - detailed_results["LScore"],
            "FGM": detailed_results["WFGM"],
            "FGA": detailed_results["WFGA"],
            "FGM3": detailed_results["WFGM3"],
            "FGA3": detailed_results["WFGA3"],
            "FTM": detailed_results["WFTM"],
            "FTA": detailed_results["WFTA"],
            "OR": detailed_results["WOR"],
            "DR": detailed_results["WDR"],
            "Ast": detailed_results["WAst"],
            "TO": detailed_results["WTO"],
            "Stl": detailed_results["WStl"],
            "Blk": detailed_results["WBlk"],
            "PF": detailed_results["WPF"],
            "OppFGM": detailed_results["LFGM"],
            "OppFGA": detailed_results["LFGA"],
            "OppFGM3": detailed_results["LFGM3"],
            "OppFGA3": detailed_results["LFGA3"],
            "OppFTM": detailed_results["LFTM"],
            "OppFTA": detailed_results["LFTA"],
            "OppOR": detailed_results["LOR"],
            "OppDR": detailed_results["LDR"],
            "OppAst": detailed_results["LAst"],
            "OppTO": detailed_results["LTO"],
            "OppStl": detailed_results["LStl"],
            "OppBlk": detailed_results["LBlk"],
            "OppPF": detailed_results["LPF"],
        }
    )

    losers = pd.DataFrame(
        {
            "Season": detailed_results["Season"],
            "DayNum": detailed_results["DayNum"],
            "TeamID": detailed_results["LTeamID"],
            "OppTeamID": detailed_results["WTeamID"],
            "Site": detailed_results["WLoc"].fillna("N").map(_flip_site),
            "IsHome": (detailed_results["WLoc"] == "A").astype(int),
            "IsAway": (detailed_results["WLoc"] == "H").astype(int),
            "IsNeutral": (detailed_results["WLoc"] == "N").astype(int),
            "NumOT": detailed_results["NumOT"],
            "IsWin": 0,
            "Score": detailed_results["LScore"],
            "OppScore": detailed_results["WScore"],
            "Margin": detailed_results["LScore"] - detailed_results["WScore"],
            "FGM": detailed_results["LFGM"],
            "FGA": detailed_results["LFGA"],
            "FGM3": detailed_results["LFGM3"],
            "FGA3": detailed_results["LFGA3"],
            "FTM": detailed_results["LFTM"],
            "FTA": detailed_results["LFTA"],
            "OR": detailed_results["LOR"],
            "DR": detailed_results["LDR"],
            "Ast": detailed_results["LAst"],
            "TO": detailed_results["LTO"],
            "Stl": detailed_results["LStl"],
            "Blk": detailed_results["LBlk"],
            "PF": detailed_results["LPF"],
            "OppFGM": detailed_results["WFGM"],
            "OppFGA": detailed_results["WFGA"],
            "OppFGM3": detailed_results["WFGM3"],
            "OppFGA3": detailed_results["WFGA3"],
            "OppFTM": detailed_results["WFTM"],
            "OppFTA": detailed_results["WFTA"],
            "OppOR": detailed_results["WOR"],
            "OppDR": detailed_results["WDR"],
            "OppAst": detailed_results["WAst"],
            "OppTO": detailed_results["WTO"],
            "OppStl": detailed_results["WStl"],
            "OppBlk": detailed_results["WBlk"],
            "OppPF": detailed_results["WPF"],
        }
    )

    out = pd.concat([winners, losers], ignore_index=True)
    return out[TEAM_GAME_COLUMNS].sort_values(["Season", "TeamID", "DayNum"]).reset_index(drop=True)



def add_derived_metrics(team_games: pd.DataFrame) -> pd.DataFrame:
    df = team_games.copy()

    team_poss = df["FGA"] - df["OR"] + df["TO"] + 0.475 * df["FTA"]
    opp_poss = df["OppFGA"] - df["OppOR"] + df["OppTO"] + 0.475 * df["OppFTA"]
    poss = (team_poss + opp_poss) / 2.0

    df["Poss"] = poss
    df["Pace"] = poss
    df["FGPct"] = _safe_divide(df["FGM"], df["FGA"])
    df["FG3Pct"] = _safe_divide(df["FGM3"], df["FGA3"])
    df["FTPct"] = _safe_divide(df["FTM"], df["FTA"])
    df["eFG"] = _safe_divide(df["FGM"] + 0.5 * df["FGM3"], df["FGA"])
    df["OppFGPct"] = _safe_divide(df["OppFGM"], df["OppFGA"])
    df["OppFG3Pct"] = _safe_divide(df["OppFGM3"], df["OppFGA3"])
    df["OppeFG"] = _safe_divide(df["OppFGM"] + 0.5 * df["OppFGM3"], df["OppFGA"])
    df["ThreeRate"] = _safe_divide(df["FGA3"], df["FGA"])
    df["FTRate"] = _safe_divide(df["FTA"], df["FGA"])
    df["ORBRate"] = _safe_divide(df["OR"], df["OR"] + df["OppDR"])
    df["DRBRate"] = _safe_divide(df["DR"], df["DR"] + df["OppOR"])
    df["TRBRate"] = _safe_divide(df["OR"] + df["DR"], df["OR"] + df["DR"] + df["OppOR"] + df["OppDR"])
    df["TOVRate"] = _safe_divide(df["TO"], df["Poss"])
    df["ASTTOV"] = _safe_divide(df["Ast"], df["TO"])
    df["TS"] = _safe_divide(df["Score"], 2 * (df["FGA"] + 0.44 * df["FTA"]))
    df["OffRating"] = 100 * _safe_divide(df["Score"], df["Poss"])
    df["DefRating"] = 100 * _safe_divide(df["OppScore"], df["Poss"])
    df["NetRating"] = df["OffRating"] - df["DefRating"]

    numeric_cols = [c for c in df.columns if c not in {"Site"}]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df



def compute_elo_features(regular_results: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(regular_results["Season"].unique())
    rows: list[dict[str, float | int]] = []

    for season in seasons:
        season_games = regular_results.loc[regular_results["Season"] == season].sort_values(
            ["DayNum", "WTeamID", "LTeamID"]
        )
        ratings: dict[int, float] = {}
        peaks: dict[int, float] = {}

        def get_rating(team_id: int) -> float:
            if team_id not in ratings:
                ratings[team_id] = 1500.0
                peaks[team_id] = 1500.0
            return ratings[team_id]

        for game in season_games.itertuples(index=False):
            wteam = int(game.WTeamID)
            lteam = int(game.LTeamID)
            w_elo = get_rating(wteam)
            l_elo = get_rating(lteam)
            site = getattr(game, "WLoc", "N")

            if site == "H":
                home_adv = 65.0
            elif site == "A":
                home_adv = -65.0
            else:
                home_adv = 0.0

            expected_w = 1.0 / (1.0 + 10 ** ((l_elo - (w_elo + home_adv)) / 400.0))
            margin = max(1, int(game.WScore) - int(game.LScore))
            multiplier = ((margin + 3.0) ** 0.8) / (7.5 + 0.006 * abs((w_elo + home_adv) - l_elo))
            k = 20.0 * multiplier
            delta = k * (1.0 - expected_w)

            ratings[wteam] = w_elo + delta
            ratings[lteam] = l_elo - delta
            peaks[wteam] = max(peaks[wteam], ratings[wteam])
            peaks[lteam] = max(peaks[lteam], ratings[lteam])

        for team_id, elo in ratings.items():
            rows.append(
                {
                    "Season": season,
                    "TeamID": team_id,
                    "EloEnd": elo,
                    "EloPeak": peaks[team_id],
                }
            )

    return pd.DataFrame(rows)



def _aggregate_recent(team_games: pd.DataFrame, n: int) -> pd.DataFrame:
    cols = [
        "Margin",
        "OffRating",
        "DefRating",
        "NetRating",
        "Score",
        "OppScore",
        "IsWin",
        "Pace",
        "eFG",
        "ThreeRate",
        "FTRate",
        "ORBRate",
        "TOVRate",
        "TS",
    ]
    recent = (
        team_games.sort_values(["Season", "TeamID", "DayNum"])
        .groupby(["Season", "TeamID"], as_index=False, group_keys=False)
        .tail(n)
        .groupby(["Season", "TeamID"], as_index=False)[cols]
        .mean()
    )
    rename = {col: f"Last{n}{col}" for col in cols}
    recent = recent.rename(columns=rename)
    recent = recent.rename(columns={f"Last{n}IsWin": f"Last{n}WinPct"})
    return recent



def _aggregate_ewm(team_games: pd.DataFrame, span: int = 8) -> pd.DataFrame:
    cols = [
        "Margin",
        "OffRating",
        "DefRating",
        "NetRating",
        "Pace",
        "Score",
        "OppScore",
        "IsWin",
        "eFG",
        "ThreeRate",
        "FTRate",
        "ORBRate",
        "TOVRate",
        "TS",
    ]
    df = team_games.sort_values(["Season", "TeamID", "DayNum"]).copy()

    for col in cols:
        df[f"EWM{span}_{col}"] = (
            df.groupby(["Season", "TeamID"], group_keys=False)[col]
            .transform(lambda s: s.ewm(span=span, adjust=False).mean())
        )

    keep_cols = ["Season", "TeamID", *[f"EWM{span}_{col}" for col in cols]]
    out = df.groupby(["Season", "TeamID"], as_index=False).tail(1)[keep_cols].copy()
    out = out.rename(columns={f"EWM{span}_IsWin": f"EWM{span}_WinPct"})
    return out.reset_index(drop=True)



def _aggregate_volatility(team_games: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Pace",
        "OffRating",
        "DefRating",
        "NetRating",
        "eFG",
        "ThreeRate",
        "FTRate",
        "ORBRate",
        "TOVRate",
        "TS",
    ]
    out = team_games.groupby(["Season", "TeamID"], as_index=False)[cols].std()
    out = out.rename(columns={col: f"{col}Std" for col in cols})
    return out.reset_index(drop=True)



def _aggregate_site_split(team_games: pd.DataFrame, site_flag: str, output_name: str) -> pd.DataFrame:
    subset = team_games.loc[team_games[site_flag] == 1, ["Season", "TeamID", "IsWin"]]
    if subset.empty:
        return pd.DataFrame(columns=["Season", "TeamID", output_name])
    out = subset.groupby(["Season", "TeamID"], as_index=False)["IsWin"].mean()
    return out.rename(columns={"IsWin": output_name})



def _aggregate_massey(massey: pd.DataFrame | None, cutoff_daynum: int) -> pd.DataFrame:
    if massey is None or massey.empty:
        return pd.DataFrame(columns=["Season", "TeamID"])
    df = massey.loc[massey["RankingDayNum"] <= cutoff_daynum].copy()
    if df.empty:
        return pd.DataFrame(columns=["Season", "TeamID"])
    df = df.sort_values(["Season", "SystemName", "TeamID", "RankingDayNum"])
    df = df.groupby(["Season", "SystemName", "TeamID"], as_index=False).tail(1)
    out = df.groupby(["Season", "TeamID"], as_index=False).agg(
        MasseyMeanRank=("OrdinalRank", "mean"),
        MasseyMedianRank=("OrdinalRank", "median"),
        MasseyBestRank=("OrdinalRank", "min"),
        MasseyWorstRank=("OrdinalRank", "max"),
        MasseyStdRank=("OrdinalRank", "std"),
        MasseySystems=("OrdinalRank", "count"),
    )
    return out



def build_team_features(
    regular_results: pd.DataFrame,
    seeds: pd.DataFrame,
    massey: pd.DataFrame | None = None,
    cutoff_daynum: int = 133,
) -> pd.DataFrame:
    regular_filtered = regular_results.loc[regular_results["DayNum"] <= cutoff_daynum].copy()
    team_games = add_derived_metrics(make_team_game_rows(regular_filtered))

    base = (
        team_games.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            Games=("IsWin", "size"),
            Wins=("IsWin", "sum"),
            ScoreMean=("Score", "mean"),
            OppScoreMean=("OppScore", "mean"),
            MarginMean=("Margin", "mean"),
            MarginStd=("Margin", "std"),
            PaceMean=("Pace", "mean"),
            OffRatingMean=("OffRating", "mean"),
            DefRatingMean=("DefRating", "mean"),
            NetRatingMean=("NetRating", "mean"),
            eFGMean=("eFG", "mean"),
            FG3PctMean=("FG3Pct", "mean"),
            FTPctMean=("FTPct", "mean"),
            ThreeRateMean=("ThreeRate", "mean"),
            FTRateMean=("FTRate", "mean"),
            ORBRateMean=("ORBRate", "mean"),
            DRBRateMean=("DRBRate", "mean"),
            TRBRateMean=("TRBRate", "mean"),
            TOVRateMean=("TOVRate", "mean"),
            ASTTOVMean=("ASTTOV", "mean"),
            TSMean=("TS", "mean"),
            OppeFGMean=("OppeFG", "mean"),
            OppFG3PctMean=("OppFG3Pct", "mean"),
            AstMean=("Ast", "mean"),
            TOMean=("TO", "mean"),
            StlMean=("Stl", "mean"),
            BlkMean=("Blk", "mean"),
            PFMean=("PF", "mean"),
            NumOTMean=("NumOT", "mean"),
            NeutralGamePct=("IsNeutral", "mean"),
        )
    )
    base["Losses"] = base["Games"] - base["Wins"]
    base["WinPct"] = _safe_divide(base["Wins"], base["Games"])

    recent5 = _aggregate_recent(team_games, n=5)
    recent10 = _aggregate_recent(team_games, n=10)
    ewm8 = _aggregate_ewm(team_games, span=8)
    volatility = _aggregate_volatility(team_games)
    home = _aggregate_site_split(team_games, "IsHome", "HomeWinPct")
    away = _aggregate_site_split(team_games, "IsAway", "AwayWinPct")
    neutral = _aggregate_site_split(team_games, "IsNeutral", "NeutralWinPct")
    elo = compute_elo_features(regular_filtered)
    massey_features = _aggregate_massey(massey, cutoff_daynum=cutoff_daynum)

    out = base.merge(recent5, on=["Season", "TeamID"], how="left")
    out = out.merge(recent10, on=["Season", "TeamID"], how="left")
    out = out.merge(home, on=["Season", "TeamID"], how="left")
    out = out.merge(away, on=["Season", "TeamID"], how="left")
    out = out.merge(neutral, on=["Season", "TeamID"], how="left")
    out = out.merge(elo, on=["Season", "TeamID"], how="left")
    out = out.merge(massey_features, on=["Season", "TeamID"], how="left")
    out = out.merge(volatility, on=["Season", "TeamID"], how="left")

    schedule_proxy = team_games[["Season", "TeamID", "OppTeamID"]].merge(
        out[["Season", "TeamID", "WinPct", "NetRatingMean"]].rename(
            columns={
                "TeamID": "OppTeamID",
                "WinPct": "OppTeamWinPct",
                "NetRatingMean": "OppTeamNetRating",
            }
        ),
        on=["Season", "OppTeamID"],
        how="left",
    )

    sos = schedule_proxy.groupby(["Season", "TeamID"], as_index=False).agg(
        SOSOppWinPct=("OppTeamWinPct", "mean"),
        SOSOppNetRating=("OppTeamNetRating", "mean"),
    )
    out = out.merge(sos, on=["Season", "TeamID"], how="left")

    form_pairs = [
        ("Last5Pace", "PaceMean", "PaceForm5"),
        ("Last10Pace", "PaceMean", "PaceForm10"),
        ("Last5NetRating", "NetRatingMean", "NetRatingForm5"),
        ("Last10NetRating", "NetRatingMean", "NetRatingForm10"),
        ("Last5ThreeRate", "ThreeRateMean", "ThreeRateForm5"),
        ("Last5FTRate", "FTRateMean", "FTRateForm5"),
        ("Last5ORBRate", "ORBRateMean", "ORBRateForm5"),
        ("Last5TOVRate", "TOVRateMean", "TOVRateForm5"),
        ("Last5TS", "TSMean", "TSForm5"),
        ("EWM8_Pace", "PaceMean", "PaceTrend8"),
        ("EWM8_NetRating", "NetRatingMean", "NetRatingTrend8"),
        ("EWM8_ThreeRate", "ThreeRateMean", "ThreeRateTrend8"),
        ("EWM8_FTRate", "FTRateMean", "FTRateTrend8"),
    ]
    for recent_col, base_col, out_col in form_pairs:
        if recent_col in out.columns and base_col in out.columns:
            out[out_col] = pd.to_numeric(out[recent_col], errors="coerce") - pd.to_numeric(out[base_col], errors="coerce")

    if {"PaceMean", "NetRatingMean"}.issubset(out.columns):
        out["TempoStrengthInteraction"] = pd.to_numeric(out["PaceMean"], errors="coerce") * pd.to_numeric(out["NetRatingMean"], errors="coerce")
    if {"PaceStd", "NetRatingStd"}.issubset(out.columns):
        out["TempoVolatilityInteraction"] = pd.to_numeric(out["PaceStd"], errors="coerce") * pd.to_numeric(out["NetRatingStd"], errors="coerce")

    seed_cols = ["Season", "TeamID", "Seed", "SeedNum", "SeedRegion", "IsPlayInSeed"]
    out = out.merge(seeds[seed_cols], on=["Season", "TeamID"], how="left")
    out["SeedNum"] = out["SeedNum"].fillna(17)
    out["IsPlayInSeed"] = out["IsPlayInSeed"].fillna(0)
    out["HasSeed"] = out["Seed"].notna().astype(int)
    out["SeedRegion"] = out["SeedRegion"].fillna("Z")

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    out = out.sort_values(["Season", "TeamID"]).reset_index(drop=True)
    return out



def build_matchup_frame(
    matchups: pd.DataFrame,
    team_features: pd.DataFrame,
    feature_prefix_a: str = "A_",
    feature_prefix_b: str = "B_",
) -> tuple[pd.DataFrame, list[str]]:
    required_cols = {"Season", "TeamAID", "TeamBID"}
    missing = required_cols - set(matchups.columns)
    if missing:
        raise ValueError(f"matchups missing required columns: {sorted(missing)}")

    left = team_features.rename(
        columns={col: f"{feature_prefix_a}{col}" for col in team_features.columns if col not in {"Season", "TeamID"}}
    )
    right = team_features.rename(
        columns={col: f"{feature_prefix_b}{col}" for col in team_features.columns if col not in {"Season", "TeamID"}}
    )

    merged = matchups.merge(
        left,
        left_on=["Season", "TeamAID"],
        right_on=["Season", "TeamID"],
        how="left",
    ).drop(columns=["TeamID"])

    merged = merged.merge(
        right,
        left_on=["Season", "TeamBID"],
        right_on=["Season", "TeamID"],
        how="left",
        suffixes=("", "_dup"),
    ).drop(columns=["TeamID"])

    numeric_feature_cols = [
        col
        for col in team_features.columns
        if col not in {"Season", "TeamID", "Seed", "SeedRegion"}
        and pd.api.types.is_numeric_dtype(team_features[col])
    ]

    diff_df = pd.DataFrame(
        {
            f"Diff_{col}": merged[f"{feature_prefix_a}{col}"] - merged[f"{feature_prefix_b}{col}"]
            for col in numeric_feature_cols
        }
    )
    avg_df = pd.DataFrame(
        {
            f"Avg_{col}": (merged[f"{feature_prefix_a}{col}"] + merged[f"{feature_prefix_b}{col}"]) / 2.0
            for col in numeric_feature_cols
        }
    )
    merged = pd.concat([merged, diff_df, avg_df], axis=1)
    diff_cols = list(diff_df.columns)
    avg_cols = list(avg_df.columns)

    region_map = {"W": 1, "X": 2, "Y": 3, "Z": 4, "A": 5, "B": 6, "C": 7, "D": 8}
    merged["A_SeedRegionCode"] = merged[f"{feature_prefix_a}SeedRegion"].astype(str).str[0].map(region_map).fillna(0)
    merged["B_SeedRegionCode"] = merged[f"{feature_prefix_b}SeedRegion"].astype(str).str[0].map(region_map).fillna(0)
    merged["Diff_SeedRegionCode"] = merged["A_SeedRegionCode"] - merged["B_SeedRegionCode"]

    feature_cols = diff_cols + avg_cols + [
        f"{feature_prefix_a}SeedNum",
        f"{feature_prefix_b}SeedNum",
        f"{feature_prefix_a}HasSeed",
        f"{feature_prefix_b}HasSeed",
        f"{feature_prefix_a}IsPlayInSeed",
        f"{feature_prefix_b}IsPlayInSeed",
        "Diff_SeedRegionCode",
    ]

    context_feature_cols = [
        col
        for col in matchups.columns
        if col not in MATCHUP_CONTEXT_EXCLUDE and pd.api.types.is_numeric_dtype(matchups[col])
    ]
    feature_cols = feature_cols + context_feature_cols

    merged[feature_cols] = merged[feature_cols].replace([np.inf, -np.inf], np.nan)
    return merged, feature_cols



def team_features_for_season(team_features: pd.DataFrame, season: int) -> pd.DataFrame:
    return team_features.loc[team_features["Season"] == season].copy()
