from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .data import normalize_team_name


KEY_TEAM_FILE = "KenPom Barttorvik.csv"
AP_POLL_FILE = "AP Poll Data.csv"
MATCHUP_FILE = "Tournament Matchups.csv"
LOCATION_FILE = "Tournament Locations.csv"


TEAM_SOURCE_COLUMNS: dict[str, list[str]] = {
    "KenPom Barttorvik.csv": [
        "KADJ T",
        "KADJ O",
        "KADJ D",
        "KADJ EM",
        "BADJ EM",
        "BADJ O",
        "BADJ D",
        "BARTHAG",
        "GAMES",
        "W",
        "L",
        "WIN%",
        "EFG%",
        "EFG%D",
        "FTR",
        "FTRD",
        "TOV%",
        "TOV%D",
        "OREB%",
        "DREB%",
        "OP OREB%",
        "OP DREB%",
        "RAW T",
        "2PT%",
        "2PT%D",
        "3PT%",
        "3PT%D",
        "BLK%",
        "BLKED%",
        "AST%",
        "OP AST%",
        "2PTR",
        "3PTR",
        "2PTRD",
        "3PTRD",
        "BADJ T",
        "AVG HGT",
        "EFF HGT",
        "EXP",
        "TALENT",
        "FT%",
        "OP FT%",
        "PPPO",
        "PPPD",
        "ELITE SOS",
        "WAB",
        "KADJ O RANK",
        "KADJ D RANK",
        "KADJ EM RANK",
        "BADJ EM RANK",
        "BARTHAG RANK",
        "PPPO RANK",
        "PPPD RANK",
        "ELITE SOS RANK",
    ],
    "Barttorvik Neutral.csv": [
        "BADJ EM",
        "BADJ O",
        "BADJ D",
        "BARTHAG",
        "G",
        "W",
        "L",
        "WIN%",
        "EFG%",
        "EFG%D",
        "FTR",
        "FTRD",
        "TOV%",
        "TOV%D",
        "OREB%",
        "DREB%",
        "OP OREB%",
        "OP DREB%",
        "2PT%",
        "2PT%D",
        "3PT%",
        "3PT%D",
        "AST%",
        "OP AST%",
        "BADJ T",
        "PPPO",
        "PPPD",
        "ELITE SOS",
    ],
    "538 Ratings.csv": ["POWER RATING", "POWER RATING RANK"],
    "EvanMiya.csv": [
        "O RATE",
        "D RATE",
        "RELATIVE RATING",
        "OPPONENT ADJUST",
        "PACE ADJUST",
        "TRUE TEMPO",
        "HOME RANK",
        "KILLSHOTS PER GAME",
        "KILL SHOTS CONCEDED PER GAME",
        "KILLSHOTS MARGIN",
        "RELATIVE RATING RANK",
        "OPPONENT ADJUST RANK",
        "PACE ADJUST RANK",
    ],
    "KenPom Preseason.csv": [
        "PRESEASON KADJ EM",
        "PRESEASON KADJ O",
        "PRESEASON KADJ D",
        "PRESEASON KADJ T",
        "KADJ EM CHANGE",
        "KADJ T CHANGE",
        "PRESEASON KADJ EM RANK",
    ],
    "RPPF Ratings.csv": [
        "RPPF RATING",
        "NPB RATING",
        "RADJ O",
        "RADJ D",
        "RADJ EM",
        "R PACE",
        "R SOS",
        "STROE",
        "STRDE",
        "STREM",
        "RPPF RATING RANK",
        "RADJ EM RANK",
        "R SOS RANK",
    ],
    "RPPF Preseason Ratings.csv": [
        "PRESEASON RPPF RATING",
        "RPPF RATING CHANGE",
        "RPPF PRESEASON RANK",
    ],
    "Resumes.csv": [
        "NET RPI",
        "RESUME",
        "WAB RANK",
        "ELO",
        "B POWER",
        "Q1 W",
        "Q2 W",
        "Q1 PLUS Q2 W",
        "Q3 Q4 L",
        "PLUS 500",
        "R SCORE",
    ],
    "Shooting Splits.csv": [
        "DUNKS FG%",
        "DUNKS SHARE",
        "DUNKS FG%D",
        "DUNKS D SHARE",
        "CLOSE TWOS FG%",
        "CLOSE TWOS SHARE",
        "CLOSE TWOS FG%D",
        "CLOSE TWOS D SHARE",
        "FARTHER TWOS FG%",
        "FARTHER TWOS SHARE",
        "FARTHER TWOS FG%D",
        "FARTHER TWOS D SHARE",
        "THREES FG%",
        "THREES SHARE",
        "THREES FG%D",
        "THREES D SHARE",
    ],
    "TeamRankings.csv": [
        "TR RATING",
        "V 1-25 WINS",
        "V 1-25 LOSS",
        "V 26-50 WINS",
        "V 26-50 LOSS",
        "V 51-100 WINS",
        "V 51-100 LOSS",
        "HI",
        "LO",
        "LAST",
        "SOS RANK",
        "SOS RATING",
        "SOS HI",
        "SOS LO",
        "SOS LAST",
        "LUCK RANK",
        "LUCK RATING",
        "CONSISTENCY RANK",
        "CONSISTENCY TR RATING",
    ],
    "TeamRankings Neutral.csv": [
        "TR RATING",
        "V 1-25 WINS",
        "V 1-25 LOSS",
        "V 26-50 WINS",
        "V 26-50 LOSS",
        "V 51-100 WINS",
        "V 51-100 LOSS",
        "HI",
        "LO",
        "LAST",
    ],
}

CONF_SOURCE_COLUMNS: dict[str, list[str]] = {
    "Conference Stats.csv": ["BADJ EM", "BADJ O", "BADJ D", "BARTHAG"],
    "Conference Stats Neutral.csv": ["BADJ EM", "BADJ O", "BADJ D", "BARTHAG"],
    "RPPF Conference Ratings.csv": ["RPPF RATING", "NPB RATING"],
}

OPTIONAL_TEAM_MISSING_THRESHOLD = 0.65


class AttachedDataError(RuntimeError):
    pass


@dataclass(slots=True)
class AttachedDataBundle:
    team_features: pd.DataFrame
    tournament_rows: pd.DataFrame
    team_name_map: dict[int, str]
    season_team_lookup: dict[tuple[int, str], int]



def _slug(name: str) -> str:
    stem = os.path.splitext(name)[0]
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").upper()



def _require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    return path



def _normalize_name(value: str) -> str:
    return normalize_team_name(value)



def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)



def _base_keys(data_dir: Path) -> pd.DataFrame:
    df = _load_csv(_require_path(data_dir / KEY_TEAM_FILE))
    required = {"YEAR", "TEAM NO", "TEAM", "SEED", "CONF", "CONF ID"}
    missing = required - set(df.columns)
    if missing:
        raise AttachedDataError(f"{KEY_TEAM_FILE} is missing columns: {sorted(missing)}")
    out = df[["YEAR", "TEAM NO", "TEAM", "SEED", "CONF", "CONF ID"]].drop_duplicates().copy()
    out["YEAR"] = out["YEAR"].astype(int)
    out["TEAM NO"] = out["TEAM NO"].astype(int)
    return out.sort_values(["YEAR", "TEAM NO"]).reset_index(drop=True)



def _team_key_lookup(base_keys: pd.DataFrame) -> dict[tuple[int, str], int]:
    lookup: dict[tuple[int, str], int] = {}
    for row in base_keys[["YEAR", "TEAM NO", "TEAM"]].to_dict("records"):
        lookup[(int(row["YEAR"]), _normalize_name(str(row["TEAM"]))) ] = int(row["TEAM NO"])
    return lookup



def _resolve_team_no(df: pd.DataFrame, base_keys: pd.DataFrame) -> pd.DataFrame:
    if "TEAM NO" not in df.columns or "YEAR" not in df.columns:
        return df
    if not df["TEAM NO"].isna().any():
        out = df.copy()
        out["TEAM NO"] = out["TEAM NO"].astype(int)
        return out
    if "TEAM" not in df.columns:
        return df
    lookup = _team_key_lookup(base_keys)
    out = df.copy()

    def _resolve_row(row: pd.Series) -> float:
        if pd.notna(row["TEAM NO"]):
            return float(int(row["TEAM NO"]))
        key = (int(row["YEAR"]), _normalize_name(str(row["TEAM"])))
        value = lookup.get(key)
        return float(value) if value is not None else np.nan

    out["TEAM NO"] = out.apply(_resolve_row, axis=1)
    out = out.loc[out["TEAM NO"].notna()].copy()
    out["TEAM NO"] = out["TEAM NO"].astype(int)
    return out



def _load_team_source(data_dir: Path, fname: str, cols: Iterable[str], base_keys: pd.DataFrame) -> pd.DataFrame:
    path = data_dir / fname
    if not path.exists():
        return pd.DataFrame(columns=["YEAR", "TEAM NO"])
    df = _resolve_team_no(_load_csv(path), base_keys)
    if {"YEAR", "TEAM NO"} - set(df.columns):
        return pd.DataFrame(columns=["YEAR", "TEAM NO"])

    keep = [col for col in cols if col in df.columns]
    if not keep:
        return pd.DataFrame(columns=["YEAR", "TEAM NO"])

    key_cols = ["YEAR", "TEAM NO"]
    if df.duplicated(key_cols).any():
        out = df.groupby(key_cols, as_index=False)[keep].mean(numeric_only=True)
    else:
        out = df[key_cols + keep].copy()
    prefix = _slug(fname)
    return out.rename(columns={col: f"{prefix}_{col}" for col in keep})



def _load_conf_source(data_dir: Path, fname: str, cols: Iterable[str]) -> pd.DataFrame:
    path = data_dir / fname
    if not path.exists():
        return pd.DataFrame(columns=["YEAR", "CONF ID"])
    df = _load_csv(path)
    required = {"YEAR", "CONF ID"}
    if required - set(df.columns):
        return pd.DataFrame(columns=["YEAR", "CONF ID"])
    keep = [col for col in cols if col in df.columns]
    if not keep:
        return pd.DataFrame(columns=["YEAR", "CONF ID"])
    prefix = _slug(fname)
    return df[["YEAR", "CONF ID", *keep]].rename(columns={col: f"{prefix}_{col}" for col in keep})



def _aggregate_ap_poll(data_dir: Path, base_keys: pd.DataFrame) -> pd.DataFrame:
    path = data_dir / AP_POLL_FILE
    if not path.exists():
        return pd.DataFrame(columns=["YEAR", "TEAM NO"])
    df = _resolve_team_no(_load_csv(path), base_keys)
    required = {"YEAR", "TEAM NO", "WEEK", "AP VOTES", "AP RANK"}
    if required - set(df.columns):
        return pd.DataFrame(columns=["YEAR", "TEAM NO"])
    df = df.sort_values(["YEAR", "TEAM NO", "WEEK"]).copy()
    if "RANK?" not in df.columns:
        df["RANK?"] = df["AP RANK"].notna().astype(int)
    if "W" not in df.columns:
        df["W"] = np.nan
    if "L" not in df.columns:
        df["L"] = np.nan
    out = df.groupby(["YEAR", "TEAM NO"], as_index=False).agg(
        AP_Weeks=("WEEK", "nunique"),
        AP_VotesMean=("AP VOTES", "mean"),
        AP_VotesLast=("AP VOTES", "last"),
        AP_RankBest=("AP RANK", "min"),
        AP_RankLast=("AP RANK", "last"),
        AP_WinsAtLast=("W", "last"),
        AP_LossesAtLast=("L", "last"),
        AP_RankedAny=("RANK?", "max"),
    )
    return out



def build_attached_team_features(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    base_keys = _base_keys(data_dir)

    out = base_keys.rename(columns={"YEAR": "Season", "TEAM NO": "TeamID", "TEAM": "TeamName"}).copy()
    out["Seed"] = out["SEED"]
    out["SeedNum"] = pd.to_numeric(out["SEED"], errors="coerce")
    out["SeedRegion"] = "Z"
    out["IsPlayInSeed"] = 0
    out["HasSeed"] = out["SeedNum"].notna().astype(int)
    out["ConfID"] = pd.to_numeric(out["CONF ID"], errors="coerce")

    base_keys_year_team = base_keys[["YEAR", "TEAM NO", "CONF ID"]].copy()

    for fname, cols in TEAM_SOURCE_COLUMNS.items():
        source_df = _load_team_source(data_dir, fname, cols, base_keys)
        if source_df.empty:
            continue
        source_df = source_df.rename(columns={"YEAR": "Season", "TEAM NO": "TeamID"})
        out = out.merge(source_df, on=["Season", "TeamID"], how="left")

    ap_df = _aggregate_ap_poll(data_dir, base_keys)
    if not ap_df.empty:
        ap_df = ap_df.rename(columns={"YEAR": "Season", "TEAM NO": "TeamID"})
        out = out.merge(ap_df, on=["Season", "TeamID"], how="left")

    conf_merge = base_keys_year_team.rename(columns={"YEAR": "Season", "TEAM NO": "TeamID"})
    for fname, cols in CONF_SOURCE_COLUMNS.items():
        conf_df = _load_conf_source(data_dir, fname, cols)
        if conf_df.empty:
            continue
        conf_df = conf_df.rename(columns={"YEAR": "Season"})
        out = out.merge(conf_df, left_on=["Season", "CONF ID"], right_on=["Season", "CONF ID"], how="left")

    numeric_cols = [
        col
        for col in out.columns
        if col not in {"Season", "TeamID", "TeamName", "SEED", "CONF", "Seed", "SeedRegion"}
        and pd.api.types.is_numeric_dtype(out[col])
    ]
    if numeric_cols:
        missing_frac = out[numeric_cols].isna().mean()
        drop_cols = [col for col in numeric_cols if missing_frac[col] > OPTIONAL_TEAM_MISSING_THRESHOLD]
        keep_numeric = [col for col in numeric_cols if col not in set(drop_cols)]
        keep_base = ["Season", "TeamID", "TeamName", "Seed", "SeedNum", "SeedRegion", "HasSeed", "IsPlayInSeed"]
        extra = [col for col in out.columns if col in keep_numeric and col not in keep_base]
        out = out[keep_base + extra].copy()

    out = out.sort_values(["Season", "TeamID"]).reset_index(drop=True)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out



def _pair_frame(df: pd.DataFrame) -> list[tuple[pd.Series, pd.Series]]:
    ordered = df.sort_values(["YEAR", "BY YEAR NO"], ascending=[False, False]).reset_index(drop=True)
    pairs: list[tuple[pd.Series, pd.Series]] = []
    if len(ordered) % 2 != 0:
        raise AttachedDataError(f"Expected an even number of rows in paired frame, found {len(ordered)}")
    for idx in range(0, len(ordered), 2):
        a = ordered.iloc[idx]
        b = ordered.iloc[idx + 1]
        ok = (
            int(a["YEAR"]) == int(b["YEAR"])
            and int(a["CURRENT ROUND"]) == int(b["CURRENT ROUND"])
            and int(a["BY YEAR NO"]) == int(b["BY YEAR NO"]) + 1
        )
        if not ok:
            raise AttachedDataError(
                "Could not pair tournament rows cleanly. Expected adjacent rows to share YEAR/CURRENT ROUND "
                "and descending BY YEAR NO."
            )
        pairs.append((a, b))
    return pairs



def build_attached_location_context(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    path = data_dir / LOCATION_FILE
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "Season",
                "CurrentRound",
                "TeamAID",
                "TeamBID",
                "LocDistanceMIA",
                "LocDistanceMIB",
                "LocDistanceMIDiff",
                "LocDistanceMIAvg",
                "LocTZCrossA",
                "LocTZCrossB",
                "LocTZCrossDiff",
                "LocTZCrossAvg",
            ]
        )

    raw = _load_csv(path)
    required = {"YEAR", "BY YEAR NO", "TEAM NO", "CURRENT ROUND"}
    if required - set(raw.columns):
        raise AttachedDataError(f"{LOCATION_FILE} is missing columns: {sorted(required - set(raw.columns))}")
    for col in ["DISTANCE (MI)", "TIME ZONES CROSSED VALUE"]:
        if col not in raw.columns:
            raw[col] = np.nan

    rows: list[dict[str, float | int]] = []
    for a, b in _pair_frame(raw):
        row = {
            "Season": int(a["YEAR"]),
            "CurrentRound": int(a["CURRENT ROUND"]),
            "TeamAID": int(a["TEAM NO"]),
            "TeamBID": int(b["TEAM NO"]),
            "LocDistanceMIA": pd.to_numeric(a["DISTANCE (MI)"], errors="coerce"),
            "LocDistanceMIB": pd.to_numeric(b["DISTANCE (MI)"], errors="coerce"),
            "LocTZCrossA": pd.to_numeric(a["TIME ZONES CROSSED VALUE"], errors="coerce"),
            "LocTZCrossB": pd.to_numeric(b["TIME ZONES CROSSED VALUE"], errors="coerce"),
        }
        row["LocDistanceMIDiff"] = row["LocDistanceMIA"] - row["LocDistanceMIB"]
        row["LocDistanceMIAvg"] = np.nanmean([row["LocDistanceMIA"], row["LocDistanceMIB"]])
        row["LocTZCrossDiff"] = row["LocTZCrossA"] - row["LocTZCrossB"]
        row["LocTZCrossAvg"] = np.nanmean([row["LocTZCrossA"], row["LocTZCrossB"]])
        rows.append(row)
        rows.append(
            {
                "Season": row["Season"],
                "CurrentRound": row["CurrentRound"],
                "TeamAID": row["TeamBID"],
                "TeamBID": row["TeamAID"],
                "LocDistanceMIA": row["LocDistanceMIB"],
                "LocDistanceMIB": row["LocDistanceMIA"],
                "LocDistanceMIDiff": -row["LocDistanceMIDiff"] if pd.notna(row["LocDistanceMIDiff"]) else np.nan,
                "LocDistanceMIAvg": row["LocDistanceMIAvg"],
                "LocTZCrossA": row["LocTZCrossB"],
                "LocTZCrossB": row["LocTZCrossA"],
                "LocTZCrossDiff": -row["LocTZCrossDiff"] if pd.notna(row["LocTZCrossDiff"]) else np.nan,
                "LocTZCrossAvg": row["LocTZCrossAvg"],
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["Season", "CurrentRound", "TeamAID", "TeamBID"]).reset_index(drop=True)



def build_attached_tournament_rows(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    raw = _load_csv(_require_path(data_dir / MATCHUP_FILE))
    required = {"YEAR", "BY YEAR NO", "TEAM NO", "TEAM", "CURRENT ROUND", "SCORE"}
    missing = required - set(raw.columns)
    if missing:
        raise AttachedDataError(f"{MATCHUP_FILE} is missing columns: {sorted(missing)}")

    location_context = build_attached_location_context(data_dir)
    rows: list[dict[str, float | int | str]] = []
    for a, b in _pair_frame(raw):
        season = int(a["YEAR"])
        current_round = int(a["CURRENT ROUND"])
        team_a = int(a["TEAM NO"])
        team_b = int(b["TEAM NO"])
        score_a = float(a["SCORE"])
        score_b = float(b["SCORE"])
        row = {
            "Season": season,
            "CurrentRound": current_round,
            "TeamAID": team_a,
            "TeamBID": team_b,
            "TeamAName": str(a["TEAM"]),
            "TeamBName": str(b["TEAM"]),
            "TeamAScore": score_a,
            "TeamBScore": score_b,
            "TeamAWin": int(score_a > score_b),
            "Margin": score_a - score_b,
            "Total": score_a + score_b,
        }
        rows.append(row)
        rows.append(
            {
                "Season": season,
                "CurrentRound": current_round,
                "TeamAID": team_b,
                "TeamBID": team_a,
                "TeamAName": str(b["TEAM"]),
                "TeamBName": str(a["TEAM"]),
                "TeamAScore": score_b,
                "TeamBScore": score_a,
                "TeamAWin": int(score_b > score_a),
                "Margin": score_b - score_a,
                "Total": score_a + score_b,
            }
        )
    out = pd.DataFrame(rows)
    if not location_context.empty:
        out = out.merge(location_context, on=["Season", "CurrentRound", "TeamAID", "TeamBID"], how="left")
    return out.sort_values(["Season", "CurrentRound", "TeamAID", "TeamBID"]).reset_index(drop=True)



def attached_team_name_map(team_features: pd.DataFrame) -> dict[int, str]:
    base = team_features[["TeamID", "TeamName"]].drop_duplicates(subset=["TeamID"])
    return dict(zip(base["TeamID"].astype(int), base["TeamName"].astype(str)))



def attached_team_lookup(team_features: pd.DataFrame) -> dict[tuple[int, str], int]:
    mapping: dict[tuple[int, str], int] = {}
    for row in team_features[["Season", "TeamID", "TeamName"]].drop_duplicates().itertuples(index=False):
        mapping[(int(row.Season), _normalize_name(str(row.TeamName)))] = int(row.TeamID)
    return mapping



def load_attached_schedule(
    schedule_csv: str | Path,
    team_features: pd.DataFrame,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    path = Path(schedule_csv)
    df = pd.read_csv(path)
    if "Season" not in df.columns:
        raise ValueError("Schedule CSV must include a Season column.")

    schedule = df.copy()
    if {"TeamAID", "TeamBID"}.issubset(schedule.columns):
        schedule["TeamAID"] = schedule["TeamAID"].astype(int)
        schedule["TeamBID"] = schedule["TeamBID"].astype(int)
    elif {"TeamA", "TeamB"}.issubset(schedule.columns):
        lookup = attached_team_lookup(team_features)

        def _resolve_team(row: pd.Series, col: str) -> int:
            key = (int(row["Season"]), _normalize_name(str(row[col])))
            if key not in lookup:
                raise KeyError(f"Could not resolve {col}={row[col]!r} for Season={int(row['Season'])}.")
            return int(lookup[key])

        schedule["TeamAID"] = schedule.apply(lambda row: _resolve_team(row, "TeamA"), axis=1)
        schedule["TeamBID"] = schedule.apply(lambda row: _resolve_team(row, "TeamB"), axis=1)
    else:
        raise ValueError("Schedule CSV must contain either Season,TeamAID,TeamBID or Season,TeamA,TeamB.")

    if "CurrentRound" not in schedule.columns:
        schedule["CurrentRound"] = 64
    schedule["CurrentRound"] = pd.to_numeric(schedule["CurrentRound"], errors="coerce").fillna(64).astype(int)

    if data_dir is not None:
        location_context = build_attached_location_context(data_dir)
        if not location_context.empty:
            schedule = schedule.merge(location_context, on=["Season", "CurrentRound", "TeamAID", "TeamBID"], how="left")

    return schedule



def load_attached_bundle(data_dir: str | Path) -> AttachedDataBundle:
    team_features = build_attached_team_features(data_dir)
    tournament_rows = build_attached_tournament_rows(data_dir)
    return AttachedDataBundle(
        team_features=team_features,
        tournament_rows=tournament_rows,
        team_name_map=attached_team_name_map(team_features),
        season_team_lookup=attached_team_lookup(team_features),
    )
