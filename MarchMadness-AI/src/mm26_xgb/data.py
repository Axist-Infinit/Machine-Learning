from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd

from .config import OPTIONAL_FILES, REQUIRED_FILES


@dataclass(slots=True)
class KaggleDataBundle:
    teams: pd.DataFrame
    regular: pd.DataFrame
    tourney: pd.DataFrame
    seeds: pd.DataFrame
    massey: pd.DataFrame | None = None
    slots: pd.DataFrame | None = None
    seed_round_slots: pd.DataFrame | None = None
    seasons: pd.DataFrame | None = None
    spellings: pd.DataFrame | None = None


SEED_RE = re.compile(r"([A-Z])(\d{2})([ab])?", re.IGNORECASE)


REQUIRED_COLUMNS: dict[str, set[str]] = {
    "teams": {"TeamID", "TeamName"},
    "regular": {
        "Season",
        "DayNum",
        "WTeamID",
        "LTeamID",
        "WLoc",
        "NumOT",
        "WScore",
        "LScore",
        "WFGM",
        "WFGA",
        "WFGM3",
        "WFGA3",
        "WFTM",
        "WFTA",
        "WOR",
        "WDR",
        "WAst",
        "WTO",
        "WStl",
        "WBlk",
        "WPF",
        "LFGM",
        "LFGA",
        "LFGM3",
        "LFGA3",
        "LFTM",
        "LFTA",
        "LOR",
        "LDR",
        "LAst",
        "LTO",
        "LStl",
        "LBlk",
        "LPF",
    },
    "tourney": {
        "Season",
        "DayNum",
        "WTeamID",
        "LTeamID",
        "NumOT",
        "WScore",
        "LScore",
        "WFGM",
        "WFGA",
        "WFGM3",
        "WFGA3",
        "WFTM",
        "WFTA",
        "WOR",
        "WDR",
        "WAst",
        "WTO",
        "WStl",
        "WBlk",
        "WPF",
        "LFGM",
        "LFGA",
        "LFGM3",
        "LFGA3",
        "LFTM",
        "LFTA",
        "LOR",
        "LDR",
        "LAst",
        "LTO",
        "LStl",
        "LBlk",
        "LPF",
    },
    "seeds": {"Season", "Seed", "TeamID"},
}


OPTIONAL_COLUMNS: dict[str, set[str]] = {
    "massey": {"Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"},
    "slots": {"Season", "Slot", "StrongSeed", "WeakSeed"},
    "seed_round_slots": {"Seed", "GameRound", "GameSlot"},
    "seasons": {"Season"},
    "spellings": {"TeamID", "TeamNameSpelling"},
}


class DataValidationError(RuntimeError):
    pass



def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)



def _check_columns(name: str, df: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise DataValidationError(f"{name} is missing required columns: {missing}.")



def load_kaggle_data(data_dir: str | Path) -> KaggleDataBundle:
    data_dir = Path(data_dir)
    missing_files = [fname for fname in REQUIRED_FILES.values() if not (data_dir / fname).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required Kaggle files in {data_dir}: {missing_files}")

    payload: dict[str, Any] = {}

    for key, fname in REQUIRED_FILES.items():
        df = _read_csv(data_dir / fname)
        _check_columns(key, df, REQUIRED_COLUMNS[key])
        payload[key] = df

    for key, fname in OPTIONAL_FILES.items():
        path = data_dir / fname
        if path.exists():
            df = _read_csv(path)
            _check_columns(key, df, OPTIONAL_COLUMNS[key])
            payload[key] = df
        else:
            payload[key] = None

    payload["seeds"] = preprocess_seeds(payload["seeds"])
    payload["teams"] = payload["teams"].copy()
    payload["teams"]["TeamName"] = payload["teams"]["TeamName"].astype(str)
    if payload.get("spellings") is not None:
        payload["spellings"] = payload["spellings"].copy()
        payload["spellings"]["TeamNameSpelling"] = payload["spellings"]["TeamNameSpelling"].astype(str)

    return KaggleDataBundle(**payload)



def preprocess_seeds(df: pd.DataFrame) -> pd.DataFrame:
    seeds = df.copy()
    parsed = seeds["Seed"].astype(str).str.extract(SEED_RE)
    seeds["SeedRegion"] = parsed[0].str.upper()
    seeds["SeedNum"] = parsed[1].astype(int)
    seeds["SeedSuffix"] = parsed[2].fillna("")
    seeds["IsPlayInSeed"] = (seeds["SeedSuffix"] != "").astype(int)
    return seeds



def team_name_map(teams_df: pd.DataFrame) -> dict[int, str]:
    return dict(zip(teams_df["TeamID"].astype(int), teams_df["TeamName"].astype(str)))



def normalize_team_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())



def team_lookup(
    teams_df: pd.DataFrame,
    spellings_df: pd.DataFrame | None = None,
    extra_aliases: dict[str, int] | None = None,
) -> dict[str, int]:
    mapping: dict[str, int] = {}

    for row in teams_df.itertuples(index=False):
        mapping[normalize_team_name(str(row.TeamName))] = int(row.TeamID)

    if spellings_df is not None and not spellings_df.empty:
        for row in spellings_df.itertuples(index=False):
            mapping[normalize_team_name(str(row.TeamNameSpelling))] = int(row.TeamID)

    if extra_aliases:
        for key, value in extra_aliases.items():
            mapping[normalize_team_name(str(key))] = int(value)

    return mapping



def load_aliases_csv(
    path: str | Path | None,
    teams_df: pd.DataFrame,
) -> dict[str, int]:
    if path is None:
        return {}
    alias_path = Path(path)
    if not alias_path.exists():
        raise FileNotFoundError(f"Alias CSV not found: {alias_path}")

    df = pd.read_csv(alias_path)
    if "ExternalName" not in df.columns:
        raise ValueError("Alias CSV must include an ExternalName column.")

    name_lookup = {normalize_team_name(name): team_id for team_id, name in team_name_map(teams_df).items()}
    aliases: dict[str, int] = {}

    for row in df.to_dict("records"):
        external = str(row["ExternalName"]).strip()
        if not external:
            continue
        if "TeamID" in row and pd.notna(row["TeamID"]):
            aliases[external] = int(row["TeamID"])
            continue
        canonical_name = None
        for key in ("TeamName", "CanonicalName"):
            if key in row and pd.notna(row[key]):
                canonical_name = str(row[key]).strip()
                break
        if canonical_name is None:
            raise ValueError("Alias CSV rows must include TeamID or TeamName/CanonicalName.")
        norm = normalize_team_name(canonical_name)
        if norm not in name_lookup:
            raise KeyError(f"Alias target team not found in MTeams.csv: {canonical_name!r}")
        aliases[external] = int(name_lookup[norm])

    return {normalize_team_name(k): int(v) for k, v in aliases.items()}



def resolve_team_identifier(value: str | int, lookup: dict[str, int]) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    key = normalize_team_name(text)
    if key not in lookup:
        raise KeyError(f"Could not resolve team identifier: {value!r}")
    return lookup[key]
