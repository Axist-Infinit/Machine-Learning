from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import difflib
import re
from typing import Any

import numpy as np
import pandas as pd

from .data import load_aliases_csv, normalize_team_name, team_lookup


BUILTIN_EXTERNAL_TEAM_ALIASES: dict[str, str] = {
    "miami": "Miami FL",
    "s florida": "South Florida",
    "e tennessee st": "East Tennessee St.",
    "ucsd": "UC San Diego",
    "n iowa": "Northern Iowa",
    "s alabama": "South Alabama",
    "c arkansas": "Central Arkansas",
    "queens": "Queens NC",
    "c connecticut": "Central Connecticut",
    "app state": "Appalachian St",
    "siu edward": "SIUE",
    "n texas": "North Texas",
    "ut rio grande": "Texas Rio Grande Valley",
    "georgia so": "Georgia Southern",
    "j madison": "James Madison",
    "bethune": "Bethune-Cookman",
    "purdue fw": "Purdue Fort Wayne",
    "loyola mymt": "Loyola Marymount",
    "w georgia": "West Georgia",
    "abl christian": "Abilene Christian",
    "tenn tech": "Tennessee Tech",
    "ar-pine bluff": "Ark Pine Bluff",
    "texas so": "Texas Southern",
    "hou christian": "Houston Christian",
    "e carolina": "East Carolina",
    "e texas a&m": "East Texas A&M",
    "s utah": "Southern Utah",
    "n arizona": "Northern Arizona",
    "nw state": "Northwestern State",
    "n alabama": "North Alabama",
    "ul monroe": "ULM",
    "n florida": "North Florida",
    "maryland es": "Maryland Eastern Shore",
    "s indiana": "Southern Indiana",
    "loyola chi": "Loyola Chicago",
}


@dataclass(slots=True)
class ExternalTeamFeatureBundle:
    frame: pd.DataFrame
    coverage: dict[str, Any]
    unresolved: dict[str, list[str]]


EXTERNAL_KEY_COLS = {"Season", "TeamID", "Team"}


def _to_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    if text.endswith("%"):
        text = text[:-1].strip()
        try:
            return float(text) / 100.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_record(value: Any) -> tuple[float, float, float, float]:
    if pd.isna(value):
        return (np.nan, np.nan, np.nan, np.nan)
    parts = re.findall(r"\d+", str(value))
    if not parts:
        return (np.nan, np.nan, np.nan, np.nan)
    nums = [float(x) for x in parts[:3]]
    while len(nums) < 3:
        nums.append(0.0)
    wins, losses, pushes = nums[:3]
    games = wins + losses + pushes
    return wins, losses, pushes, games


def _normalize_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _rename_external_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    taken: set[str] = set()
    for col in df.columns:
        key = _normalize_column_name(col)
        target = mapping.get(key)
        if target and target not in taken:
            renamed[col] = target
            taken.add(target)
    return df.rename(columns=renamed)


def _seasonize(df: pd.DataFrame, target_season: int, default_season: int | None = None) -> pd.DataFrame:
    out = df.copy()
    if "Season" in out.columns:
        out["Season"] = pd.to_numeric(out["Season"], errors="coerce").fillna(int(default_season or target_season)).astype(int)
    else:
        out["Season"] = int(default_season or target_season)
    return out


def _resolve_team_id(name: Any, lookup: dict[str, int]) -> int | None:
    if pd.isna(name):
        return None
    raw = str(name).strip()
    if not raw:
        return None

    norm = normalize_team_name(raw)
    if norm in lookup:
        return int(lookup[norm])

    alias_target = BUILTIN_EXTERNAL_TEAM_ALIASES.get(raw) or BUILTIN_EXTERNAL_TEAM_ALIASES.get(raw.lower()) or BUILTIN_EXTERNAL_TEAM_ALIASES.get(norm)
    if alias_target is not None:
        alias_norm = normalize_team_name(alias_target)
        if alias_norm in lookup:
            return int(lookup[alias_norm])

    candidates = difflib.get_close_matches(norm, list(lookup.keys()), n=2, cutoff=0.88)
    if len(candidates) == 1:
        return int(lookup[candidates[0]])
    if len(candidates) >= 2:
        best = difflib.SequenceMatcher(None, norm, candidates[0]).ratio()
        second = difflib.SequenceMatcher(None, norm, candidates[1]).ratio()
        if best >= 0.92 and (best - second) >= 0.05:
            return int(lookup[candidates[0]])
    return None


def _resolve_teams(
    df: pd.DataFrame,
    teams_df: pd.DataFrame,
    spellings_df: pd.DataFrame | None,
    aliases_csv: str | Path | None,
    strict: bool,
) -> tuple[pd.DataFrame, list[str]]:
    if "TeamID" in df.columns:
        out = df.copy()
        out["TeamID"] = pd.to_numeric(out["TeamID"], errors="coerce").astype("Int64")
        unresolved = out.loc[out["TeamID"].isna(), "Team"].astype(str).tolist() if "Team" in out.columns else []
        if strict and unresolved:
            raise ValueError(f"Could not resolve team IDs for: {sorted(set(unresolved))[:20]}")
        out = out.dropna(subset=["TeamID"]).copy()
        out["TeamID"] = out["TeamID"].astype(int)
        return out, sorted(set(unresolved))

    if "Team" not in df.columns:
        raise ValueError("External team stats CSV must include Team or TeamID column.")

    extra_aliases = load_aliases_csv(aliases_csv, teams_df) if aliases_csv else {}
    lookup = team_lookup(teams_df, spellings_df, extra_aliases)
    out = df.copy()
    out["TeamID"] = out["Team"].apply(lambda x: _resolve_team_id(x, lookup)).astype("Int64")
    unresolved = sorted(set(out.loc[out["TeamID"].isna(), "Team"].astype(str).tolist()))
    if strict and unresolved:
        raise ValueError(f"Could not resolve external team stats for: {unresolved[:20]}")
    out = out.dropna(subset=["TeamID"]).copy()
    out["TeamID"] = out["TeamID"].astype(int)
    return out, unresolved


def _zscore_by_season(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        grp = out.groupby("Season")[col]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        out[f"{col}Z"] = ((out[col] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _finalize_external_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c not in EXTERNAL_KEY_COLS]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = _zscore_by_season(df, numeric_cols)

    side_terms = [c for c in [
        "ExtWL_WinPctZ",
        "ExtWL_MOVZ",
        "ExtWL_ATSPlusMinusZ",
        "ExtATS_CoverPctZ",
        "ExtATS_ATSPlusMinusZ",
    ] if c in df.columns]
    if side_terms:
        weights = {
            "ExtWL_WinPctZ": 0.35,
            "ExtWL_MOVZ": 0.30,
            "ExtWL_ATSPlusMinusZ": 0.10,
            "ExtATS_CoverPctZ": 0.15,
            "ExtATS_ATSPlusMinusZ": 0.10,
        }
        df["ExtCompositeSide"] = sum(float(weights.get(col, 0.0)) * df[col] for col in side_terms)

    total_terms = [c for c in [
        "ExtOU_OverPctZ",
        "ExtOU_UnderPctZ",
        "ExtOU_TotalPlusMinusZ",
    ] if c in df.columns]
    if total_terms:
        weights = {
            "ExtOU_OverPctZ": 0.45,
            "ExtOU_UnderPctZ": -0.15,
            "ExtOU_TotalPlusMinusZ": 0.40,
        }
        df["ExtCompositeTotal"] = sum(float(weights.get(col, 0.0)) * df[col] for col in total_terms)

    tempo_terms = [c for c in [
        "ExtTempo_AdjOffZ",
        "ExtTempo_AdjDefZ",
        "ExtTempo_AdjTempoZ",
        "ExtTempo_eFGZ",
        "ExtTempo_ThreeRateZ",
        "ExtTempo_FTRateZ",
        "ExtTempo_ORBRateZ",
        "ExtTempo_TOVRateZ",
    ] if c in df.columns]
    if tempo_terms:
        weights = {
            "ExtTempo_AdjOffZ": 0.28,
            "ExtTempo_AdjDefZ": -0.24,
            "ExtTempo_AdjTempoZ": 0.12,
            "ExtTempo_eFGZ": 0.10,
            "ExtTempo_ThreeRateZ": 0.07,
            "ExtTempo_FTRateZ": 0.06,
            "ExtTempo_ORBRateZ": 0.06,
            "ExtTempo_TOVRateZ": -0.07,
        }
        df["ExtTempoComposite"] = sum(float(weights.get(col, 0.0)) * df[col] for col in tempo_terms)

    lineup_terms = [c for c in [
        "ExtLineup_ContinuityPctZ",
        "ExtLineup_ReturningMinutesPctZ",
        "ExtLineup_Top6MinutesPctZ",
        "ExtLineup_Top5MinutesPctZ",
        "ExtLineup_RotationPlayersZ",
        "ExtLineup_InjuryImpactZ",
    ] if c in df.columns]
    if lineup_terms:
        weights = {
            "ExtLineup_ContinuityPctZ": 0.30,
            "ExtLineup_ReturningMinutesPctZ": 0.25,
            "ExtLineup_Top6MinutesPctZ": 0.15,
            "ExtLineup_Top5MinutesPctZ": 0.10,
            "ExtLineup_RotationPlayersZ": -0.05,
            "ExtLineup_InjuryImpactZ": -0.15,
        }
        df["ExtLineupComposite"] = sum(float(weights.get(col, 0.0)) * df[col] for col in lineup_terms)

    keep = ["Season", "TeamID", *[c for c in df.columns if c not in EXTERNAL_KEY_COLS]]
    out = df[keep].sort_values(["Season", "TeamID"]).reset_index(drop=True)
    return out


def _load_win_loss_features(
    path: str | Path,
    teams_df: pd.DataFrame,
    spellings_df: pd.DataFrame | None,
    target_season: int,
    aliases_csv: str | Path | None,
    strict: bool,
    default_season: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    df = _seasonize(df, target_season=target_season, default_season=default_season)
    df, unresolved = _resolve_teams(df, teams_df, spellings_df, aliases_csv=aliases_csv, strict=strict)

    wins, losses, pushes, games = zip(*df["Win-Loss Record"].map(_parse_record))
    out = pd.DataFrame(
        {
            "Season": df["Season"].astype(int),
            "TeamID": df["TeamID"].astype(int),
            "ExtWL_Wins": wins,
            "ExtWL_Losses": losses,
            "ExtWL_Pushes": pushes,
            "ExtWL_Games": games,
            "ExtWL_WinPct": df["Win %"].map(_to_float),
            "ExtWL_MOV": df["MOV"].map(_to_float),
            "ExtWL_ATSPlusMinus": df["ATS +/-"].map(_to_float),
        }
    )
    return out, unresolved


def _load_ats_features(
    path: str | Path,
    teams_df: pd.DataFrame,
    spellings_df: pd.DataFrame | None,
    target_season: int,
    aliases_csv: str | Path | None,
    strict: bool,
    default_season: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    df = _seasonize(df, target_season=target_season, default_season=default_season)
    df, unresolved = _resolve_teams(df, teams_df, spellings_df, aliases_csv=aliases_csv, strict=strict)

    wins, losses, pushes, games = zip(*df["ATS Record"].map(_parse_record))
    out = pd.DataFrame(
        {
            "Season": df["Season"].astype(int),
            "TeamID": df["TeamID"].astype(int),
            "ExtATS_Wins": wins,
            "ExtATS_Losses": losses,
            "ExtATS_Pushes": pushes,
            "ExtATS_Games": games,
            "ExtATS_CoverPct": df["Cover %"].map(_to_float),
            "ExtATS_MOV": df["MOV"].map(_to_float),
            "ExtATS_ATSPlusMinus": df["ATS +/-"].map(_to_float),
        }
    )
    return out, unresolved


def _load_ou_features(
    path: str | Path,
    teams_df: pd.DataFrame,
    spellings_df: pd.DataFrame | None,
    target_season: int,
    aliases_csv: str | Path | None,
    strict: bool,
    default_season: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    df = _seasonize(df, target_season=target_season, default_season=default_season)
    df, unresolved = _resolve_teams(df, teams_df, spellings_df, aliases_csv=aliases_csv, strict=strict)

    overs, unders, pushes, games = zip(*df["Over Record"].map(_parse_record))
    out = pd.DataFrame(
        {
            "Season": df["Season"].astype(int),
            "TeamID": df["TeamID"].astype(int),
            "ExtOU_Overs": overs,
            "ExtOU_Unders": unders,
            "ExtOU_Pushes": pushes,
            "ExtOU_Games": games,
            "ExtOU_OverPct": df["Over %"].map(_to_float),
            "ExtOU_UnderPct": df["Under %"].map(_to_float),
            "ExtOU_TotalPlusMinus": df["Total +/-"].map(_to_float),
        }
    )
    return out, unresolved



def _load_tempo_features(
    path: str | Path,
    teams_df: pd.DataFrame,
    spellings_df: pd.DataFrame | None,
    target_season: int,
    aliases_csv: str | Path | None,
    strict: bool,
    default_season: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    mapping = {
        "season": "Season",
        "team": "Team",
        "teamname": "Team",
        "school": "Team",
        "teamid": "TeamID",
        "adjtempo": "ExtTempo_AdjTempo",
        "tempo": "ExtTempo_AdjTempo",
        "pace": "ExtTempo_AdjTempo",
        "tempoavg": "ExtTempo_AdjTempo",
        "adjoe": "ExtTempo_AdjOff",
        "adjo": "ExtTempo_AdjOff",
        "adjoff": "ExtTempo_AdjOff",
        "adjoffeff": "ExtTempo_AdjOff",
        "adjde": "ExtTempo_AdjDef",
        "adjd": "ExtTempo_AdjDef",
        "adjdef": "ExtTempo_AdjDef",
        "adjdefeff": "ExtTempo_AdjDef",
        "efg": "ExtTempo_eFG",
        "efgpct": "ExtTempo_eFG",
        "threerate": "ExtTempo_ThreeRate",
        "threeparate": "ExtTempo_ThreeRate",
        "ftrate": "ExtTempo_FTRate",
        "orbrate": "ExtTempo_ORBRate",
        "tovrate": "ExtTempo_TOVRate",
        "ts": "ExtTempo_TS",
        "trueshooting": "ExtTempo_TS",
        "rimrate": "ExtTempo_RimRate",
        "midrangerate": "ExtTempo_MidRangeRate",
        "shotquality": "ExtTempo_ShotQuality",
    }
    df = pd.read_csv(path)
    df = _rename_external_columns(df, mapping)
    df = _seasonize(df, target_season=target_season, default_season=default_season)
    df, unresolved = _resolve_teams(df, teams_df, spellings_df, aliases_csv=aliases_csv, strict=strict)

    keep_numeric = [
        col for col in [
            "ExtTempo_AdjTempo",
            "ExtTempo_AdjOff",
            "ExtTempo_AdjDef",
            "ExtTempo_eFG",
            "ExtTempo_ThreeRate",
            "ExtTempo_FTRate",
            "ExtTempo_ORBRate",
            "ExtTempo_TOVRate",
            "ExtTempo_TS",
            "ExtTempo_RimRate",
            "ExtTempo_MidRangeRate",
            "ExtTempo_ShotQuality",
        ] if col in df.columns
    ]
    out = pd.DataFrame({
        "Season": df["Season"].astype(int),
        "TeamID": df["TeamID"].astype(int),
    })
    for col in keep_numeric:
        out[col] = df[col].map(_to_float)
    return out, unresolved



def _load_lineup_features(
    path: str | Path,
    teams_df: pd.DataFrame,
    spellings_df: pd.DataFrame | None,
    target_season: int,
    aliases_csv: str | Path | None,
    strict: bool,
    default_season: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    mapping = {
        "season": "Season",
        "team": "Team",
        "teamname": "Team",
        "school": "Team",
        "teamid": "TeamID",
        "continuitypct": "ExtLineup_ContinuityPct",
        "continuity": "ExtLineup_ContinuityPct",
        "returningminutespct": "ExtLineup_ReturningMinutesPct",
        "returningminutes": "ExtLineup_ReturningMinutesPct",
        "top5minutespct": "ExtLineup_Top5MinutesPct",
        "top6minutespct": "ExtLineup_Top6MinutesPct",
        "benchminutespct": "ExtLineup_BenchMinutesPct",
        "starterminutespct": "ExtLineup_StarterMinutesPct",
        "rotationplayers": "ExtLineup_RotationPlayers",
        "minutesconcentration": "ExtLineup_MinutesConcentration",
        "injuryimpact": "ExtLineup_InjuryImpact",
    }
    df = pd.read_csv(path)
    df = _rename_external_columns(df, mapping)
    df = _seasonize(df, target_season=target_season, default_season=default_season)
    df, unresolved = _resolve_teams(df, teams_df, spellings_df, aliases_csv=aliases_csv, strict=strict)

    keep_numeric = [
        col for col in [
            "ExtLineup_ContinuityPct",
            "ExtLineup_ReturningMinutesPct",
            "ExtLineup_Top5MinutesPct",
            "ExtLineup_Top6MinutesPct",
            "ExtLineup_BenchMinutesPct",
            "ExtLineup_StarterMinutesPct",
            "ExtLineup_RotationPlayers",
            "ExtLineup_MinutesConcentration",
            "ExtLineup_InjuryImpact",
        ] if col in df.columns
    ]
    out = pd.DataFrame({
        "Season": df["Season"].astype(int),
        "TeamID": df["TeamID"].astype(int),
    })
    for col in keep_numeric:
        out[col] = df[col].map(_to_float)
    return out, unresolved


def load_external_team_features(
    teams_df: pd.DataFrame,
    spellings_df: pd.DataFrame | None,
    target_season: int,
    win_loss_csv: str | Path | None = None,
    ats_csv: str | Path | None = None,
    ou_csv: str | Path | None = None,
    tempo_csv: str | Path | None = None,
    lineup_csv: str | Path | None = None,
    aliases_csv: str | Path | None = None,
    strict: bool = False,
    default_season: int | None = None,
) -> ExternalTeamFeatureBundle:
    frames: list[pd.DataFrame] = []
    coverage: dict[str, Any] = {}
    unresolved: dict[str, list[str]] = {}

    if win_loss_csv:
        wl, missing = _load_win_loss_features(
            path=win_loss_csv,
            teams_df=teams_df,
            spellings_df=spellings_df,
            target_season=target_season,
            aliases_csv=aliases_csv,
            strict=strict,
            default_season=default_season,
        )
        frames.append(wl)
        coverage["win_loss_rows"] = int(len(wl))
        unresolved["win_loss"] = missing

    if ats_csv:
        ats, missing = _load_ats_features(
            path=ats_csv,
            teams_df=teams_df,
            spellings_df=spellings_df,
            target_season=target_season,
            aliases_csv=aliases_csv,
            strict=strict,
            default_season=default_season,
        )
        frames.append(ats)
        coverage["ats_rows"] = int(len(ats))
        unresolved["ats"] = missing

    if ou_csv:
        ou, missing = _load_ou_features(
            path=ou_csv,
            teams_df=teams_df,
            spellings_df=spellings_df,
            target_season=target_season,
            aliases_csv=aliases_csv,
            strict=strict,
            default_season=default_season,
        )
        frames.append(ou)
        coverage["ou_rows"] = int(len(ou))
        unresolved["ou"] = missing

    if tempo_csv:
        tempo, missing = _load_tempo_features(
            path=tempo_csv,
            teams_df=teams_df,
            spellings_df=spellings_df,
            target_season=target_season,
            aliases_csv=aliases_csv,
            strict=strict,
            default_season=default_season,
        )
        frames.append(tempo)
        coverage["tempo_rows"] = int(len(tempo))
        unresolved["tempo"] = missing

    if lineup_csv:
        lineup, missing = _load_lineup_features(
            path=lineup_csv,
            teams_df=teams_df,
            spellings_df=spellings_df,
            target_season=target_season,
            aliases_csv=aliases_csv,
            strict=strict,
            default_season=default_season,
        )
        frames.append(lineup)
        coverage["lineup_rows"] = int(len(lineup))
        unresolved["lineup"] = missing

    if not frames:
        return ExternalTeamFeatureBundle(frame=pd.DataFrame(columns=["Season", "TeamID"]), coverage=coverage, unresolved=unresolved)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["Season", "TeamID"], how="outer")
    merged = _finalize_external_frame(merged)
    coverage["merged_rows"] = int(len(merged))
    coverage["seasons"] = sorted(int(x) for x in merged["Season"].dropna().unique().tolist())
    return ExternalTeamFeatureBundle(frame=merged, coverage=coverage, unresolved=unresolved)


def merge_external_team_features(team_features: pd.DataFrame, external_features: pd.DataFrame | None) -> pd.DataFrame:
    if external_features is None or external_features.empty:
        return team_features.copy()
    merged = team_features.merge(external_features, on=["Season", "TeamID"], how="left")
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return merged.sort_values(["Season", "TeamID"]).reset_index(drop=True)
