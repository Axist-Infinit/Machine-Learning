#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import hashlib
from datetime import datetime, timedelta, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import textwrap
import time
import warnings
import zipfile
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mm26_xgb.bracket import simulate_bracket
from mm26_xgb.data import (
    load_kaggle_data,
    load_aliases_csv,
    normalize_team_name,
    resolve_team_identifier,
    team_lookup,
    team_name_map,
)
from mm26_xgb.execution import build_consensus_matchups, price_bookmaker_sides
from mm26_xgb.ledger import settle_ledger
from mm26_xgb.markets import (
    ODDS_API_HOST,
    flatten_odds_api_response,
    flatten_scores_api_response,
    load_api_key,
)
from mm26_xgb.modeling import (
    fit_external_prior_models,
    load_bundle,
    merge_market_history_into_rows,
    predict_matchups,
    save_bundle,
    train_model_bundle,
)
from mm26_xgb.features import build_team_features
from mm26_xgb.external_features import load_external_team_features, merge_external_team_features
from mm26_xgb.config import TrainConfig
from mm26_xgb.cli import build_parser as build_mm26_parser


ET = ZoneInfo("America/New_York")
UTC = timezone.utc
DEFAULT_BOOKMAKER_LABEL = "prep_bundle"
_BOOTSTRAP_READY_NOTICE_EMITTED = False


@dataclass(frozen=True)
class AppPaths:
    root: Path
    workspace: Path
    outputs: Path
    cache: Path
    api_cache: Path
    artifacts: Path
    assets: Path
    kaggle_zip: Path
    kaggle_data: Path
    prep_dir: Path
    trained_model_dir: Path
    win_loss_csv: Path
    ats_csv: Path
    ou_csv: Path
    tempo_csv: Path
    lineup_csv: Path
    defaults_json: Path


def build_paths() -> AppPaths:
    workspace = PROJECT_ROOT / "workspace"
    return AppPaths(
        root=PROJECT_ROOT,
        workspace=workspace,
        outputs=workspace / "outputs",
        cache=workspace / "cache",
        api_cache=workspace / "cache" / "odds_api",
        artifacts=workspace / "artifacts",
        assets=PROJECT_ROOT / "assets",
        kaggle_zip=PROJECT_ROOT / "assets" / "march-machine-learning-mania-2026.zip",
        kaggle_data=workspace / "data" / "kaggle_data",
        prep_dir=PROJECT_ROOT / "assets" / "mm26_first_four_round1_prep",
        trained_model_dir=PROJECT_ROOT / "trained_kaggle_market_model",
        win_loss_csv=PROJECT_ROOT / "assets" / "win_loss.csv",
        ats_csv=PROJECT_ROOT / "assets" / "ats.csv",
        ou_csv=PROJECT_ROOT / "assets" / "ou.csv",
        tempo_csv=PROJECT_ROOT / "assets" / "tempo_features.csv",
        lineup_csv=PROJECT_ROOT / "assets" / "lineup_features.csv",
        defaults_json=PROJECT_ROOT / "config" / "defaults.json",
    )


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def safe_score_pair_series(team_a: pd.Series, team_b: pd.Series) -> pd.Series:
    a = safe_int_series(team_a)
    b = safe_int_series(team_b)
    out = pd.Series("", index=a.index, dtype="object")
    mask = a.notna() & b.notna()
    if mask.any():
        out.loc[mask] = a.loc[mask].astype(str) + "-" + b.loc[mask].astype(str)
    return out


def write_unresolved_market_rows(raw_rows: pd.DataFrame, output_path: Path) -> int:
    if raw_rows.empty or not {"TeamAID", "TeamBID"}.issubset(raw_rows.columns):
        return 0
    unresolved = raw_rows.loc[raw_rows["TeamAID"].isna() | raw_rows["TeamBID"].isna()].copy()
    if unresolved.empty:
        with contextlib.suppress(FileNotFoundError):
            output_path.unlink()
        return 0
    keep_cols = [
        col
        for col in [
            "SnapshotRequestedUTC",
            "SnapshotTimeUTC",
            "CommenceTime",
            "EventID",
            "Bookmaker",
            "TeamA",
            "TeamB",
            "TeamAID",
            "TeamBID",
        ]
        if col in unresolved.columns
    ]
    unresolved = unresolved[keep_cols].drop_duplicates()
    sort_cols = [col for col in ["CommenceTime", "TeamA", "TeamB", "SnapshotRequestedUTC"] if col in unresolved.columns]
    if sort_cols:
        unresolved = unresolved.sort_values(sort_cols).reset_index(drop=True)
    ensure_dir(output_path.parent)
    unresolved.to_csv(output_path, index=False)
    return int(len(unresolved))


def load_settings(paths: AppPaths) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    if paths.defaults_json.exists():
        settings.update(json.loads(paths.defaults_json.read_text(encoding="utf-8")))
    env_key = os.getenv("ODDS_API_KEY")
    if env_key:
        settings["odds_api_key"] = env_key
    return settings


def bootstrap(paths: AppPaths, force: bool = False) -> None:
    global _BOOTSTRAP_READY_NOTICE_EMITTED

    ensure_dir(paths.workspace)
    ensure_dir(paths.outputs)
    ensure_dir(paths.cache)
    ensure_dir(paths.api_cache)
    ensure_dir(paths.artifacts)
    ensure_dir(paths.kaggle_data.parent)

    required = [
        paths.kaggle_data / "MTeams.csv",
        paths.kaggle_data / "MRegularSeasonDetailedResults.csv",
        paths.kaggle_data / "MNCAATourneyDetailedResults.csv",
        paths.kaggle_data / "MNCAATourneySeeds.csv",
    ]
    if force and paths.kaggle_data.exists():
        shutil.rmtree(paths.kaggle_data)
        _BOOTSTRAP_READY_NOTICE_EMITTED = False
    if not all(path.exists() for path in required):
        if not paths.kaggle_zip.exists():
            raise FileNotFoundError(f"Kaggle zip not found: {paths.kaggle_zip}")
        log(f"Extracting Kaggle data to {paths.kaggle_data} ...")
        with zipfile.ZipFile(paths.kaggle_zip, "r") as zf:
            zf.extractall(paths.kaggle_data)
        _BOOTSTRAP_READY_NOTICE_EMITTED = True
    elif not _BOOTSTRAP_READY_NOTICE_EMITTED:
        log(f"Kaggle data already extracted: {paths.kaggle_data}")
        _BOOTSTRAP_READY_NOTICE_EMITTED = True


def read_api_key(settings: dict[str, Any], explicit: str | None = None) -> str:
    candidate = explicit or settings.get("odds_api_key")
    api_key = load_api_key(candidate)
    if not api_key:
        raise ValueError(
            "No Odds API key found. Set ODDS_API_KEY or edit config/defaults.json or pass --api-key."
        )
    return api_key


def build_alias_lookup(data, aliases_csv: str | Path | None = None) -> dict[str, int]:
    alias_source = aliases_csv
    if alias_source is None:
        for candidate in [
            PROJECT_ROOT / "assets" / "team_aliases.csv",
            PROJECT_ROOT / "config" / "team_aliases.csv",
            PROJECT_ROOT / "workspace" / "team_aliases.csv",
        ]:
            if candidate.exists():
                alias_source = candidate
                break
    extra = load_aliases_csv(alias_source, data.teams) if alias_source else {}
    return team_lookup(data.teams, data.spellings, extra)


def pair_key(team_a: int, team_b: int) -> str:
    a = int(team_a)
    b = int(team_b)
    return f"{min(a, b)}-{max(a, b)}"


def format_pct(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}%"


def format_line(value: Any, decimals: int = 1, signed: bool = True) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    return f"{number:+.{decimals}f}" if signed else f"{number:.{decimals}f}"


def format_moneyline(value: Any) -> str:
    if pd.isna(value):
        return ""
    odds = int(round(float(value)))
    return f"+{odds}" if odds > 0 else str(odds)


def format_num(value: Any, decimals: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}"


def format_ratio(numerator: Any, denominator: Any) -> str:
    try:
        num = int(numerator)
        den = int(denominator)
    except Exception:
        return ""
    return f"{num}/{den}"


def bool_to_flag(value: Any) -> str:
    if pd.isna(value):
        return ""
    return "Y" if bool(value) else "N"


def safe_probability_clip(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(1e-6, 1.0 - 1e-6)


def ordered_round_summary(round_summary: pd.DataFrame) -> pd.DataFrame:
    if round_summary.empty or "RoundLabel" not in round_summary.columns:
        return round_summary
    order = {
        "First Four": 1,
        "Round of 64": 2,
        "Round of 32": 3,
        "Sweet 16": 4,
        "Elite 8": 5,
        "Final Four": 6,
        "Championship": 7,
        "Tournament": 8,
    }
    out = round_summary.copy()
    out["_round_order"] = out["RoundLabel"].map(order).fillna(999)
    out = out.sort_values(["_round_order", "RoundLabel"]).drop(columns=["_round_order"])
    return out.reset_index(drop=True)


def normalize_cache_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_cache_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [normalize_cache_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def build_api_cache_key(path: str, params: dict[str, Any]) -> str:
    payload = {
        "path": path,
        "params": normalize_cache_value(params),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def api_cache_file(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.json"


def describe_cache_dir(cache_dir: Path, pattern: str = "*") -> dict[str, Any]:
    ensure_dir(cache_dir)
    files = sorted(cache_dir.glob(pattern))
    total_bytes = sum(file.stat().st_size for file in files if file.exists())
    return {
        "cache_dir": str(cache_dir),
        "entries": int(len(files)),
        "size_bytes": int(total_bytes),
        "size_mb": round(total_bytes / (1024 * 1024), 3),
    }


def describe_api_cache(paths: AppPaths) -> dict[str, Any]:
    return describe_cache_dir(paths.api_cache, "*.json")


def describe_season_state_cache(paths: AppPaths) -> dict[str, Any]:
    return describe_cache_dir(paths.cache / "season_state", "*.csv")


def read_api_cache_entry(
    cache_dir: Path,
    cache_key: str,
    max_age_seconds: int | None = None,
) -> dict[str, Any] | None:
    path = api_cache_file(cache_dir, cache_key)
    if not path.exists():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    created_at = entry.get("created_at_utc")
    if max_age_seconds is not None and created_at:
        try:
            created_dt = pd.to_datetime(created_at, utc=True).to_pydatetime()
            age_seconds = (datetime.now(UTC) - created_dt).total_seconds()
        except Exception:
            return None
        if age_seconds > max_age_seconds:
            return None
    return entry


def write_api_cache_entry(
    cache_dir: Path,
    cache_key: str,
    request_path: str,
    request_params: dict[str, Any],
    payload: Any,
    headers: dict[str, str],
) -> Path:
    ensure_dir(cache_dir)
    path = api_cache_file(cache_dir, cache_key)
    entry = {
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "request_path": request_path,
        "request_params": normalize_cache_value(request_params),
        "headers": headers,
        "payload": payload,
    }
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(entry, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp_path.replace(path)
    return path


def clear_api_cache(paths: AppPaths, include_season_state: bool = False) -> dict[str, Any]:
    api_before = describe_api_cache(paths)
    season_state_before = describe_season_state_cache(paths)
    if paths.api_cache.exists():
        shutil.rmtree(paths.api_cache)
    ensure_dir(paths.api_cache)

    season_state_dir_path = paths.cache / "season_state"
    if include_season_state and season_state_dir_path.exists():
        shutil.rmtree(season_state_dir_path)
    if include_season_state:
        ensure_dir(season_state_dir_path)

    api_after = describe_api_cache(paths)
    result: dict[str, Any] = {
        "api_cache": {
            "before": api_before,
            "after": api_after,
        },
        "season_state_cache": {
            "before": season_state_before,
            "after": describe_season_state_cache(paths) if include_season_state else season_state_before,
            "cleared": bool(include_season_state),
        },
    }
    return result


def odds_api_get(
    path: str,
    api_key: str,
    params: dict[str, Any],
    timeout: int = 60,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    max_age_seconds: int | None = None,
    max_retries: int = 4,
    retry_backoff_seconds: float = 3.0,
) -> tuple[Any, dict[str, str], dict[str, Any]]:
    request_params = dict(params)
    cache_meta = {
        "cache_hit": False,
        "cache_key": None,
        "cache_path": None,
        "cache_created_at": None,
        "request_attempts": 0,
    }

    if cache_dir is not None and use_cache:
        cache_key = build_api_cache_key(path, request_params)
        cache_path = api_cache_file(cache_dir, cache_key)
        cache_meta.update({
            "cache_key": cache_key,
            "cache_path": str(cache_path),
        })
        if not refresh_cache:
            cached = read_api_cache_entry(cache_dir, cache_key, max_age_seconds=max_age_seconds)
            if cached is not None:
                cache_meta.update({
                    "cache_hit": True,
                    "cache_created_at": cached.get("created_at_utc"),
                    "request_attempts": 0,
                })
                return cached.get("payload"), dict(cached.get("headers") or {}), cache_meta

    request_params["apiKey"] = api_key
    last_exc: Exception | None = None
    attempts = max(1, int(max_retries))

    for attempt in range(1, attempts + 1):
        cache_meta["request_attempts"] = attempt
        try:
            response = requests.get(f"{ODDS_API_HOST}{path}", params=request_params, timeout=timeout)
            if response.status_code == 429:
                raise RuntimeError(
                    "The Odds API returned 429 (rate limit / quota). Reduce the pull range, increase the interval, or wait for quota reset."
                )
            if response.status_code >= 500:
                response.raise_for_status()
            response.raise_for_status()
            payload = response.json()
            headers = dict(response.headers)

            if cache_dir is not None and use_cache:
                cache_key = build_api_cache_key(path, params)
                cache_path = write_api_cache_entry(cache_dir, cache_key, path, params, payload, headers)
                cache_meta.update({
                    "cache_key": cache_key,
                    "cache_path": str(cache_path),
                    "cache_created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                })

            return payload, headers, cache_meta
        except RuntimeError:
            raise
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            retryable = isinstance(
                exc,
                (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.HTTPError,
                ),
            )
            if isinstance(exc, requests.exceptions.HTTPError):
                status = getattr(exc.response, "status_code", None)
                retryable = status is not None and int(status) >= 500
            if attempt >= attempts or not retryable:
                break
            sleep_seconds = float(retry_backoff_seconds) * attempt
            time.sleep(max(0.0, sleep_seconds))

    assert last_exc is not None
    raise last_exc


def fetch_historical_snapshot(
    api_key: str,
    sport: str,
    regions: str,
    markets: str,
    snapshot_date_utc: str,
    bookmakers: str | None = None,
    commence_time_from: str | None = None,
    commence_time_to: str | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    timeout: int = 60,
    max_retries: int = 4,
    retry_backoff_seconds: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    params: dict[str, Any] = {
        "regions": regions,
        "markets": markets,
        "date": snapshot_date_utc,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    if commence_time_from:
        params["commenceTimeFrom"] = commence_time_from
    if commence_time_to:
        params["commenceTimeTo"] = commence_time_to

    payload, headers, cache_meta = odds_api_get(
        path=f"/v4/historical/sports/{sport}/odds",
        api_key=api_key,
        params=params,
        timeout=timeout,
        cache_dir=cache_dir,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    data = payload.get("data", []) if isinstance(payload, dict) else []
    flat = flatten_odds_api_response(list(data))
    meta = {
        "requested_snapshot_utc": snapshot_date_utc,
        "snapshot_time_utc": payload.get("timestamp") if isinstance(payload, dict) else None,
        "previous_snapshot_utc": payload.get("previous_timestamp") if isinstance(payload, dict) else None,
        "next_snapshot_utc": payload.get("next_timestamp") if isinstance(payload, dict) else None,
        "quota_remaining": headers.get("x-requests-remaining"),
        "quota_used": headers.get("x-requests-used"),
        "quota_last": headers.get("x-requests-last"),
        "rows": int(len(flat)),
        **cache_meta,
    }
    return flat, meta


def fetch_live_odds(
    api_key: str,
    sport: str,
    regions: str,
    markets: str,
    bookmakers: str | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    max_age_seconds: int | None = None,
    timeout: int = 60,
    max_retries: int = 4,
    retry_backoff_seconds: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    params: dict[str, Any] = {
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    payload, headers, cache_meta = odds_api_get(
        path=f"/v4/sports/{sport}/odds",
        api_key=api_key,
        params=params,
        timeout=timeout,
        cache_dir=cache_dir,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        max_age_seconds=max_age_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    flat = flatten_odds_api_response(list(payload))
    meta = {
        "quota_remaining": headers.get("x-requests-remaining"),
        "quota_used": headers.get("x-requests-used"),
        "quota_last": headers.get("x-requests-last"),
        "rows": int(len(flat)),
        **cache_meta,
    }
    return flat, meta


def resolve_market_team_identifier(
    value: Any,
    alias_lookup: dict[str, int],
) -> int:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise KeyError(f"Could not resolve team identifier: {value!r}")

    try:
        return int(resolve_team_identifier(text, alias_lookup))
    except Exception:
        pass

    # Odds API team names often include mascot suffixes that Kaggle names/spellings omit,
    # e.g. "Duke Blue Devils" -> "Duke" or "North Carolina Tar Heels" -> "North Carolina".
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for end in range(len(tokens) - 1, 0, -1):
        candidate = normalize_team_name(" ".join(tokens[:end]))
        if candidate in alias_lookup:
            return int(alias_lookup[candidate])

    norm = normalize_team_name(text)
    prefix_hits = [key for key in alias_lookup.keys() if norm.startswith(key)]
    if prefix_hits:
        best = max(prefix_hits, key=len)
        return int(alias_lookup[best])

    raise KeyError(f"Could not resolve team identifier: {value!r}")


def safe_resolve_market_team_ids(
    flat: pd.DataFrame,
    alias_lookup: dict[str, int],
    season: int,
) -> tuple[pd.DataFrame, list[str]]:
    if flat.empty:
        out = flat.copy()
        out["Season"] = int(season)
        out["TeamAID"] = np.nan
        out["TeamBID"] = np.nan
        return out, []

    out = flat.copy()
    team_a_ids: list[float] = []
    team_b_ids: list[float] = []
    unresolved: set[str] = set()

    for row in out.to_dict("records"):
        team_a = str(row.get("TeamA", "")).strip()
        team_b = str(row.get("TeamB", "")).strip()
        try:
            team_a_id = int(resolve_market_team_identifier(team_a, alias_lookup))
        except Exception:
            team_a_id = math.nan
            if team_a:
                unresolved.add(team_a)
        try:
            team_b_id = int(resolve_market_team_identifier(team_b, alias_lookup))
        except Exception:
            team_b_id = math.nan
            if team_b:
                unresolved.add(team_b)
        team_a_ids.append(team_a_id)
        team_b_ids.append(team_b_id)

    out["Season"] = int(season)
    out["TeamAID"] = team_a_ids
    out["TeamBID"] = team_b_ids
    return out, sorted(unresolved)


def select_latest_pregame_snapshot(
    raw_snapshots: pd.DataFrame,
    pregame_buffer_minutes: int = 5,
) -> pd.DataFrame:
    if raw_snapshots.empty:
        return raw_snapshots.copy()

    work = raw_snapshots.copy()
    if "SnapshotTimeUTC" not in work.columns:
        raise ValueError("raw_snapshots must include SnapshotTimeUTC.")
    work["SnapshotTimeParsed"] = pd.to_datetime(work["SnapshotTimeUTC"], utc=True, errors="coerce")
    work["CommenceTimeParsed"] = pd.to_datetime(work["CommenceTime"], utc=True, errors="coerce")
    work = work.sort_values(["EventID", "Bookmaker", "SnapshotTimeParsed"]).reset_index(drop=True)

    buffer_delta = pd.Timedelta(minutes=int(pregame_buffer_minutes))

    kept_parts: list[pd.DataFrame] = []
    group_cols = ["EventID", "Bookmaker"]
    for _, group in work.groupby(group_cols, dropna=False):
        commence = group["CommenceTimeParsed"].dropna()
        if commence.empty:
            picked = group.tail(1)
        else:
            commence_time = commence.iloc[0]
            eligible = group.loc[group["SnapshotTimeParsed"] <= (commence_time - buffer_delta)]
            if eligible.empty:
                eligible = group.loc[group["SnapshotTimeParsed"] <= commence_time]
            if eligible.empty:
                eligible = group
            picked = eligible.tail(1)
        kept_parts.append(picked)

    closing = pd.concat(kept_parts, ignore_index=True)
    closing = closing.drop(columns=["SnapshotTimeParsed", "CommenceTimeParsed"], errors="ignore")
    return closing


def date_range_from_year(
    paths: AppPaths,
    year: int,
    scope: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[datetime, datetime]:
    if start_date and end_date:
        start_dt = pd.Timestamp(start_date, tz="UTC").to_pydatetime()
        end_dt = pd.Timestamp(end_date, tz="UTC").to_pydatetime()
        return start_dt, end_dt

    bootstrap(paths)
    data = load_kaggle_data(paths.kaggle_data)
    season_row = pd.read_csv(paths.kaggle_data / "MSeasons.csv").loc[lambda d: d["Season"] == int(year)]
    if season_row.empty:
        if scope == "tournament":
            return datetime(year, 3, 15, 0, 0, tzinfo=UTC), datetime(year, 4, 9, 23, 55, tzinfo=UTC)
        return datetime(year - 1, 11, 1, 0, 0, tzinfo=UTC), datetime(year, 4, 15, 23, 55, tzinfo=UTC)

    day_zero = pd.to_datetime(season_row.iloc[0]["DayZero"])
    if scope == "tournament":
        results = data.tourney.loc[data.tourney["Season"] == int(year)].copy()
        if results.empty:
            return datetime(year, 3, 15, 0, 0, tzinfo=UTC), datetime(year, 4, 9, 23, 55, tzinfo=UTC)
    else:
        regular = data.regular.loc[data.regular["Season"] == int(year)].copy()
        tourney = data.tourney.loc[data.tourney["Season"] == int(year)].copy()
        results = pd.concat([regular, tourney], ignore_index=True)
        if results.empty:
            return datetime(year - 1, 11, 1, 0, 0, tzinfo=UTC), datetime(year, 4, 15, 23, 55, tzinfo=UTC)

    min_day = int(results["DayNum"].min())
    max_day = int(results["DayNum"].max())
    start_local = (day_zero + pd.to_timedelta(min_day, unit="D")).to_pydatetime().replace(tzinfo=ET)
    end_local = (day_zero + pd.to_timedelta(max_day, unit="D")).to_pydatetime().replace(
        hour=23, minute=55, second=0, tzinfo=ET
    )
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def iter_snapshots(start_dt_utc: datetime, end_dt_utc: datetime, interval_hours: int) -> list[datetime]:
    if interval_hours <= 0:
        raise ValueError("--interval-hours must be > 0")
    points: list[datetime] = []
    current = start_dt_utc
    delta = timedelta(hours=int(interval_hours))
    while current <= end_dt_utc:
        points.append(current)
        current += delta
    if not points or points[-1] < end_dt_utc:
        points.append(end_dt_utc)
    return points


def estimate_historical_credit_cost(markets: str, regions: str, calls: int) -> int:
    market_count = len([x for x in markets.split(",") if x.strip()])
    region_count = len([x for x in regions.split(",") if x.strip()])
    return int(calls * 10 * market_count * region_count)



def validate_bundle_runtime_compatibility(artifact_dir: Path) -> tuple[bool, str]:
    bundle_path = Path(artifact_dir) / "model_bundle.joblib"
    if not bundle_path.exists():
        return False, f"Missing bundle file: {bundle_path}"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundle = load_bundle(artifact_dir)

            main_cols = list(getattr(bundle, "feature_columns", []) or [])
            if main_cols and hasattr(bundle.win_model, "predict_proba"):
                main_fill = getattr(bundle, "fill_values", {}) or {}
                probe_main = pd.DataFrame([{col: float(main_fill.get(col, 0.0)) for col in main_cols}])
                bundle.win_model.predict_proba(probe_main)

            prior_models = getattr(bundle, "external_prior_models", {}) or {}
            prior_cols = list(getattr(bundle, "external_prior_feature_columns", []) or [])
            win_prior = prior_models.get("win")
            if win_prior is not None and prior_cols and hasattr(win_prior, "predict_proba"):
                prior_fill = getattr(bundle, "external_prior_fill_values", {}) or {}
                probe_prior = pd.DataFrame([{col: float(prior_fill.get(col, 0.0)) for col in prior_cols}])
                win_prior.predict_proba(probe_prior)

        return True, ""
    except Exception as exc:
        return False, str(exc)

def train_model_if_needed(
    paths: AppPaths,
    target_season: int,
    force_retrain: bool = False,
) -> Path:
    bootstrap(paths)
    target_season = int(target_season)

    artifact_dir = paths.artifacts / f"mm26_market_{target_season}"
    model_path = artifact_dir / "model_bundle.joblib"

    if not force_retrain:
        if model_path.exists():
            ok, reason = validate_bundle_runtime_compatibility(artifact_dir)
            if ok:
                return artifact_dir
            log(f"Existing local model bundle is not runtime-compatible; retraining locally. Reason: {reason}")

        if target_season == 2026 and paths.trained_model_dir.exists():
            ok, reason = validate_bundle_runtime_compatibility(paths.trained_model_dir)
            if ok:
                return paths.trained_model_dir
            log(f"Bundled 2026 model is not runtime-compatible on this machine; retraining locally. Reason: {reason}")

    parser = build_mm26_parser()
    cli_args = [
        "train",
        "--data-dir",
        str(paths.kaggle_data),
        "--artifact-dir",
        str(artifact_dir),
        "--target-season",
        str(target_season),
        "--skip-backtest",
    ]
    if target_season == 2026:
        cli_args.extend(
            [
                "--win-loss-csv",
                str(paths.win_loss_csv),
                "--ats-csv",
                str(paths.ats_csv),
                "--ou-csv",
                str(paths.ou_csv),
            ]
        )
    args = parser.parse_args(cli_args)
    log(f"Training model bundle for {target_season} ...")
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        rc = int(args.func(args))
    if rc != 0:
        detail = captured.getvalue().strip()
        if detail:
            raise RuntimeError(f"Training failed with exit code {rc}: {detail}")
        raise RuntimeError(f"Training failed with exit code {rc}")
    log(f"Saved model artifacts to: {artifact_dir}")
    return artifact_dir

def load_team_features_from_artifacts(artifact_dir: Path) -> pd.DataFrame:
    team_features_path = artifact_dir / "team_features.csv"
    if not team_features_path.exists():
        raise FileNotFoundError(f"team_features.csv not found in {artifact_dir}")
    return pd.read_csv(team_features_path, low_memory=False)


def build_actual_games(
    paths: AppPaths,
    year: int,
    scope: str,
) -> pd.DataFrame:
    bootstrap(paths)
    data = load_kaggle_data(paths.kaggle_data)
    seasons = pd.read_csv(paths.kaggle_data / "MSeasons.csv")
    season_row = seasons.loc[seasons["Season"] == int(year)]
    if season_row.empty:
        raise ValueError(f"Season {year} not present in Kaggle data.")
    day_zero = pd.to_datetime(season_row.iloc[0]["DayZero"])

    regular = data.regular.loc[data.regular["Season"] == int(year)].copy()
    regular["CompetitionPhase"] = "regular_season"
    regular["RoundLabel"] = "Regular Season"

    tourney = data.tourney.loc[data.tourney["Season"] == int(year)].copy()
    tourney["CompetitionPhase"] = "tournament"
    unique_days = sorted(int(x) for x in tourney["DayNum"].dropna().unique().tolist())
    round_labels = [
        "First Four",
        "First Four",
        "Round of 64",
        "Round of 64",
        "Round of 32",
        "Round of 32",
        "Sweet 16",
        "Sweet 16",
        "Elite 8",
        "Elite 8",
        "Final Four",
        "Championship",
    ]
    day_to_round = {day: round_labels[idx] if idx < len(round_labels) else f"Tournament Day {idx + 1}" for idx, day in enumerate(unique_days)}
    tourney["RoundLabel"] = tourney["DayNum"].map(day_to_round).fillna("Tournament")

    if scope == "tournament":
        games = tourney
    elif scope == "season":
        games = pd.concat([regular, tourney], ignore_index=True)
    else:
        raise ValueError("--scope must be 'tournament' or 'season'")

    games = games.copy()
    games["GameDateET"] = pd.to_datetime(day_zero) + pd.to_timedelta(games["DayNum"], unit="D")
    games["GameDateET"] = pd.to_datetime(games["GameDateET"]).dt.date.astype(str)
    games["PairKey"] = games.apply(lambda r: pair_key(r["WTeamID"], r["LTeamID"]), axis=1)
    games["ActualWinnerID"] = games["WTeamID"].astype(int)
    games["ActualLoserID"] = games["LTeamID"].astype(int)
    return games[
        [
            "Season",
            "DayNum",
            "CompetitionPhase",
            "RoundLabel",
            "GameDateET",
            "WTeamID",
            "LTeamID",
            "WScore",
            "LScore",
            "ActualWinnerID",
            "ActualLoserID",
            "PairKey",
        ]
    ].reset_index(drop=True)


def match_market_events_to_actuals(
    consensus: pd.DataFrame,
    actual_games: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if consensus.empty:
        return consensus.copy(), pd.DataFrame()

    work = consensus.copy()
    work["PairKey"] = work.apply(lambda r: pair_key(r["TeamAID"], r["TeamBID"]), axis=1)
    work["CommenceTimeParsed"] = pd.to_datetime(work["CommenceTime"], utc=True, errors="coerce")
    work["CommenceDateET"] = work["CommenceTimeParsed"].dt.tz_convert(ET).dt.date.astype(str)

    actual_by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in actual_games.to_dict("records"):
        actual_by_pair.setdefault(str(row["PairKey"]), []).append(row)

    matched_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []

    for row in work.to_dict("records"):
        candidates = list(actual_by_pair.get(str(row["PairKey"]), []))
        if not candidates:
            unmatched_rows.append(row)
            continue

        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            market_date = pd.Timestamp(row.get("CommenceDateET")) if row.get("CommenceDateET") else None
            def score_candidate(candidate: dict[str, Any]) -> tuple[int, int]:
                candidate_date = pd.Timestamp(candidate["GameDateET"])
                day_gap = abs((candidate_date - market_date).days) if market_date is not None else 999
                phase_penalty = 0 if str(candidate.get("CompetitionPhase")) == "tournament" else 1
                return (day_gap, phase_penalty)
            chosen = sorted(candidates, key=score_candidate)[0]

        if int(chosen["WTeamID"]) == int(row["TeamAID"]) and int(chosen["LTeamID"]) == int(row["TeamBID"]):
            actual_score_a = int(chosen["WScore"])
            actual_score_b = int(chosen["LScore"])
            actual_winner_id = int(chosen["WTeamID"])
        elif int(chosen["WTeamID"]) == int(row["TeamBID"]) and int(chosen["LTeamID"]) == int(row["TeamAID"]):
            actual_score_a = int(chosen["LScore"])
            actual_score_b = int(chosen["WScore"])
            actual_winner_id = int(chosen["WTeamID"])
        else:
            unmatched_rows.append(row)
            continue

        enriched = dict(row)
        enriched.update(
            {
                "ActualScoreTeamA": actual_score_a,
                "ActualScoreTeamB": actual_score_b,
                "ActualMargin": actual_score_a - actual_score_b,
                "ActualTotal": actual_score_a + actual_score_b,
                "ActualWinnerTeamID": actual_winner_id,
                "CompetitionPhase": chosen["CompetitionPhase"],
                "RoundLabel": chosen["RoundLabel"],
                "GameDateET": chosen["GameDateET"],
                "DayNum": int(chosen["DayNum"]),
            }
        )
        matched_rows.append(enriched)

    matched = pd.DataFrame(matched_rows)
    unmatched = pd.DataFrame(unmatched_rows)
    if matched.empty:
        return matched, unmatched

    matched = matched.copy()
    matched["__ActualGameKey"] = matched["PairKey"].astype(str) + "|" + matched["DayNum"].astype(str)
    matched["__DateGap"] = (
        pd.to_datetime(matched["CommenceDateET"], errors="coerce")
        - pd.to_datetime(matched["GameDateET"], errors="coerce")
    ).abs().dt.days.fillna(9999)
    matched["__BookCountSort"] = pd.to_numeric(matched.get("MarketBookCount"), errors="coerce").fillna(0)
    matched["__CommenceSort"] = pd.to_datetime(matched["CommenceTime"], utc=True, errors="coerce")

    matched_sorted = matched.sort_values(
        ["__ActualGameKey", "__DateGap", "__BookCountSort", "__CommenceSort"],
        ascending=[True, True, False, True],
    )
    deduped = matched_sorted.drop_duplicates(subset=["__ActualGameKey"], keep="first").copy()

    dropped = matched_sorted.loc[~matched_sorted.index.isin(deduped.index)].copy()
    if not dropped.empty:
        dropped["UnmatchedReason"] = "duplicate_market_event_for_actual_game"
        unmatched = pd.concat([unmatched, dropped.drop(columns=["__ActualGameKey", "__DateGap", "__BookCountSort", "__CommenceSort"], errors="ignore")], ignore_index=True)

    deduped = deduped.drop(columns=["__ActualGameKey", "__DateGap", "__BookCountSort", "__CommenceSort"], errors="ignore")
    return deduped.reset_index(drop=True), unmatched.reset_index(drop=True)


def build_scores_frame_from_matched_events(matched_events: pd.DataFrame) -> pd.DataFrame:
    if matched_events.empty:
        return pd.DataFrame(
            columns=["EventID", "CommenceTime", "TeamA", "TeamB", "ScoreTeamA", "ScoreTeamB", "Completed"]
        )
    out = matched_events[
        ["EventID", "CommenceTime", "TeamA", "TeamB", "ActualScoreTeamA", "ActualScoreTeamB"]
    ].copy()
    out = out.rename(columns={"ActualScoreTeamA": "ScoreTeamA", "ActualScoreTeamB": "ScoreTeamB"})
    out["Completed"] = True
    return out


def prepare_backtest_predictions(
    matched_consensus: pd.DataFrame,
    artifact_dir: Path,
) -> pd.DataFrame:
    if matched_consensus.empty:
        return matched_consensus.copy()
    bundle = load_bundle(artifact_dir)
    team_features = load_team_features_from_artifacts(artifact_dir)
    work = matched_consensus.copy()
    for missing_col in ["MoneylineHold", "SpreadHold", "TotalHold"]:
        if missing_col not in work.columns:
            work[missing_col] = np.nan
    inputs = work[
        [
            "Season",
            "TeamAID",
            "TeamBID",
            "EventID",
            "CommenceTime",
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
        ]
    ].copy()
    return predict_matchups(inputs, team_features, bundle)


def compute_backtest_game_report(
    matched_consensus: pd.DataFrame,
    predictions: pd.DataFrame,
    team_names: dict[int, str],
) -> pd.DataFrame:
    if matched_consensus.empty or predictions.empty:
        return pd.DataFrame()

    report = matched_consensus.merge(
        predictions,
        on=["Season", "TeamAID", "TeamBID", "EventID", "CommenceTime"],
        how="inner",
        suffixes=("_actual", ""),
    )
    if report.empty:
        return report

    report["TeamAName"] = report["TeamAID"].map(team_names)
    report["TeamBName"] = report["TeamBID"].map(team_names)
    report["PredWinnerTeamName"] = report["PredWinnerTeamID"].map(team_names)
    report["ActualWinnerTeamName"] = report["ActualWinnerTeamID"].map(team_names)

    pred_winner_id = safe_int_series(report["PredWinnerTeamID"])
    actual_winner_id = safe_int_series(report["ActualWinnerTeamID"])
    team_a_id = safe_int_series(report["TeamAID"])

    winner_correct = pd.Series(pd.NA, index=report.index, dtype="boolean")
    winner_mask = pred_winner_id.notna() & actual_winner_id.notna()
    if winner_mask.any():
        winner_correct.loc[winner_mask] = pred_winner_id.loc[winner_mask] == actual_winner_id.loc[winner_mask]
    report["WinnerCorrect"] = winner_correct

    if "MarketProbTeamA" in report.columns:
        market_prob_a = pd.to_numeric(report["MarketProbTeamA"], errors="coerce")
        report["MarketFavoriteTeamID"] = np.where(market_prob_a.fillna(0.5) >= 0.5, report["TeamAID"], report["TeamBID"])
        market_favorite_id = safe_int_series(report["MarketFavoriteTeamID"])
        report["MarketFavoriteTeamName"] = market_favorite_id.map(team_names)

        market_favorite_correct = pd.Series(pd.NA, index=report.index, dtype="boolean")
        market_favorite_mask = market_favorite_id.notna() & actual_winner_id.notna()
        if market_favorite_mask.any():
            market_favorite_correct.loc[market_favorite_mask] = market_favorite_id.loc[market_favorite_mask] == actual_winner_id.loc[market_favorite_mask]
        report["MarketFavoriteCorrect"] = market_favorite_correct

        disagreement = pd.Series(False, index=report.index, dtype="boolean")
        disagreement_mask = pred_winner_id.notna() & market_favorite_id.notna()
        if disagreement_mask.any():
            disagreement.loc[disagreement_mask] = pred_winner_id.loc[disagreement_mask] != market_favorite_id.loc[disagreement_mask]
        report["ModelVsMarketDisagreement"] = disagreement
    else:
        report["MarketFavoriteTeamID"] = np.nan
        report["MarketFavoriteTeamName"] = ""
        report["MarketFavoriteCorrect"] = pd.Series(pd.NA, index=report.index, dtype="boolean")
        report["ModelVsMarketDisagreement"] = pd.Series(False, index=report.index, dtype="boolean")

    def spread_result(row: pd.Series) -> str:
        line = row.get("MarketSpreadTeamA")
        if pd.isna(line):
            return ""
        value = float(row["ActualMargin"]) + float(line)
        if value > 0:
            return "TeamA"
        if value < 0:
            return "TeamB"
        return "Push"

    def total_result(row: pd.Series) -> str:
        total_line = row.get("MarketTotal")
        if pd.isna(total_line):
            return ""
        value = float(row["ActualTotal"]) - float(total_line)
        if value > 0:
            return "Over"
        if value < 0:
            return "Under"
        return "Push"

    report["ModelSpreadLean"] = np.where(pd.to_numeric(report["ModelProbCoverTeamA"], errors="coerce").fillna(0.5) >= 0.5, "TeamA", "TeamB")
    report["ModelSpreadLeanTeamName"] = np.where(report["ModelSpreadLean"] == "TeamA", report["TeamAName"], report["TeamBName"])
    report["ActualSpreadResult"] = report.apply(spread_result, axis=1)
    report["ActualSpreadResultTeamName"] = np.select(
        [report["ActualSpreadResult"] == "TeamA", report["ActualSpreadResult"] == "TeamB", report["ActualSpreadResult"] == "Push"],
        [report["TeamAName"], report["TeamBName"], "Push"],
        default="",
    )
    report["SpreadCorrect"] = np.where(
        report["ActualSpreadResult"] == "Push",
        np.nan,
        report["ModelSpreadLean"] == report["ActualSpreadResult"],
    )

    report["ModelTotalLean"] = np.where(pd.to_numeric(report["ModelProbOver"], errors="coerce").fillna(0.5) >= 0.5, "Over", "Under")
    report["ActualTotalResult"] = report.apply(total_result, axis=1)
    report["TotalCorrect"] = np.where(
        report["ActualTotalResult"] == "Push",
        np.nan,
        report["ModelTotalLean"] == report["ActualTotalResult"],
    )

    report["MarginAbsError"] = (pd.to_numeric(report["PredMarginQ50"], errors="coerce") - pd.to_numeric(report["ActualMargin"], errors="coerce")).abs()
    report["TotalAbsError"] = (pd.to_numeric(report["PredTotalQ50"], errors="coerce") - pd.to_numeric(report["ActualTotal"], errors="coerce")).abs()
    report["ActualScore"] = safe_score_pair_series(report["ActualScoreTeamA"], report["ActualScoreTeamB"])
    report["PredictedScore"] = safe_score_pair_series(report["PredScoreTeamA"], report["PredScoreTeamB"])

    actual_winner_is_team_a = pd.Series(pd.NA, index=report.index, dtype="Int64")
    actual_winner_mask = actual_winner_id.notna() & team_a_id.notna()
    if actual_winner_mask.any():
        actual_winner_is_team_a.loc[actual_winner_mask] = (actual_winner_id.loc[actual_winner_mask] == team_a_id.loc[actual_winner_mask]).astype(int)
    report["ActualWinnerIsTeamA"] = actual_winner_is_team_a
    report["Matchup"] = report["TeamAName"].fillna(report["TeamA"]) + " vs " + report["TeamBName"].fillna(report["TeamB"])
    return report.sort_values(["CompetitionPhase", "DayNum", "CommenceTime", "EventID"]).reset_index(drop=True)


def build_backtest_round_summary(game_report: pd.DataFrame) -> pd.DataFrame:
    if game_report.empty:
        return pd.DataFrame()
    round_summary = (
        game_report.groupby("RoundLabel", dropna=False)
        .agg(
            Games=("EventID", "count"),
            MoneylineCorrect=("WinnerCorrect", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            MoneylineAccuracy=("WinnerCorrect", "mean"),
            SpreadGraded=("SpreadCorrect", lambda s: int(pd.Series(s).notna().sum())),
            SpreadCorrect=("SpreadCorrect", lambda s: int((pd.Series(s) == True).sum())),
            SpreadAccuracy=("SpreadCorrect", lambda s: pd.to_numeric(pd.Series(s), errors="coerce").mean()),
            TotalGraded=("TotalCorrect", lambda s: int(pd.Series(s).notna().sum())),
            TotalCorrect=("TotalCorrect", lambda s: int((pd.Series(s) == True).sum())),
            TotalAccuracy=("TotalCorrect", lambda s: pd.to_numeric(pd.Series(s), errors="coerce").mean()),
        )
        .reset_index()
    )
    return ordered_round_summary(round_summary)


def build_backtest_side_breakdown(game_report: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if game_report.empty:
        empty = pd.DataFrame()
        return {"moneyline": empty, "spread": empty, "total": empty}

    moneyline_source = game_report.copy()
    moneyline_source["MoneylinePickType"] = np.where(
        moneyline_source["ModelVsMarketDisagreement"] == True,
        "Model upset / disagreement",
        "Market favorite / agreement",
    )
    moneyline = (
        moneyline_source.groupby("MoneylinePickType", dropna=False)
        .agg(
            Games=("EventID", "count"),
            Correct=("WinnerCorrect", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            Accuracy=("WinnerCorrect", "mean"),
        )
        .reset_index()
        .sort_values("MoneylinePickType")
    )

    spread = (
        game_report.loc[game_report["SpreadCorrect"].notna()]
        .groupby("ModelSpreadLean", dropna=False)
        .agg(
            Graded=("EventID", "count"),
            Correct=("SpreadCorrect", lambda s: int((pd.Series(s) == True).sum())),
            Accuracy=("SpreadCorrect", lambda s: pd.to_numeric(pd.Series(s), errors="coerce").mean()),
        )
        .reset_index()
    )
    if not spread.empty:
        spread["Lean"] = np.where(spread["ModelSpreadLean"] == "TeamA", "TeamA side", "TeamB side")

    total = (
        game_report.loc[game_report["TotalCorrect"].notna()]
        .groupby("ModelTotalLean", dropna=False)
        .agg(
            Graded=("EventID", "count"),
            Correct=("TotalCorrect", lambda s: int((pd.Series(s) == True).sum())),
            Accuracy=("TotalCorrect", lambda s: pd.to_numeric(pd.Series(s), errors="coerce").mean()),
        )
        .reset_index()
        .sort_values("ModelTotalLean")
    )

    return {"moneyline": moneyline, "spread": spread, "total": total}



def build_backtest_accuracy_table(summary: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "Metric": "Winner / moneyline pick",
            "Correct": int(summary.get("winner_correct", 0) or 0),
            "Graded": int(summary.get("games_scored", 0) or 0),
            "Incorrect": int(summary.get("winner_incorrect", 0) or 0),
            "Pushes": 0,
            "Accuracy": summary.get("winner_accuracy"),
            "Definition": "Model picked the actual game winner.",
        },
        {
            "Metric": "Spread pick",
            "Correct": int(summary.get("spread_correct", 0) or 0),
            "Graded": int(summary.get("spread_graded", 0) or 0),
            "Incorrect": int(summary.get("spread_incorrect", 0) or 0),
            "Pushes": int(summary.get("spread_pushes", 0) or 0),
            "Accuracy": summary.get("spread_accuracy"),
            "Definition": "Model picked the correct side against the closing spread.",
        },
        {
            "Metric": "Total pick (Over/Under)",
            "Correct": int(summary.get("total_correct", 0) or 0),
            "Graded": int(summary.get("total_graded", 0) or 0),
            "Incorrect": int(summary.get("total_incorrect", 0) or 0),
            "Pushes": int(summary.get("total_pushes", 0) or 0),
            "Accuracy": summary.get("total_accuracy"),
            "Definition": "Model picked whether the final score finished over or under the closing total.",
        },
    ]
    return pd.DataFrame(rows)


def build_backtest_round_accuracy_table(round_summary: pd.DataFrame) -> pd.DataFrame:
    if round_summary.empty:
        return pd.DataFrame(
            columns=[
                "RoundLabel",
                "Games",
                "MoneylineCorrect",
                "MoneylineAccuracy",
                "SpreadCorrect",
                "SpreadGraded",
                "SpreadAccuracy",
                "TotalCorrect",
                "TotalGraded",
                "TotalAccuracy",
            ]
        )
    out = round_summary.copy()
    keep = [
        "RoundLabel",
        "Games",
        "MoneylineCorrect",
        "MoneylineAccuracy",
        "SpreadCorrect",
        "SpreadGraded",
        "SpreadAccuracy",
        "TotalCorrect",
        "TotalGraded",
        "TotalAccuracy",
    ]
    return out[keep].copy()


def build_backtest_summary(
    year: int,
    scope: str,
    artifact_dir: Path,
    raw_odds_path: Path,
    consensus: pd.DataFrame,
    matched_consensus: pd.DataFrame,
    unmatched: pd.DataFrame,
    game_report: pd.DataFrame,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "year": int(year),
        "scope": str(scope),
        "artifact_dir": str(artifact_dir),
        "raw_odds_csv": str(raw_odds_path),
        "market_events": int(len(consensus)),
        "matched_games": int(len(matched_consensus)),
        "unmatched_market_events": int(len(unmatched)),
        "training_note": (
            f"Holdout model for {int(year)} trained on historical NCAA tournament games before {int(year)}; "
            f"{int(year)} regular-season team features are used when scoring the {int(year)} tournament field."
        ),
    }
    if game_report.empty:
        summary.update(
            {
                "games_scored": 0,
                "winner_correct": 0,
                "winner_incorrect": 0,
                "winner_accuracy": None,
                "moneyline_brier": None,
                "moneyline_log_loss": None,
                "market_favorite_accuracy": None,
                "model_market_disagreement_games": 0,
                "model_market_disagreement_accuracy": None,
                "spread_graded": 0,
                "spread_correct": 0,
                "spread_incorrect": 0,
                "spread_pushes": 0,
                "spread_accuracy": None,
                "total_graded": 0,
                "total_correct": 0,
                "total_incorrect": 0,
                "total_pushes": 0,
                "total_accuracy": None,
                "margin_mae_mean": None,
                "margin_mae_median": None,
                "total_mae_mean": None,
                "total_mae_median": None,
            }
        )
        return summary

    outcome_team_a = pd.to_numeric(game_report["ActualWinnerIsTeamA"], errors="coerce").fillna(0.0)
    win_prob_team_a = safe_probability_clip(game_report["WinProbTeamA"])
    brier = float(((win_prob_team_a - outcome_team_a) ** 2).mean())
    log_loss = float((-(outcome_team_a * np.log(win_prob_team_a) + (1.0 - outcome_team_a) * np.log(1.0 - win_prob_team_a))).mean())

    spread_series = pd.to_numeric(game_report["SpreadCorrect"], errors="coerce")
    total_series = pd.to_numeric(game_report["TotalCorrect"], errors="coerce")
    disagreements = game_report.loc[game_report["ModelVsMarketDisagreement"] == True].copy()

    winner_correct = int(pd.Series(game_report["WinnerCorrect"]).fillna(False).astype(bool).sum())
    winner_total = int(len(game_report))
    spread_graded = int(spread_series.notna().sum())
    spread_correct = int((spread_series == 1).sum())
    total_graded = int(total_series.notna().sum())
    total_correct = int((total_series == 1).sum())

    summary.update(
        {
            "games_scored": winner_total,
            "winner_correct": winner_correct,
            "winner_incorrect": int(winner_total - winner_correct),
            "winner_accuracy": float(game_report["WinnerCorrect"].mean()),
            "moneyline_brier": brier,
            "moneyline_log_loss": log_loss,
            "market_favorite_accuracy": float(pd.to_numeric(game_report["MarketFavoriteCorrect"], errors="coerce").mean()) if game_report["MarketFavoriteCorrect"].notna().any() else None,
            "model_market_disagreement_games": int(len(disagreements)),
            "model_market_disagreement_accuracy": float(disagreements["WinnerCorrect"].mean()) if not disagreements.empty else None,
            "spread_graded": spread_graded,
            "spread_correct": spread_correct,
            "spread_incorrect": int(spread_graded - spread_correct),
            "spread_pushes": int((game_report["ActualSpreadResult"] == "Push").sum()),
            "spread_accuracy": float(spread_series.mean()) if spread_graded else None,
            "total_graded": total_graded,
            "total_correct": total_correct,
            "total_incorrect": int(total_graded - total_correct),
            "total_pushes": int((game_report["ActualTotalResult"] == "Push").sum()),
            "total_accuracy": float(total_series.mean()) if total_graded else None,
            "margin_mae_mean": float(pd.to_numeric(game_report["MarginAbsError"], errors="coerce").mean()),
            "margin_mae_median": float(pd.to_numeric(game_report["MarginAbsError"], errors="coerce").median()),
            "total_mae_mean": float(pd.to_numeric(game_report["TotalAbsError"], errors="coerce").mean()),
            "total_mae_median": float(pd.to_numeric(game_report["TotalAbsError"], errors="coerce").median()),
        }
    )
    return summary


def render_simple_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    return df.to_string(index=False)


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def build_backtest_markdown(
    summary: dict[str, Any],
    game_report: pd.DataFrame,
    round_summary: pd.DataFrame,
    side_breakdown: dict[str, pd.DataFrame],
) -> str:
    lines: list[str] = []
    lines.append(f"# {summary['year']} {summary['scope'].replace('_', ' ').title()} Market Backtest")
    lines.append("")
    lines.append(summary["training_note"])
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Market events matched to actual games: {summary['matched_games']} / {summary['market_events']}")
    lines.append(f"- Games scored: {summary.get('games_scored', 0)}")
    lines.append(f"- Moneyline correct: {format_ratio(summary.get('winner_correct'), summary.get('games_scored'))} ({format_pct(summary.get('winner_accuracy'))})")
    lines.append(f"- Spread correct: {format_ratio(summary.get('spread_correct'), summary.get('spread_graded'))} ({format_pct(summary.get('spread_accuracy'))}); pushes={summary.get('spread_pushes', 0)}")
    lines.append(f"- Total correct: {format_ratio(summary.get('total_correct'), summary.get('total_graded'))} ({format_pct(summary.get('total_accuracy'))}); pushes={summary.get('total_pushes', 0)}")
    lines.append(f"- Moneyline Brier score: {format_num(summary.get('moneyline_brier'), 4)}")
    lines.append(f"- Moneyline log loss: {format_num(summary.get('moneyline_log_loss'), 4)}")
    lines.append(f"- Market favorite moneyline accuracy: {format_pct(summary.get('market_favorite_accuracy'))}")
    lines.append(
        f"- Model/market moneyline disagreements: {summary.get('model_market_disagreement_games', 0)} games, accuracy {format_pct(summary.get('model_market_disagreement_accuracy'))}"
    )
    lines.append(f"- Margin MAE mean/median: {format_num(summary.get('margin_mae_mean'))} / {format_num(summary.get('margin_mae_median'))}")
    lines.append(f"- Total MAE mean/median: {format_num(summary.get('total_mae_mean'))} / {format_num(summary.get('total_mae_median'))}")
    lines.append("")

    if not round_summary.empty:
        display = round_summary.copy()
        for col in ["MoneylineAccuracy", "SpreadAccuracy", "TotalAccuracy"]:
            if col in display.columns:
                display[col] = display[col].map(format_pct)
        lines.append("## By round")
        lines.append("")
        lines.append(render_simple_table(display))
        lines.append("")

    moneyline_breakdown = side_breakdown.get("moneyline", pd.DataFrame())
    if not moneyline_breakdown.empty:
        display = moneyline_breakdown.copy().head(15)
        display["Accuracy"] = display["Accuracy"].map(format_pct)
        lines.append("## Moneyline pick breakdown")
        lines.append("")
        lines.append(render_simple_table(display[["MoneylinePickType", "Games", "Correct", "Accuracy"]]))
        lines.append("")

    spread_breakdown = side_breakdown.get("spread", pd.DataFrame())
    if not spread_breakdown.empty:
        display = spread_breakdown.copy()
        display["Accuracy"] = display["Accuracy"].map(format_pct)
        lines.append("## Spread pick breakdown")
        lines.append("")
        lines.append(render_simple_table(display[["ModelSpreadLean", "Lean", "Graded", "Correct", "Accuracy"]]))
        lines.append("")

    total_breakdown = side_breakdown.get("total", pd.DataFrame())
    if not total_breakdown.empty:
        display = total_breakdown.copy()
        display["Accuracy"] = display["Accuracy"].map(format_pct)
        lines.append("## Total pick breakdown")
        lines.append("")
        lines.append(render_simple_table(display[["ModelTotalLean", "Graded", "Correct", "Accuracy"]]))
        lines.append("")

    if not game_report.empty:
        game_cols = [
            "RoundLabel",
            "GameDateET",
            "Matchup",
            "MarketFavoriteTeamName",
            "PredWinnerTeamName",
            "ActualWinnerTeamName",
            "WinnerCorrect",
            "MarketSpreadTeamA",
            "ModelSpreadLeanTeamName",
            "ActualSpreadResultTeamName",
            "SpreadCorrect",
            "MarketTotal",
            "ModelTotalLean",
            "ActualTotalResult",
            "TotalCorrect",
            "ActualScore",
            "PredictedScore",
            "MarginAbsError",
            "TotalAbsError",
        ]
        display = game_report[game_cols].copy()
        display["WinnerCorrect"] = display["WinnerCorrect"].map(bool_to_flag)
        display["SpreadCorrect"] = display["SpreadCorrect"].map(bool_to_flag)
        display["TotalCorrect"] = display["TotalCorrect"].map(bool_to_flag)
        display["MarketSpreadTeamA"] = display["MarketSpreadTeamA"].map(lambda x: format_line(x, 1, True))
        display["MarketTotal"] = display["MarketTotal"].map(lambda x: format_num(x, 1))
        display["MarginAbsError"] = display["MarginAbsError"].map(lambda x: format_num(x, 1))
        display["TotalAbsError"] = display["TotalAbsError"].map(lambda x: format_num(x, 1))
        lines.append("## Game by game")
        lines.append("")
        lines.append(render_simple_table(display))
        lines.append("")
    return "\n".join(lines)


def build_current_market_markdown(
    summary: dict[str, Any],
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append(f"# {summary['season']} Current Tournament Market Report")
    lines.append("")
    lines.append(f"- Market source: {summary['market_source']}")
    lines.append(f"- Games scored: {summary['games_scored']}")
    lines.append(f"- Candidate sides: {summary['candidate_rows']}")
    lines.append(f"- Recommended bets: {summary['recommended_bets']}")
    lines.append("")
    if not candidates.empty:
        top = candidates.loc[candidates["IsRecommended"]].copy()
        top = top.sort_values(["EVPerUnit", "EdgeProb", "StakeAmount"], ascending=[False, False, False]).head(25)
        lines.append("## Top recommended bets")
        lines.append("")
        lines.append(
            render_simple_table(
                top[
                    [
                        "Round",
                        "GameDateET",
                        "TeamAName",
                        "TeamBName",
                        "Bookmaker",
                        "BetType",
                        "BetSide",
                        "Line",
                        "OddsAmerican",
                        "ModelProb",
                        "EVPerUnit",
                        "StakeAmount",
                    ]
                ]
            )
        )
        lines.append("")

    if not predictions.empty:
        display = predictions.copy()
        display = display.sort_values(["GameDateET", "CommenceTime", "Round", "TeamAName", "TeamBName"])
        lines.append("## Game-by-game model view")
        lines.append("")
        lines.append(
            render_simple_table(
                display[
                    [
                        "Round",
                        "GameDateET",
                        "TeamAName",
                        "TeamBName",
                        "WinProbTeamA",
                        "PredScoreTeamA",
                        "PredScoreTeamB",
                        "FairMoneylineTeamA",
                        "FairMoneylineTeamB",
                        "MarketMoneylineTeamA",
                        "MarketMoneylineTeamB",
                        "FairSpreadTeamA",
                        "MarketSpreadTeamA",
                        "FairTotal",
                        "MarketTotal",
                        "ModelProbCoverTeamA",
                        "ModelProbOver",
                    ]
                ]
            )
        )
        lines.append("")
    return "\n".join(lines)


def build_bracket_markdown(
    season: int,
    bracket: pd.DataFrame,
) -> str:
    lines = [f"# {season} Full Bracket Simulation", ""]
    if bracket.empty:
        lines.append("(no rows)")
        return "\n".join(lines)
    champion = bracket.sort_values("RoundNum").tail(1).iloc[0]
    lines.append(f"- Predicted champion: {champion['PredWinnerTeamName']}")
    lines.append("")
    lines.append("## Simulation rows")
    lines.append("")
    lines.append(
        render_simple_table(
            bracket[
                [
                    "RoundNum",
                    "RoundName",
                    "Slot",
                    "StrongSource",
                    "WeakSource",
                    "TeamAName",
                    "TeamBName",
                    "PredWinnerTeamName",
                    "PredMargin",
                    "PredTotal",
                ]
            ]
        )
    )
    lines.append("")
    return "\n".join(lines)


def prep_raw_market_rows(paths: AppPaths) -> pd.DataFrame:
    prep = pd.read_csv(paths.prep_dir / "mm26_cleaned_tournament_odds_subset_2026.csv")
    if "Season" not in prep.columns:
        prep["Season"] = 2026
    raw = prep[
        [
            "EventID",
            "CommenceTimeUTC",
            "OfficialTeamA",
            "OfficialTeamB",
            "MarketMoneylineTeamA",
            "MarketMoneylineTeamB",
            "MarketSpreadTeamA",
            "MarketSpreadPriceA",
            "MarketSpreadPriceB",
            "MarketTotal",
            "MarketOverPrice",
            "MarketUnderPrice",
            "Season",
            "TeamAID",
            "TeamBID",
            "MarketBookCount",
            "MoneylineHold",
            "SpreadHold",
            "TotalHold",
        ]
    ].rename(
        columns={
            "CommenceTimeUTC": "CommenceTime",
            "OfficialTeamA": "TeamA",
            "OfficialTeamB": "TeamB",
        }
    )
    raw["Bookmaker"] = DEFAULT_BOOKMAKER_LABEL
    raw["BookTitle"] = "Bundled Prep Lines"
    raw["BookLastUpdate"] = pd.NaT
    return raw[
        [
            "EventID",
            "CommenceTime",
            "Bookmaker",
            "BookTitle",
            "BookLastUpdate",
            "TeamA",
            "TeamB",
            "MarketMoneylineTeamA",
            "MarketMoneylineTeamB",
            "MarketSpreadTeamA",
            "MarketSpreadPriceA",
            "MarketSpreadPriceB",
            "MarketTotal",
            "MarketOverPrice",
            "MarketUnderPrice",
            "Season",
            "TeamAID",
            "TeamBID",
            "MarketBookCount",
            "MoneylineHold",
            "SpreadHold",
            "TotalHold",
        ]
    ].copy()


def official_tournament_metadata(paths: AppPaths) -> pd.DataFrame:
    annotated = pd.read_csv(paths.prep_dir / "mm26_first_four_and_round1_annotated_schedule.csv")
    meta = annotated[
        [
            "Round",
            "GameDateET",
            "TipTimeET",
            "Network",
            "Region",
            "TeamAID",
            "TeamBID",
        ]
    ].copy()
    meta = meta.loc[meta["TeamAID"].notna() & meta["TeamBID"].notna()].drop_duplicates()
    meta["TeamAID"] = meta["TeamAID"].astype(int)
    meta["TeamBID"] = meta["TeamBID"].astype(int)
    meta["PairKey"] = meta.apply(lambda r: pair_key(r["TeamAID"], r["TeamBID"]), axis=1)
    return meta


def prepare_current_market_inputs(
    paths: AppPaths,
    settings: dict[str, Any],
    market_source: str,
    api_key: str | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    live_cache_ttl_seconds: int | None = None,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    bootstrap(paths)
    data = load_kaggle_data(paths.kaggle_data)
    aliases = build_alias_lookup(data)
    meta = official_tournament_metadata(paths)
    pair_whitelist = set(meta["PairKey"].tolist())

    prep_raw = prep_raw_market_rows(paths).copy()
    prep_raw["PairKey"] = prep_raw.apply(lambda r: pair_key(r["TeamAID"], r["TeamBID"]), axis=1)
    prep_raw = prep_raw.loc[prep_raw["PairKey"].isin(pair_whitelist)].copy()

    if market_source == "prep":
        return prep_raw.drop(columns=["PairKey"]), "prep_bundle", {"live_rows": 0, "prep_rows": int(len(prep_raw))}

    live_rows = pd.DataFrame()
    source_label = market_source
    fetch_meta: dict[str, Any] = {"live_rows": 0, "prep_rows": int(len(prep_raw))}
    if market_source in {"live", "hybrid"}:
        resolved_api_key = read_api_key(settings, api_key)
        flat_live, live_meta = fetch_live_odds(
            api_key=resolved_api_key,
            sport=str(settings.get("sport", "basketball_ncaab")),
            regions=str(settings.get("regions", "us")),
            markets=str(settings.get("markets", "h2h,spreads,totals")),
            cache_dir=paths.api_cache,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            max_age_seconds=live_cache_ttl_seconds,
            timeout=int(settings.get("request_timeout_seconds", 60)),
            max_retries=int(settings.get("request_max_retries", 4)),
            retry_backoff_seconds=float(settings.get("request_retry_backoff_seconds", 3.0)),
        )
        resolved_live, unresolved = safe_resolve_market_team_ids(flat_live, aliases, season=2026)
        resolved_live = resolved_live.loc[resolved_live["TeamAID"].notna() & resolved_live["TeamBID"].notna()].copy()
        resolved_live["TeamAID"] = resolved_live["TeamAID"].astype(int)
        resolved_live["TeamBID"] = resolved_live["TeamBID"].astype(int)
        resolved_live["PairKey"] = resolved_live.apply(lambda r: pair_key(r["TeamAID"], r["TeamBID"]), axis=1)
        live_rows = resolved_live.loc[resolved_live["PairKey"].isin(pair_whitelist)].copy()
        fetch_meta = {
            **live_meta,
            "live_rows": int(len(live_rows)),
            "prep_rows": int(len(prep_raw)),
            "unresolved_names": unresolved,
        }

    if market_source == "live":
        if live_rows.empty:
            raise RuntimeError("No live tournament rows matched the official current bracket schedule.")
        return live_rows.drop(columns=["PairKey"]), "live", fetch_meta

    # hybrid
    live_pairs = set(live_rows["PairKey"].tolist()) if not live_rows.empty else set()
    prep_missing = prep_raw.loc[~prep_raw["PairKey"].isin(live_pairs)].copy()
    combined = pd.concat([live_rows, prep_missing], ignore_index=True)
    source_bits = []
    if not live_rows.empty:
        source_bits.append("live")
    if not prep_missing.empty:
        source_bits.append("prep_fallback")
    return combined.drop(columns=["PairKey"]), "+".join(source_bits) if source_bits else "prep_bundle", fetch_meta


def enrich_current_predictions_with_metadata(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    paths: AppPaths,
    team_names: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = official_tournament_metadata(paths)[["Round", "GameDateET", "TipTimeET", "Network", "Region", "PairKey"]].copy()
    pred = predictions.copy()
    pred["PairKey"] = pred.apply(lambda r: pair_key(r["TeamAID"], r["TeamBID"]), axis=1)
    pred = pred.merge(meta, on="PairKey", how="left")
    pred["TeamAName"] = pred["TeamAID"].map(team_names)
    pred["TeamBName"] = pred["TeamBID"].map(team_names)
    pred["PredWinnerTeamName"] = pred["PredWinnerTeamID"].map(team_names)

    cand = candidates.copy()
    if not cand.empty:
        cand["PairKey"] = cand.apply(lambda r: pair_key(r["TeamAID"], r["TeamBID"]), axis=1)
        cand = cand.merge(meta, on="PairKey", how="left")
        cand["TeamAName"] = cand["TeamAID"].map(team_names).fillna(cand["TeamAName"])
        cand["TeamBName"] = cand["TeamBID"].map(team_names).fillna(cand["TeamBName"])
    return pred.drop(columns=["PairKey"]), cand.drop(columns=["PairKey"], errors="ignore")


def invoke_action(label: str, func, *func_args) -> int:
    try:
        return int(func(*func_args))
    except KeyboardInterrupt:
        log(f"{label}: cancelled.")
        return 1
    except Exception as exc:
        log(f"{label}: {exc}")
        return 1


def prompt_choice(label: str, options: list[str], default: str) -> str:
    normalized = {option.lower(): option for option in options}
    prompt = "/".join(options)
    while True:
        value = input(f"{label} ({prompt}) [{default}]: " ).strip()
        if not value:
            return default
        lowered = value.lower()
        if lowered in normalized:
            return normalized[lowered]
        log(f"Invalid choice: {value}")


def prompt_int(label: str, default: int, minimum: int | None = None) -> int:
    while True:
        raw = input(f"{label} [{default}]: " ).strip()
        if not raw:
            value = int(default)
        else:
            try:
                value = int(raw)
            except ValueError:
                log(f"Invalid integer: {raw}")
                continue
        if minimum is not None and value < minimum:
            log(f"Value must be >= {minimum}")
            continue
        return value


def prompt_float(label: str, default: float, minimum: float | None = None) -> float:
    while True:
        raw = input(f"{label} [{default}]: " ).strip()
        if not raw:
            value = float(default)
        else:
            try:
                value = float(raw)
            except ValueError:
                log(f"Invalid number: {raw}")
                continue
        if minimum is not None and value < minimum:
            log(f"Value must be >= {minimum}")
            continue
        return value



def normalize_optimize_target(value: Any) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "ou": "total",
        "o/u": "total",
        "over_under": "total",
        "overunder": "total",
        "over/under": "total",
        "totals": "total",
        "moneyline": "winner",
        "ml": "winner",
        "win": "winner",
        "winner": "winner",
        "spreads": "spread",
    }
    norm = aliases.get(raw, raw)
    if norm not in {"winner", "spread", "total"}:
        norm = "spread"
    return norm


def prompt_optional_text(label: str, default: str | None = None) -> str | None:
    display_default = "" if default is None else default
    value = input(f"{label} [{display_default}]: " ).strip()
    if value:
        return value
    return default if display_default else None


def command_bootstrap(args: argparse.Namespace, paths: AppPaths, settings: dict[str, Any]) -> int:
    bootstrap(paths, force=bool(args.force))
    log(f"Workspace ready at: {paths.workspace}")
    return 0


def command_cache_info(args: argparse.Namespace, paths: AppPaths, settings: dict[str, Any]) -> int:
    stats = {
        "api_cache": describe_api_cache(paths),
        "season_state_cache": describe_season_state_cache(paths),
    }
    log(json.dumps(stats, indent=2))
    return 0


def command_clear_cache(args: argparse.Namespace, paths: AppPaths, settings: dict[str, Any]) -> int:
    result = clear_api_cache(paths, include_season_state=bool(getattr(args, "include_season_state", False)))
    log(json.dumps(result, indent=2))
    return 0


def command_pull_odds(args: argparse.Namespace, paths: AppPaths, settings: dict[str, Any]) -> int:
    bootstrap(paths)
    api_key = read_api_key(settings, args.api_key)
    year = int(args.year)
    scope = str(args.scope)
    optimize_for = normalize_optimize_target(getattr(args, "optimize_for", None) or settings.get("default_optimize_for", "spread"))
    interval_hours = int(args.interval_hours)
    sport = str(args.sport or settings.get("sport", "basketball_ncaab"))
    regions = str(args.regions or settings.get("regions", "us"))
    markets = str(args.markets or settings.get("markets", "h2h,spreads,totals"))
    output_dir = Path(args.output_dir) if args.output_dir else (paths.outputs / "odds_history" / f"{year}_{scope}")
    ensure_dir(output_dir)
    use_cache = not bool(getattr(args, "no_cache", False))
    refresh_cache = bool(getattr(args, "refresh_cache", False))
    timeout_seconds = int(getattr(args, "timeout_seconds", None) or settings.get("request_timeout_seconds", 60))
    max_retries = int(getattr(args, "max_retries", None) or settings.get("request_max_retries", 4))
    retry_backoff_seconds = float(getattr(args, "retry_backoff_seconds", None) or settings.get("request_retry_backoff_seconds", 3.0))
    fail_fast = bool(getattr(args, "fail_fast", False))

    start_dt, end_dt = date_range_from_year(paths, year, scope, args.start_date, args.end_date)
    snapshots = iter_snapshots(start_dt, end_dt, interval_hours)
    estimated_credits = estimate_historical_credit_cost(markets, regions, len(snapshots))
    log(
        f"Historical pull window: {start_dt.isoformat()} -> {end_dt.isoformat()} | "
        f"snapshots={len(snapshots)} | rough max credits={estimated_credits}"
    )

    raw_path = output_dir / "raw_snapshots.csv"
    metadata_path = output_dir / "fetch_metadata.json"
    closing_path = output_dir / "closing_book_rows.csv"
    consensus_path = output_dir / "closing_consensus.csv"

    existing = pd.read_csv(raw_path, low_memory=False) if raw_path.exists() and not args.force else pd.DataFrame()
    done_requests = set(existing["SnapshotRequestedUTC"].dropna().astype(str).tolist()) if not existing.empty and "SnapshotRequestedUTC" in existing.columns else set()

    data = load_kaggle_data(paths.kaggle_data)
    aliases = build_alias_lookup(data)
    all_parts: list[pd.DataFrame] = []
    fetch_records: list[dict[str, Any]] = []

    if not existing.empty:
        all_parts.append(existing)
        if metadata_path.exists():
            try:
                fetch_records.extend(json.loads(metadata_path.read_text(encoding="utf-8")).get("requests", []))
            except Exception:
                pass

    failures = 0
    for idx, point in enumerate(snapshots, start=1):
        requested = point.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        if requested in done_requests:
            continue
        log(f"[{idx}/{len(snapshots)}] Resolving historical snapshot {requested}")
        try:
            flat, meta = fetch_historical_snapshot(
                api_key=api_key,
                sport=sport,
                regions=regions,
                markets=markets,
                snapshot_date_utc=requested,
                bookmakers=args.bookmakers,
                commence_time_from=args.commence_time_from,
                commence_time_to=args.commence_time_to,
                cache_dir=paths.api_cache,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
                timeout=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
        except Exception as exc:
            failures += 1
            error_meta = {
                "requested_snapshot_utc": requested,
                "rows": 0,
                "error": str(exc),
                "failed": True,
            }
            fetch_records.append(error_meta)
            metadata = {
                "year": year,
                "scope": scope,
                "sport": sport,
                "regions": regions,
                "markets": markets,
                "interval_hours": interval_hours,
                "start_utc": start_dt.isoformat(),
                "end_utc": end_dt.isoformat(),
                "cache_enabled": use_cache,
                "refresh_cache": refresh_cache,
                "timeout_seconds": timeout_seconds,
                "max_retries": max_retries,
                "retry_backoff_seconds": retry_backoff_seconds,
                "failures": failures,
                "requests": fetch_records,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            log(f"    -> failed after retries: {exc}")
            if fail_fast:
                raise
            continue

        resolved, unresolved = safe_resolve_market_team_ids(flat, aliases, season=year)
        resolved["SnapshotRequestedUTC"] = meta["requested_snapshot_utc"]
        resolved["SnapshotTimeUTC"] = meta["snapshot_time_utc"]
        resolved["SnapshotPreviousUTC"] = meta["previous_snapshot_utc"]
        resolved["SnapshotNextUTC"] = meta["next_snapshot_utc"]
        resolved["QuotaRemaining"] = meta["quota_remaining"]
        resolved["QuotaUsed"] = meta["quota_used"]
        resolved["QuotaLast"] = meta["quota_last"]
        resolved["CacheHit"] = bool(meta.get("cache_hit", False))
        resolved["CachePath"] = meta.get("cache_path")
        resolved["RequestAttempts"] = int(meta.get("request_attempts", 0) or 0)
        all_parts.append(resolved)
        source_label = "cache" if meta.get("cache_hit") else "api"
        log(
            f"    -> {source_label}: rows={meta.get('rows', 0)} quota_remaining={meta.get('quota_remaining')} attempts={meta.get('request_attempts', 0)}"
        )
        fetch_records.append({**meta, "unresolved_names": unresolved, "failed": False})

        combined = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
        combined.to_csv(raw_path, index=False)
        metadata = {
            "year": year,
            "scope": scope,
            "sport": sport,
            "regions": regions,
            "markets": markets,
            "interval_hours": interval_hours,
            "start_utc": start_dt.isoformat(),
            "end_utc": end_dt.isoformat(),
            "cache_enabled": use_cache,
            "refresh_cache": refresh_cache,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "retry_backoff_seconds": retry_backoff_seconds,
            "failures": failures,
            "requests": fetch_records,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    combined = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    if not combined.empty:
        unresolved_rows_path = output_dir / "unresolved_market_rows.csv"
        unresolved_count = write_unresolved_market_rows(combined, unresolved_rows_path)
        if unresolved_count:
            log(f"Detected {unresolved_count} unresolved market row(s). They will be excluded from consensus matching. See {unresolved_rows_path}")
        closing = select_latest_pregame_snapshot(
            combined,
            pregame_buffer_minutes=int(settings.get("historical_pregame_buffer_minutes", 5)),
        )
        closing.to_csv(closing_path, index=False)
        consensus = build_consensus_matchups(closing)
        if not consensus.empty:
            consensus.to_csv(consensus_path, index=False)
    else:
        closing = pd.DataFrame()
        consensus = pd.DataFrame()

    log(f"Saved raw snapshot history to: {raw_path}")
    log(f"Saved closing per-book rows to: {closing_path}")
    log(f"Saved closing consensus to: {consensus_path}")
    if failures:
        log(f"Historical pull completed with {failures} failed snapshot request(s). Re-run later to fill gaps; cache/raw files preserve completed work.")
    return 0


def _legacy_command_backtest(args: argparse.Namespace, paths: AppPaths, settings: dict[str, Any]) -> int:
    bootstrap(paths)
    year = int(args.year)
    scope = str(args.scope)
    output_dir = Path(args.output_dir) if args.output_dir else (paths.outputs / "backtests" / f"{year}_{scope}")
    ensure_dir(output_dir)

    raw_odds_path = Path(args.raw_odds_csv) if args.raw_odds_csv else (paths.outputs / "odds_history" / f"{year}_{scope}" / "raw_snapshots.csv")
    if not raw_odds_path.exists():
        if not args.auto_fetch:
            raise FileNotFoundError(
                f"Historical raw odds file not found: {raw_odds_path}. Re-run with --auto-fetch or pass --raw-odds-csv."
            )
        pull_args = argparse.Namespace(
            year=year,
            scope=scope,
            interval_hours=args.interval_hours or int(settings.get("default_historical_interval_hours", 6)),
            sport=args.sport or settings.get("sport", "basketball_ncaab"),
            regions=args.regions or settings.get("regions", "us"),
            markets=args.markets or settings.get("markets", "h2h,spreads,totals"),
            output_dir=str(paths.outputs / "odds_history" / f"{year}_{scope}"),
            force=False,
            api_key=args.api_key,
            bookmakers=args.bookmakers,
            start_date=args.start_date,
            end_date=args.end_date,
            commence_time_from=None,
            commence_time_to=None,
            no_cache=getattr(args, "no_cache", False),
            refresh_cache=getattr(args, "refresh_cache", False),
            timeout_seconds=getattr(args, "timeout_seconds", None),
            max_retries=getattr(args, "max_retries", None),
            retry_backoff_seconds=getattr(args, "retry_backoff_seconds", None),
            fail_fast=getattr(args, "fail_fast", False),
        )
        command_pull_odds(pull_args, paths, settings)

    raw_snapshots = pd.read_csv(raw_odds_path, low_memory=False)
    if raw_snapshots.empty:
        raise RuntimeError(f"No rows in historical odds file: {raw_odds_path}")

    data = load_kaggle_data(paths.kaggle_data)
    aliases = build_alias_lookup(data)
    raw_snapshots, unresolved_names = safe_resolve_market_team_ids(raw_snapshots, aliases, season=year)

    # Repair stale raw files from earlier runs now that the resolver can handle bookmaker
    # names like "Duke Blue Devils" or "Houston Cougars".
    raw_snapshots.to_csv(raw_odds_path, index=False)

    unresolved_rows_path = output_dir / "unresolved_market_rows.csv"
    unresolved_count = write_unresolved_market_rows(raw_snapshots, unresolved_rows_path)
    if unresolved_count:
        preview = ", ".join(unresolved_names[:10])
        more = "" if len(unresolved_names) <= 10 else f" (+{len(unresolved_names) - 10} more)"
        log(
            f"Found {unresolved_count} unresolved historical market row(s). "
            f"They will be excluded from consensus matching. See {unresolved_rows_path}. "
            f"Unresolved names: {preview}{more}"
        )

    artifact_dir = train_model_if_needed(paths, target_season=year, force_retrain=bool(args.force_retrain))

    closing_raw = select_latest_pregame_snapshot(
        raw_snapshots,
        pregame_buffer_minutes=int(settings.get("historical_pregame_buffer_minutes", 5)),
    )
    consensus = build_consensus_matchups(closing_raw)
    actual_games = build_actual_games(paths, year, scope)
    matched_consensus, unmatched = match_market_events_to_actuals(consensus, actual_games)
    matched_event_ids = set(matched_consensus["EventID"].astype(str).tolist()) if not matched_consensus.empty else set()
    matched_raw = closing_raw.loc[closing_raw["EventID"].astype(str).isin(matched_event_ids)].copy()

    data = load_kaggle_data(paths.kaggle_data)
    names = team_name_map(data.teams)
    predictions = prepare_backtest_predictions(matched_consensus, artifact_dir)
    game_report = compute_backtest_game_report(matched_consensus, predictions, names)
    round_summary = build_backtest_round_summary(game_report)
    side_breakdown = build_backtest_side_breakdown(game_report)
    summary = build_backtest_summary(
        year=year,
        scope=scope,
        artifact_dir=artifact_dir,
        raw_odds_path=raw_odds_path,
        consensus=consensus,
        matched_consensus=matched_consensus,
        unmatched=unmatched,
        game_report=game_report,
    )

    summary_path = output_dir / "backtest_summary.json"
    matched_consensus.to_csv(output_dir / "matched_market_consensus.csv", index=False)
    matched_raw.to_csv(output_dir / "matched_market_book_rows.csv", index=False)
    predictions.to_csv(output_dir / "backtest_predictions.csv", index=False)
    game_report.to_csv(output_dir / "backtest_game_report.csv", index=False)
    round_summary.to_csv(output_dir / "backtest_round_summary.csv", index=False)
    for key, frame in side_breakdown.items():
        frame.to_csv(output_dir / f"backtest_{key}_breakdown.csv", index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown = build_backtest_markdown(summary, game_report, round_summary, side_breakdown)
    write_text(output_dir / "backtest_report.md", markdown)

    log(json.dumps(summary, indent=2))
    log(f"Saved backtest outputs to: {output_dir}")
    return 0


def _legacy_command_predict_current(args: argparse.Namespace, paths: AppPaths, settings: dict[str, Any]) -> int:
    bootstrap(paths)
    season = int(args.season)
    artifact_dir = train_model_if_needed(paths, target_season=season, force_retrain=bool(args.force_retrain))
    market_source = str(args.market_source or settings.get("default_market_source", "hybrid"))
    use_cache = not bool(getattr(args, "no_cache", False))
    refresh_cache = bool(getattr(args, "refresh_cache", False))
    raw_live_cache_ttl = getattr(args, "live_cache_ttl_minutes", None)
    if raw_live_cache_ttl is None:
        raw_live_cache_ttl = settings.get("live_cache_ttl_minutes", 10)
    live_cache_ttl_minutes = int(raw_live_cache_ttl)
    raw_market, source_label, fetch_meta = prepare_current_market_inputs(
        paths=paths,
        settings=settings,
        market_source=market_source,
        api_key=args.api_key,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        live_cache_ttl_seconds=max(0, live_cache_ttl_minutes) * 60,
    )
    if raw_market.empty:
        raise RuntimeError("No current tournament market rows found.")

    consensus = build_consensus_matchups(raw_market)
    bundle = load_bundle(artifact_dir)
    team_features = load_team_features_from_artifacts(artifact_dir)
    predictions = predict_matchups(consensus, team_features, bundle)
    candidates = price_bookmaker_sides(
        flat_odds_resolved=raw_market,
        event_predictions=predictions,
        bankroll=float(args.bankroll or settings.get("bankroll", 5000.0)),
        fractional_kelly=float(args.fractional_kelly or settings.get("fractional_kelly", 0.25)),
        max_stake_fraction=float(args.max_stake_fraction or settings.get("max_stake_fraction", 0.02)),
        min_moneyline_ev=float(args.min_moneyline_ev),
        min_spread_ev=float(args.min_spread_ev),
        min_total_ev=float(args.min_total_ev),
        min_edge_prob=float(args.min_edge_prob),
        min_market_books=int(args.min_market_books),
    )

    data = load_kaggle_data(paths.kaggle_data)
    names = team_name_map(data.teams)
    predictions, candidates = enrich_current_predictions_with_metadata(predictions, candidates, paths, names)

    output_dir = Path(args.output_dir) if args.output_dir else (paths.outputs / "current_market" / str(season))
    ensure_dir(output_dir)
    summary = {
        "season": season,
        "artifact_dir": str(artifact_dir),
        "market_source": source_label,
        "games_scored": int(len(predictions)),
        "candidate_rows": int(len(candidates)),
        "recommended_bets": int(candidates["IsRecommended"].sum()) if not candidates.empty else 0,
        "cache_enabled": use_cache,
        "refresh_cache": refresh_cache,
        "live_cache_ttl_minutes": live_cache_ttl_minutes,
        "fetch_meta": fetch_meta,
    }
    (output_dir / "market_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    raw_market.to_csv(output_dir / "raw_market_rows.csv", index=False)
    consensus.to_csv(output_dir / "market_consensus.csv", index=False)
    predictions.to_csv(output_dir / "market_predictions.csv", index=False)
    candidates.to_csv(output_dir / "market_candidate_sides.csv", index=False)
    markdown = build_current_market_markdown(summary, predictions, candidates)
    write_text(output_dir / "market_report.md", markdown)

    log(json.dumps(summary, indent=2))
    log(f"Saved current market outputs to: {output_dir}")
    return 0


def _legacy_command_predict_bracket(args: argparse.Namespace, paths: AppPaths, settings: dict[str, Any]) -> int:
    bootstrap(paths)
    season = int(args.season)
    artifact_dir = train_model_if_needed(paths, target_season=season, force_retrain=bool(args.force_retrain))
    data = load_kaggle_data(paths.kaggle_data)
    team_features = load_team_features_from_artifacts(artifact_dir)
    bundle = load_bundle(artifact_dir)
    bracket = simulate_bracket(season, data.seeds, data.slots, team_features, bundle)
    names = team_name_map(data.teams)
    bracket = bracket.copy()
    bracket["TeamAName"] = bracket["TeamAID"].map(names)
    bracket["TeamBName"] = bracket["TeamBID"].map(names)
    bracket["PredWinnerTeamName"] = bracket["PredWinnerTeamID"].map(names)

    output_dir = Path(args.output_dir) if args.output_dir else (paths.outputs / "brackets" / str(season))
    ensure_dir(output_dir)
    bracket.to_csv(output_dir / f"predicted_bracket_{season}.csv", index=False)
    write_text(output_dir / f"predicted_bracket_{season}.md", build_bracket_markdown(season, bracket))
    log(f"Saved bracket simulation to: {output_dir}")
    return 0


def command_smoke_test(args: argparse.Namespace, paths: AppPaths, settings: dict[str, Any]) -> int:
    bootstrap(paths)
    market_args = argparse.Namespace(
        season=2026,
        force_retrain=False,
        market_source="prep",
        api_key=None,
        bankroll=settings.get("bankroll", 5000.0),
        fractional_kelly=settings.get("fractional_kelly", 0.25),
        max_stake_fraction=settings.get("max_stake_fraction", 0.02),
        min_moneyline_ev=0.015,
        min_spread_ev=0.010,
        min_total_ev=0.010,
        min_edge_prob=0.015,
        min_market_books=1,
        optimize_for=normalize_optimize_target(settings.get("default_optimize_for", "spread")),
        output_dir=str(paths.outputs / "smoke_test" / "current_market"),
        no_cache=True,
        refresh_cache=False,
        live_cache_ttl_minutes=0,
    )
    command_predict_current(market_args, paths, settings)

    bracket_args = argparse.Namespace(
        season=2026,
        force_retrain=False,
        optimize_for=normalize_optimize_target(settings.get("default_optimize_for", "spread")),
        output_dir=str(paths.outputs / "smoke_test" / "bracket"),
    )
    command_predict_bracket(bracket_args, paths, settings)
    log(f"Smoke test outputs written to: {paths.outputs / 'smoke_test'}")
    return 0


def prompt_text(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def prompt_yes_no(label: str, default: bool = True) -> bool:
    default_label = "Y/n" if default else "y/N"
    value = input(f"{label} [{default_label}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def interactive_menu(paths: AppPaths, settings: dict[str, Any]) -> int:
    bootstrap(paths)

    while True:
        api_stats = describe_api_cache(paths)
        season_state_stats = describe_season_state_cache(paths)
        menu = textwrap.dedent(
            f"""
            March Madness market console

            Workspace: {paths.workspace}
            API cache: {api_stats['entries']} entries | {api_stats['size_mb']} MB | {paths.api_cache}
            Season-state cache: {season_state_stats['entries']} files | {season_state_stats['size_mb']} MB | {paths.cache / 'season_state'}

            1) Pull historical Odds API data for a year / scope
            2) Backtest selected tournament with selected-season + prior-season training
            3) Score the current tournament market with selected-season + prior-season training
            4) Simulate the full bracket from the current snapshot model
            5) Run offline smoke test
            6) Show cache stats
            7) Clear cache
            0) Exit
            """
        ).strip()

        print()
        print(menu)
        choice = input("Select an option or command: " ).strip().lower()
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        if choice in {"1", "pull", "pull-odds"}:
            year = prompt_int("Year", 2025, minimum=2021)
            scope = prompt_choice("Scope", ["season", "tournament"], "tournament")
            interval = prompt_int("Historical snapshot interval in hours", int(settings.get("default_historical_interval_hours", 6)), minimum=1)
            bookmakers = prompt_optional_text("Bookmakers override (blank for default/all books)", None)
            use_cache = prompt_yes_no("Use persistent API cache?", True)
            refresh_cache = prompt_yes_no("Force refresh even if cached?", False) if use_cache else False
            args = argparse.Namespace(
                year=year,
                scope=scope,
                interval_hours=interval,
                sport=settings.get("sport", "basketball_ncaab"),
                regions=settings.get("regions", "us"),
                markets=settings.get("markets", "h2h,spreads,totals"),
                output_dir=None,
                force=False,
                api_key=None,
                bookmakers=bookmakers,
                start_date=None,
                end_date=None,
                commence_time_from=None,
                commence_time_to=None,
                no_cache=not use_cache,
                refresh_cache=refresh_cache,
            )
            invoke_action("pull-odds", command_pull_odds, args, paths, settings)
        elif choice in {"2", "backtest"}:
            year = prompt_int("Backtest year", 2025, minimum=2021)
            scope = "tournament"
            optimize_for = prompt_choice("Optimize for", ["spread", "total", "winner"], normalize_optimize_target(settings.get("default_optimize_for", "spread")))
            auto_fetch = prompt_yes_no("Auto-fetch historical odds if missing?", True)
            use_cache = prompt_yes_no("Use persistent API cache for auto-fetch?", True) if auto_fetch else True
            refresh_cache = prompt_yes_no("Force refresh cached historical API responses?", False) if auto_fetch and use_cache else False
            interval_hours = prompt_int("Historical snapshot interval in hours", int(settings.get("default_historical_interval_hours", 6)), minimum=1) if auto_fetch else int(settings.get("default_historical_interval_hours", 6))
            args = argparse.Namespace(
                year=year,
                scope=scope,
                raw_odds_csv=None,
                auto_fetch=auto_fetch,
                interval_hours=interval_hours,
                sport=settings.get("sport", "basketball_ncaab"),
                regions=settings.get("regions", "us"),
                markets=settings.get("markets", "h2h,spreads,totals"),
                api_key=None,
                bookmakers=None,
                start_date=None,
                end_date=None,
                bankroll=None,
                fractional_kelly=None,
                max_stake_fraction=None,
                min_moneyline_ev=0.015,
                min_spread_ev=0.010,
                min_total_ev=0.010,
                min_edge_prob=0.015,
                min_market_books=1,
                force_retrain=False,
                optimize_for=optimize_for,
                output_dir=None,
                no_cache=not use_cache,
                refresh_cache=refresh_cache,
                timeout_seconds=int(settings.get("request_timeout_seconds", 60)),
                max_retries=int(settings.get("request_max_retries", 4)),
                retry_backoff_seconds=float(settings.get("request_retry_backoff_seconds", 3.0)),
                fail_fast=False,
            )
            invoke_action("backtest", command_backtest, args, paths, settings)
        elif choice in {"3", "current", "predict-current"}:
            season = prompt_int("Season", 2026, minimum=2021)
            market_source = prompt_choice("Market source", ["hybrid", "live", "prep"], str(settings.get("default_market_source", "hybrid")))
            optimize_for = prompt_choice("Optimize for", ["spread", "total", "winner"], normalize_optimize_target(settings.get("default_optimize_for", "spread")))
            use_cache = prompt_yes_no("Use persistent API cache for live queries?", True) if market_source != "prep" else True
            refresh_cache = prompt_yes_no("Force refresh cached live responses?", False) if market_source != "prep" and use_cache else False
            live_cache_ttl = prompt_int("Live odds cache TTL in minutes", int(settings.get("live_cache_ttl_minutes", 10)), minimum=0) if market_source != "prep" else int(settings.get("live_cache_ttl_minutes", 10))
            args = argparse.Namespace(
                season=season,
                force_retrain=False,
                optimize_for=optimize_for,
                market_source=market_source,
                api_key=None,
                bankroll=None,
                fractional_kelly=None,
                max_stake_fraction=None,
                min_moneyline_ev=0.015,
                min_spread_ev=0.010,
                min_total_ev=0.010,
                min_edge_prob=0.015,
                min_market_books=1,
                output_dir=None,
                no_cache=not use_cache,
                refresh_cache=refresh_cache,
                live_cache_ttl_minutes=live_cache_ttl,
            )
            invoke_action("predict-current", command_predict_current, args, paths, settings)
        elif choice in {"4", "bracket", "predict-bracket"}:
            season = prompt_int("Season", 2026, minimum=2021)
            optimize_for = prompt_choice("Optimize for", ["spread", "total", "winner"], normalize_optimize_target(settings.get("default_optimize_for", "spread")))
            args = argparse.Namespace(
                season=season,
                force_retrain=False,
                optimize_for=optimize_for,
                output_dir=None,
            )
            invoke_action("predict-bracket", command_predict_bracket, args, paths, settings)
        elif choice in {"5", "smoke", "smoke-test"}:
            invoke_action("smoke-test", command_smoke_test, argparse.Namespace(), paths, settings)
        elif choice in {"6", "cache", "cache-info"}:
            invoke_action("cache-info", command_cache_info, argparse.Namespace(), paths, settings)
        elif choice in {"7", "clear", "clear-cache"}:
            if prompt_yes_no("Delete the persistent Odds API cache?", False):
                include_season_state = prompt_yes_no("Also delete season-state cache (observed market/scores)?", False)
                invoke_action("clear-cache", command_clear_cache, argparse.Namespace(include_season_state=include_season_state), paths, settings)
        else:
            log("Invalid choice.")


SNAPSHOT_MODEL_VERSION = "season-snapshot-v3-multiseason-direct-market"


def discover_team_aliases_path() -> str | None:
    for candidate in [
        PROJECT_ROOT / "assets" / "team_aliases.csv",
        PROJECT_ROOT / "config" / "team_aliases.csv",
        PROJECT_ROOT / "workspace" / "team_aliases.csv",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def build_prediction_input_frame(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    work = events.copy()
    defaults = {
        "MarketProbTeamA": np.nan,
        "MarketProbTeamB": np.nan,
        "MarketMoneylineTeamA": np.nan,
        "MarketMoneylineTeamB": np.nan,
        "MarketSpreadTeamA": np.nan,
        "MarketSpreadPriceA": np.nan,
        "MarketSpreadPriceB": np.nan,
        "MarketTotal": np.nan,
        "MarketOverPrice": np.nan,
        "MarketUnderPrice": np.nan,
        "MarketBookCount": 0.0,
        "MoneylineHold": np.nan,
        "SpreadHold": np.nan,
        "TotalHold": np.nan,
        "EventID": "",
        "CommenceTime": "",
    }
    for col, default in defaults.items():
        if col not in work.columns:
            work[col] = default

    cols = [
        "Season",
        "TeamAID",
        "TeamBID",
        "EventID",
        "CommenceTime",
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
    ]
    return work[cols].copy()


def predict_market_event_frame(
    events: pd.DataFrame,
    team_features: pd.DataFrame,
    bundle: Any,
) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    return predict_matchups(build_prediction_input_frame(events), team_features, bundle)


def dataframe_signature(df: pd.DataFrame, sort_cols: list[str] | None = None) -> str:
    if df is None or df.empty:
        return "empty"
    work = df.copy()
    if sort_cols:
        existing = [col for col in sort_cols if col in work.columns]
        if existing:
            work = work.sort_values(existing).reset_index(drop=True)
    blob = work.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def season_state_dir(paths: AppPaths) -> Path:
    return ensure_dir(paths.cache / "season_state")


def observed_market_cache_path(paths: AppPaths, season: int) -> Path:
    return season_state_dir(paths) / f"{int(season)}_market_rows.csv"


def observed_scores_cache_path(paths: AppPaths, season: int) -> Path:
    return season_state_dir(paths) / f"{int(season)}_scores.csv"


def season_snapshot_model_root(paths: AppPaths, season: int) -> Path:
    return ensure_dir(paths.artifacts / "season_snapshot_models" / str(int(season)))


def read_cached_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def get_season_day_zero(paths: AppPaths, season: int) -> pd.Timestamp:
    seasons = pd.read_csv(paths.kaggle_data / "MSeasons.csv")
    row = seasons.loc[seasons["Season"] == int(season)]
    if row.empty:
        raise ValueError(f"Season {season} not present in MSeasons.csv")
    return pd.to_datetime(row.iloc[0]["DayZero"]).normalize()


def commence_series_to_daynum(paths: AppPaths, season: int, commence_series: pd.Series) -> pd.Series:
    day_zero = get_season_day_zero(paths, season)
    commence = pd.to_datetime(commence_series, utc=True, errors="coerce")
    event_days = commence.dt.tz_convert(ET).dt.tz_localize(None).dt.normalize()
    return (event_days - day_zero).dt.days


def tournament_field_team_ids(data, season: int) -> set[int]:
    seeds = data.seeds.loc[data.seeds["Season"] == int(season), "TeamID"]
    return set(pd.to_numeric(seeds, errors="coerce").dropna().astype(int).tolist())


def filter_market_rows_to_tournament_field(
    rows: pd.DataFrame,
    data,
    season: int,
    paths: AppPaths,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    team_ids = tournament_field_team_ids(data, season)
    if not team_ids:
        return rows.copy()

    start_utc, end_utc = date_range_from_year(paths, int(season), "tournament")
    start_bound = pd.to_datetime(start_utc, utc=True, errors="coerce") - pd.Timedelta(days=2)
    end_bound = pd.to_datetime(end_utc, utc=True, errors="coerce") + pd.Timedelta(days=2)

    work = rows.copy()
    work["TeamAID"] = pd.to_numeric(work.get("TeamAID"), errors="coerce")
    work["TeamBID"] = pd.to_numeric(work.get("TeamBID"), errors="coerce")
    work["CommenceTimeParsed"] = pd.to_datetime(work.get("CommenceTime"), utc=True, errors="coerce")

    mask = (
        work["TeamAID"].isin(team_ids)
        & work["TeamBID"].isin(team_ids)
        & work["CommenceTimeParsed"].notna()
        & (work["CommenceTimeParsed"] >= start_bound)
        & (work["CommenceTimeParsed"] <= end_bound)
    )
    out = work.loc[mask].copy()
    return out.drop(columns=["CommenceTimeParsed"], errors="ignore").reset_index(drop=True)


def load_optional_external_team_features_for_season(paths: AppPaths, data, season: int) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    candidate_paths = [paths.win_loss_csv, paths.ats_csv, paths.ou_csv, paths.tempo_csv, paths.lineup_csv]
    if not any(path.exists() for path in candidate_paths):
        return None, {"status": "no_external_files_present"}

    try:
        ext_bundle = load_external_team_features(
            teams_df=data.teams,
            spellings_df=data.spellings,
            target_season=int(season),
            win_loss_csv=str(paths.win_loss_csv) if paths.win_loss_csv.exists() else None,
            ats_csv=str(paths.ats_csv) if paths.ats_csv.exists() else None,
            ou_csv=str(paths.ou_csv) if paths.ou_csv.exists() else None,
            tempo_csv=str(paths.tempo_csv) if paths.tempo_csv.exists() else None,
            lineup_csv=str(paths.lineup_csv) if paths.lineup_csv.exists() else None,
            aliases_csv=discover_team_aliases_path(),
            strict=False,
            default_season=None,
        )
        return ext_bundle.frame, {
            "status": "ok",
            "coverage": ext_bundle.coverage,
            "unresolved": ext_bundle.unresolved,
        }
    except Exception as exc:
        return None, {"status": "error", "error": str(exc)}


def observed_results_to_actual_games(paths: AppPaths, season: int, observed_results: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Season",
        "DayNum",
        "CompetitionPhase",
        "RoundLabel",
        "GameDateET",
        "WTeamID",
        "LTeamID",
        "WScore",
        "LScore",
        "ActualWinnerID",
        "ActualLoserID",
        "PairKey",
    ]
    if observed_results.empty:
        return pd.DataFrame(columns=cols)

    work = observed_results.copy()
    for col in ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.loc[work[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]].notna().all(axis=1)].copy()
    if work.empty:
        return pd.DataFrame(columns=cols)

    day_zero = get_season_day_zero(paths, season)
    game_dates = (day_zero + pd.to_timedelta(work["DayNum"].astype(int), unit="D")).dt.date.astype(str)
    out = pd.DataFrame(
        {
            "Season": work["Season"].astype(int),
            "DayNum": work["DayNum"].astype(int),
            "CompetitionPhase": "tournament",
            "RoundLabel": "Tournament",
            "GameDateET": game_dates,
            "WTeamID": work["WTeamID"].astype(int),
            "LTeamID": work["LTeamID"].astype(int),
            "WScore": work["WScore"].astype(int),
            "LScore": work["LScore"].astype(int),
            "ActualWinnerID": work["WTeamID"].astype(int),
            "ActualLoserID": work["LTeamID"].astype(int),
        }
    )
    out["PairKey"] = out.apply(lambda r: pair_key(r["WTeamID"], r["LTeamID"]), axis=1)
    return out[cols].sort_values(["DayNum", "WTeamID", "LTeamID"]).reset_index(drop=True)



def build_actual_games_with_observed(
    paths: AppPaths,
    data,
    season: int,
    scope: str = "season",
    observed_scores_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = build_actual_games(paths, season, scope).copy()
    observed_results = scores_frame_to_results(paths, season, observed_scores_df if observed_scores_df is not None else pd.DataFrame())
    observed_games = observed_results_to_actual_games(paths, season, observed_results)
    if observed_games.empty:
        return base
    if base.empty:
        return observed_games
    combined = pd.concat([base, observed_games], ignore_index=True)
    combined["__GameKey"] = combined["PairKey"].astype(str) + "|" + combined["DayNum"].astype(str) + "|" + combined["CompetitionPhase"].astype(str)
    combined = combined.sort_values(["__GameKey", "CompetitionPhase"]).drop_duplicates(subset=["__GameKey"], keep="last")
    return combined.drop(columns=["__GameKey"], errors="ignore").reset_index(drop=True)



def market_history_frame_columns() -> list[str]:
    return [
        "Season",
        "TeamAID",
        "TeamBID",
        "EventID",
        "CommenceTime",
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
        "DayNum",
        "CompetitionPhase",
        "RoundLabel",
        "PairKey",
    ]



def standardize_market_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = market_history_frame_columns()
    if frame.empty:
        return pd.DataFrame(columns=cols)
    work = frame.copy()
    defaults = {
        "EventID": "",
        "CommenceTime": "",
        "MarketProbTeamA": np.nan,
        "MarketProbTeamB": np.nan,
        "MarketMoneylineTeamA": np.nan,
        "MarketMoneylineTeamB": np.nan,
        "MarketSpreadTeamA": np.nan,
        "MarketSpreadPriceA": np.nan,
        "MarketSpreadPriceB": np.nan,
        "MarketTotal": np.nan,
        "MarketOverPrice": np.nan,
        "MarketUnderPrice": np.nan,
        "MarketBookCount": 0.0,
        "MoneylineHold": np.nan,
        "SpreadHold": np.nan,
        "TotalHold": np.nan,
        "CompetitionPhase": "",
        "RoundLabel": "",
        "PairKey": "",
    }
    for col, default in defaults.items():
        if col not in work.columns:
            work[col] = default
    for col in [
        "Season",
        "TeamAID",
        "TeamBID",
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
        "DayNum",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    return work[[col for col in cols if col in work.columns]].copy()



def load_same_season_market_training_history(
    paths: AppPaths,
    data,
    season: int,
    cutoff_daynum: int | None = None,
    observed_scores_df: pd.DataFrame | None = None,
    aliases: dict[str, int] | None = None,
) -> pd.DataFrame:
    actual_games = build_actual_games_with_observed(paths, data, season, scope="season", observed_scores_df=observed_scores_df)
    if actual_games.empty:
        return pd.DataFrame(columns=market_history_frame_columns())

    if aliases is None:
        aliases = build_alias_lookup(data)

    frames: list[pd.DataFrame] = []
    for scope_name in ["season", "tournament"]:
        raw_path = paths.outputs / "odds_history" / f"{int(season)}_{scope_name}" / "raw_snapshots.csv"
        if not raw_path.exists():
            continue
        try:
            raw = pd.read_csv(raw_path, low_memory=False)
        except Exception:
            continue
        if raw.empty:
            continue
        raw, _ = safe_resolve_market_team_ids(raw, aliases, season=season)
        closing = select_latest_pregame_snapshot(
            raw,
            pregame_buffer_minutes=int(load_settings(paths).get("historical_pregame_buffer_minutes", 5)),
        )
        if closing.empty:
            continue
        consensus = build_consensus_matchups(closing)
        matched, _ = match_market_events_to_actuals(consensus, actual_games)
        if not matched.empty:
            source = standardize_market_history_frame(matched)
            source["HistorySource"] = f"raw_{scope_name}"
            frames.append(source)

    observed_market = read_cached_frame(observed_market_cache_path(paths, season))
    if not observed_market.empty:
        observed_market, _ = safe_resolve_market_team_ids(observed_market, aliases, season=season)
        observed_market = filter_market_rows_to_tournament_field(observed_market, data, season, paths)
        if not observed_market.empty:
            observed_consensus = build_consensus_matchups(observed_market)
            observed_matched, _ = match_market_events_to_actuals(observed_consensus, actual_games)
            if not observed_matched.empty:
                source = standardize_market_history_frame(observed_matched)
                source["HistorySource"] = "observed_market_cache"
                frames.append(source)

    if not frames:
        return pd.DataFrame(columns=market_history_frame_columns() + ["HistorySource"])

    combined = pd.concat(frames, ignore_index=True)
    if cutoff_daynum is not None:
        combined = combined.loc[pd.to_numeric(combined["DayNum"], errors="coerce") <= int(cutoff_daynum)].copy()
    if combined.empty:
        return pd.DataFrame(columns=market_history_frame_columns() + ["HistorySource"])

    if "PairKey" not in combined.columns or combined["PairKey"].eq("").all():
        combined["PairKey"] = combined.apply(lambda r: pair_key(r["TeamAID"], r["TeamBID"]), axis=1)
    combined["__GameKey"] = combined["PairKey"].astype(str) + "|" + pd.to_numeric(combined["DayNum"], errors="coerce").fillna(-1).astype(int).astype(str)
    combined["__BookSort"] = pd.to_numeric(combined["MarketBookCount"], errors="coerce").fillna(0.0)
    combined["__CommenceSort"] = pd.to_datetime(combined.get("CommenceTime"), utc=True, errors="coerce")
    combined = combined.sort_values(["__GameKey", "__BookSort", "__CommenceSort"], ascending=[True, False, True])
    combined = combined.drop_duplicates(subset=["__GameKey"], keep="first")
    return combined.drop(columns=["__GameKey", "__BookSort", "__CommenceSort"], errors="ignore").reset_index(drop=True)




def snapshot_training_seasons(target_season: int, settings: dict[str, Any]) -> list[int]:
    lookback = max(0, int(settings.get("snapshot_training_prior_seasons", 1)))
    min_season = int(settings.get("snapshot_training_min_season", 2003))
    start = max(min_season, int(target_season) - lookback)
    return list(range(start, int(target_season) + 1))


def direct_market_training_seasons(target_season: int, settings: dict[str, Any]) -> list[int]:
    lookback = max(0, int(settings.get("snapshot_direct_market_history_lookback_seasons", 1)))
    min_season = int(settings.get("snapshot_direct_market_history_min_season", 2021))
    start = max(min_season, int(target_season) - lookback)
    return list(range(start, int(target_season) + 1))



def load_multiseason_market_training_history(
    paths: AppPaths,
    data,
    target_season: int,
    settings: dict[str, Any],
    cutoff_daynum: int | None = None,
    observed_scores_df: pd.DataFrame | None = None,
    aliases: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    seasons = direct_market_training_seasons(target_season, settings)
    frames: list[pd.DataFrame] = []
    used_seasons: list[int] = []
    rows_by_season: dict[str, int] = {}
    missing_cached_seasons: list[int] = []

    for season in seasons:
        season_cutoff = int(cutoff_daynum) if (cutoff_daynum is not None and int(season) == int(target_season)) else None
        observed = observed_scores_df if int(season) == int(target_season) else pd.DataFrame()
        frame = load_same_season_market_training_history(
            paths=paths,
            data=data,
            season=int(season),
            cutoff_daynum=season_cutoff,
            observed_scores_df=observed,
            aliases=aliases,
        )
        if frame.empty:
            missing_cached_seasons.append(int(season))
            continue
        frame = standardize_market_history_frame(frame)
        frame["HistoryTargetSeason"] = int(target_season)
        frame["HistoryTrainingSeason"] = int(season)
        frames.append(frame)
        used_seasons.append(int(season))
        rows_by_season[str(int(season))] = int(len(frame))

    if not frames:
        meta = {
            "requested_seasons": seasons,
            "used_seasons": [],
            "missing_cached_seasons": missing_cached_seasons,
            "rows": 0,
            "rows_by_season": {},
        }
        return pd.DataFrame(columns=market_history_frame_columns() + ["HistoryTargetSeason", "HistoryTrainingSeason"]), meta

    combined = pd.concat(frames, ignore_index=True)
    combined["Season"] = pd.to_numeric(combined["Season"], errors="coerce")
    if "PairKey" not in combined.columns or combined["PairKey"].eq("").all():
        combined["PairKey"] = combined.apply(lambda r: pair_key(r["TeamAID"], r["TeamBID"]), axis=1)
    combined["__GameKey"] = (
        pd.to_numeric(combined["Season"], errors="coerce").fillna(-1).astype(int).astype(str)
        + "|"
        + combined["PairKey"].astype(str)
        + "|"
        + pd.to_numeric(combined["DayNum"], errors="coerce").fillna(-1).astype(int).astype(str)
    )
    combined["__BookSort"] = pd.to_numeric(combined["MarketBookCount"], errors="coerce").fillna(0.0)
    combined["__CommenceSort"] = pd.to_datetime(combined.get("CommenceTime"), utc=True, errors="coerce")
    combined = combined.sort_values(["__GameKey", "__BookSort", "__CommenceSort"], ascending=[True, False, True])
    combined = combined.drop_duplicates(subset=["__GameKey"], keep="first")
    combined = combined.drop(columns=["__GameKey", "__BookSort", "__CommenceSort"], errors="ignore").reset_index(drop=True)
    meta = {
        "requested_seasons": seasons,
        "used_seasons": sorted(int(x) for x in combined["Season"].dropna().unique().tolist()),
        "missing_cached_seasons": missing_cached_seasons,
        "rows": int(len(combined)),
        "rows_by_season": rows_by_season,
    }
    return combined, meta



def fetch_recent_scores(
    api_key: str,
    sport: str,
    days_from: int = 3,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    max_age_seconds: int | None = None,
    timeout: int = 60,
    max_retries: int = 4,
    retry_backoff_seconds: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    params: dict[str, Any] = {
        "daysFrom": int(max(1, min(3, int(days_from)))),
        "dateFormat": "iso",
    }
    payload, headers, cache_meta = odds_api_get(
        path=f"/v4/sports/{sport}/scores",
        api_key=api_key,
        params=params,
        timeout=timeout,
        cache_dir=cache_dir,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        max_age_seconds=max_age_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    flat = flatten_scores_api_response(list(payload))
    meta = {
        "quota_remaining": headers.get("x-requests-remaining"),
        "quota_used": headers.get("x-requests-used"),
        "quota_last": headers.get("x-requests-last"),
        "rows": int(len(flat)),
        **cache_meta,
    }
    return flat, meta


def merge_market_observation_cache(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if existing.empty and new_rows.empty:
        return pd.DataFrame()
    if existing.empty:
        combined = new_rows.copy()
    elif new_rows.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True)

    if "ObservedAtUTC" not in combined.columns:
        combined["ObservedAtUTC"] = ""
    combined["ObservedAtParsed"] = pd.to_datetime(combined["ObservedAtUTC"], utc=True, errors="coerce")
    combined["BookLastUpdateParsed"] = pd.to_datetime(combined.get("BookLastUpdate"), utc=True, errors="coerce")

    subset = [col for col in ["EventID", "Bookmaker", "TeamAID", "TeamBID", "CommenceTime"] if col in combined.columns]
    if not subset:
        return combined.drop(columns=["ObservedAtParsed", "BookLastUpdateParsed"], errors="ignore")

    combined = combined.sort_values(subset + ["ObservedAtParsed", "BookLastUpdateParsed"], na_position="last")
    combined = combined.drop_duplicates(subset=subset, keep="last")
    combined = combined.sort_values(["CommenceTime", "TeamAID", "TeamBID", "Bookmaker"], na_position="last")
    return combined.drop(columns=["ObservedAtParsed", "BookLastUpdateParsed"], errors="ignore").reset_index(drop=True)


def merge_score_observation_cache(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if existing.empty and new_rows.empty:
        return pd.DataFrame()
    if existing.empty:
        combined = new_rows.copy()
    elif new_rows.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True)

    combined["LastUpdateParsed"] = pd.to_datetime(combined.get("LastUpdate"), utc=True, errors="coerce")
    combined["CommenceTimeParsed"] = pd.to_datetime(combined.get("CommenceTime"), utc=True, errors="coerce")
    subset = [col for col in ["EventID"] if col in combined.columns]
    if not subset:
        subset = [col for col in ["TeamAID", "TeamBID", "CommenceTime"] if col in combined.columns]

    if subset:
        combined = combined.sort_values(subset + ["Completed", "LastUpdateParsed", "CommenceTimeParsed"], ascending=[True] * len(subset) + [True, True, True], na_position="last")
        combined = combined.drop_duplicates(subset=subset, keep="last")
    return combined.drop(columns=["LastUpdateParsed", "CommenceTimeParsed"], errors="ignore").reset_index(drop=True)


def update_observed_market_cache(
    paths: AppPaths,
    season: int,
    market_rows: pd.DataFrame,
    source_label: str,
) -> pd.DataFrame:
    cache_path = observed_market_cache_path(paths, season)
    existing = read_cached_frame(cache_path)
    if market_rows.empty:
        return existing
    stamped = market_rows.copy()
    stamped["Season"] = int(season)
    stamped["ObservedAtUTC"] = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    stamped["ObservedSource"] = source_label
    merged = merge_market_observation_cache(existing, stamped)
    ensure_dir(cache_path.parent)
    merged.to_csv(cache_path, index=False)
    return merged


def update_observed_score_cache(
    paths: AppPaths,
    season: int,
    score_rows: pd.DataFrame,
) -> pd.DataFrame:
    cache_path = observed_scores_cache_path(paths, season)
    existing = read_cached_frame(cache_path)
    if score_rows.empty:
        return existing
    stamped = score_rows.copy()
    stamped["Season"] = int(season)
    merged = merge_score_observation_cache(existing, stamped)
    ensure_dir(cache_path.parent)
    merged.to_csv(cache_path, index=False)
    return merged


def scores_frame_to_results(
    paths: AppPaths,
    season: int,
    scores_df: pd.DataFrame,
) -> pd.DataFrame:
    if scores_df.empty:
        return pd.DataFrame(columns=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "EventID", "CommenceTime"])
    work = scores_df.copy()
    work["ScoreTeamA"] = pd.to_numeric(work["ScoreTeamA"], errors="coerce")
    work["ScoreTeamB"] = pd.to_numeric(work["ScoreTeamB"], errors="coerce")
    work["TeamAID"] = pd.to_numeric(work["TeamAID"], errors="coerce")
    work["TeamBID"] = pd.to_numeric(work["TeamBID"], errors="coerce")
    work["Completed"] = work["Completed"].fillna(False).astype(bool)
    work = work.loc[
        work["Completed"]
        & work["ScoreTeamA"].notna()
        & work["ScoreTeamB"].notna()
        & work["TeamAID"].notna()
        & work["TeamBID"].notna()
        & (work["ScoreTeamA"] != work["ScoreTeamB"])
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "EventID", "CommenceTime"])
    work["DayNum"] = commence_series_to_daynum(paths, season, work["CommenceTime"])
    work = work.loc[work["DayNum"].notna()].copy()
    work["DayNum"] = pd.to_numeric(work["DayNum"], errors="coerce").astype(int)
    team_a_win = work["ScoreTeamA"] > work["ScoreTeamB"]
    out = pd.DataFrame(
        {
            "Season": int(season),
            "DayNum": work["DayNum"].astype(int),
            "WTeamID": np.where(team_a_win, work["TeamAID"], work["TeamBID"]).astype(int),
            "LTeamID": np.where(team_a_win, work["TeamBID"], work["TeamAID"]).astype(int),
            "WScore": np.where(team_a_win, work["ScoreTeamA"], work["ScoreTeamB"]).astype(int),
            "LScore": np.where(team_a_win, work["ScoreTeamB"], work["ScoreTeamA"]).astype(int),
            "EventID": work["EventID"].astype(str),
            "CommenceTime": work["CommenceTime"].astype(str),
        }
    )
    out = out.drop_duplicates(subset=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]).sort_values(["DayNum", "WTeamID", "LTeamID"]).reset_index(drop=True)
    return out


def results_to_training_rows(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["Season", "TeamAID", "TeamBID", "TeamAScore", "TeamBScore", "TeamAWin", "Margin", "Total"])

    base = results.copy()
    forward = pd.DataFrame(
        {
            "Season": base["Season"].astype(int),
            "TeamAID": base["WTeamID"].astype(int),
            "TeamBID": base["LTeamID"].astype(int),
            "TeamAScore": pd.to_numeric(base["WScore"], errors="coerce"),
            "TeamBScore": pd.to_numeric(base["LScore"], errors="coerce"),
            "TeamAWin": 1,
        }
    )
    reverse = pd.DataFrame(
        {
            "Season": base["Season"].astype(int),
            "TeamAID": base["LTeamID"].astype(int),
            "TeamBID": base["WTeamID"].astype(int),
            "TeamAScore": pd.to_numeric(base["LScore"], errors="coerce"),
            "TeamBScore": pd.to_numeric(base["WScore"], errors="coerce"),
            "TeamAWin": 0,
        }
    )
    rows = pd.concat([forward, reverse], ignore_index=True)
    rows["Margin"] = rows["TeamAScore"] - rows["TeamBScore"]
    rows["Total"] = rows["TeamAScore"] + rows["TeamBScore"]
    return rows.loc[rows["TeamAScore"].notna() & rows["TeamBScore"].notna()].reset_index(drop=True)


def combine_simple_results_for_snapshot(
    data,
    paths: AppPaths,
    season: int,
    cutoff_daynum: int,
    observed_scores_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regular = data.regular.loc[
        (data.regular["Season"] == int(season))
        & (data.regular["DayNum"] <= int(cutoff_daynum)),
        ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"],
    ].copy()

    kaggle_tourney = data.tourney.loc[
        (data.tourney["Season"] == int(season))
        & (data.tourney["DayNum"] <= int(cutoff_daynum)),
        ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"],
    ].copy()

    observed_results = scores_frame_to_results(paths, season, observed_scores_df if observed_scores_df is not None else pd.DataFrame())
    observed_results = observed_results.loc[observed_results["DayNum"] <= int(cutoff_daynum), ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]].copy()

    tournament_results = pd.concat([kaggle_tourney, observed_results], ignore_index=True)
    if not tournament_results.empty:
        tournament_results = (
            tournament_results.drop_duplicates(subset=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"])
            .sort_values(["DayNum", "WTeamID", "LTeamID"])
            .reset_index(drop=True)
        )

    training_results = pd.concat([regular, tournament_results], ignore_index=True)
    if not training_results.empty:
        training_results = (
            training_results.drop_duplicates(subset=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"])
            .sort_values(["DayNum", "WTeamID", "LTeamID"])
            .reset_index(drop=True)
        )
    return training_results, tournament_results




def combine_windowed_results_for_snapshot(
    data,
    paths: AppPaths,
    season: int,
    cutoff_daynum: int,
    settings: dict[str, Any],
    observed_scores_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_training_results, tournament_results = combine_simple_results_for_snapshot(
        data=data,
        paths=paths,
        season=season,
        cutoff_daynum=cutoff_daynum,
        observed_scores_df=observed_scores_df,
    )

    frames: list[pd.DataFrame] = []
    if not current_training_results.empty:
        frames.append(current_training_results)

    for prior_season in snapshot_training_seasons(season, settings):
        if int(prior_season) == int(season):
            continue
        regular = data.regular.loc[
            data.regular["Season"] == int(prior_season),
            ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"],
        ].copy()
        kaggle_tourney = data.tourney.loc[
            data.tourney["Season"] == int(prior_season),
            ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"],
        ].copy()
        prior_results = pd.concat([regular, kaggle_tourney], ignore_index=True)
        if not prior_results.empty:
            prior_results = (
                prior_results.drop_duplicates(subset=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"])
                .sort_values(["Season", "DayNum", "WTeamID", "LTeamID"])
                .reset_index(drop=True)
            )
            frames.append(prior_results)

    if not frames:
        training_results = pd.DataFrame(columns=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"])
    else:
        training_results = pd.concat(frames, ignore_index=True)
        training_results = (
            training_results.drop_duplicates(subset=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"])
            .sort_values(["Season", "DayNum", "WTeamID", "LTeamID"])
            .reset_index(drop=True)
        )
    return training_results, tournament_results


def combine_simple_results_for_direct_market_training(
    data,
    paths: AppPaths,
    target_season: int,
    settings: dict[str, Any],
    cutoff_daynum: int | None = None,
    observed_scores_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    seasons = direct_market_training_seasons(target_season, settings)
    frames: list[pd.DataFrame] = []

    for season in seasons:
        if int(season) == int(target_season):
            season_cutoff = int(cutoff_daynum) if cutoff_daynum is not None else 999
            training_results, _ = combine_simple_results_for_snapshot(
                data=data,
                paths=paths,
                season=int(season),
                cutoff_daynum=season_cutoff,
                observed_scores_df=observed_scores_df,
            )
        else:
            regular = data.regular.loc[
                data.regular["Season"] == int(season),
                ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"],
            ].copy()
            kaggle_tourney = data.tourney.loc[
                data.tourney["Season"] == int(season),
                ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"],
            ].copy()
            training_results = pd.concat([regular, kaggle_tourney], ignore_index=True)
            if not training_results.empty:
                training_results = (
                    training_results.drop_duplicates(subset=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"])
                    .sort_values(["Season", "DayNum", "WTeamID", "LTeamID"])
                    .reset_index(drop=True)
                )
        if not training_results.empty:
            frames.append(training_results)

    if not frames:
        return pd.DataFrame(columns=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"])

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.drop_duplicates(subset=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"])
        .sort_values(["Season", "DayNum", "WTeamID", "LTeamID"])
        .reset_index(drop=True)
    )
    return combined



def build_multiseason_direct_team_features(
    data,
    target_season: int,
    cutoff_daynum: int,
    direct_training_results: pd.DataFrame,
    current_tournament_results: pd.DataFrame,
    external_team_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if direct_training_results is None or direct_training_results.empty:
        return pd.DataFrame(columns=["Season", "TeamID"])

    seasons = sorted(int(x) for x in pd.to_numeric(direct_training_results["Season"], errors="coerce").dropna().unique().tolist())
    prior_seasons = [season for season in seasons if int(season) != int(target_season)]

    detailed_frames: list[pd.DataFrame] = []
    if prior_seasons:
        prior_regular = data.regular.loc[data.regular["Season"].isin(prior_seasons)].copy()
        prior_tourney = data.tourney.loc[data.tourney["Season"].isin(prior_seasons)].copy()
        if not prior_tourney.empty:
            if "WLoc" not in prior_tourney.columns:
                prior_tourney["WLoc"] = "N"
            regular_cols = list(prior_regular.columns) if not prior_regular.empty else list(data.regular.columns)
            for col in regular_cols:
                if col not in prior_tourney.columns:
                    prior_tourney[col] = np.nan
            prior_tourney = prior_tourney[regular_cols]
        if not prior_regular.empty:
            detailed_frames.append(prior_regular)
        if not prior_tourney.empty:
            detailed_frames.append(prior_tourney)

    target_detailed = build_detailed_results_for_snapshot(data, target_season, cutoff_daynum)
    if not target_detailed.empty:
        detailed_frames.append(target_detailed)

    if not detailed_frames:
        return pd.DataFrame(columns=["Season", "TeamID"])

    detailed = pd.concat(detailed_frames, ignore_index=True).reset_index(drop=True)
    seeds = data.seeds.loc[data.seeds["Season"].isin(seasons)].copy()
    massey = data.massey.loc[data.massey["Season"].isin(seasons)].copy() if getattr(data, "massey", None) is not None and not data.massey.empty else None

    team_features = build_team_features(
        regular_results=detailed,
        seeds=seeds,
        massey=massey,
        cutoff_daynum=999,
    )

    tournament_extra = build_score_only_tournament_features(current_tournament_results)
    if not tournament_extra.empty:
        team_features = team_features.merge(tournament_extra, on=["Season", "TeamID"], how="left")
        extra_cols = [col for col in tournament_extra.columns if col not in {"Season", "TeamID"}]
        for col in extra_cols:
            team_features[col] = pd.to_numeric(team_features[col], errors="coerce").fillna(0.0)

    if external_team_features is not None and not external_team_features.empty:
        team_features = merge_external_team_features(team_features, external_team_features)

    return team_features.sort_values(["Season", "TeamID"]).reset_index(drop=True)



def build_detailed_results_for_snapshot(
    data,
    season: int,
    cutoff_daynum: int,
) -> pd.DataFrame:
    regular = data.regular.loc[
        (data.regular["Season"] == int(season))
        & (data.regular["DayNum"] <= int(cutoff_daynum))
    ].copy()

    tourney = data.tourney.loc[
        (data.tourney["Season"] == int(season))
        & (data.tourney["DayNum"] <= int(cutoff_daynum))
    ].copy()
    if not tourney.empty:
        if "WLoc" not in tourney.columns:
            tourney["WLoc"] = "N"
        for col in regular.columns:
            if col not in tourney.columns:
                tourney[col] = np.nan
        tourney = tourney[regular.columns]

    if regular.empty and tourney.empty:
        return regular
    if regular.empty:
        return tourney.reset_index(drop=True)
    if tourney.empty:
        return regular.reset_index(drop=True)
    return pd.concat([regular, tourney], ignore_index=True).reset_index(drop=True)


def build_score_only_tournament_features(tournament_results: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Season",
        "TeamID",
        "CurrTourneyGames",
        "CurrTourneyWins",
        "CurrTourneyWinPct",
        "CurrTourneyPointsForMean",
        "CurrTourneyPointsAgainstMean",
        "CurrTourneyMarginMean",
        "CurrTourneyTotalMean",
        "CurrTourneyLastPointsFor",
        "CurrTourneyLastPointsAgainst",
        "CurrTourneyLastMargin",
        "CurrTourneyLastTotal",
    ]
    if tournament_results.empty:
        return pd.DataFrame(columns=cols)

    source = tournament_results.copy()
    for col in ["WScore", "LScore", "DayNum", "WTeamID", "LTeamID", "Season"]:
        source[col] = pd.to_numeric(source[col], errors="coerce")
    source = source.loc[source[["Season", "WTeamID", "LTeamID", "WScore", "LScore", "DayNum"]].notna().all(axis=1)].copy()
    if source.empty:
        return pd.DataFrame(columns=cols)

    winners = pd.DataFrame(
        {
            "Season": source["Season"].astype(int),
            "TeamID": source["WTeamID"].astype(int),
            "DayNum": source["DayNum"].astype(int),
            "PointsFor": source["WScore"].astype(float),
            "PointsAgainst": source["LScore"].astype(float),
            "Margin": source["WScore"].astype(float) - source["LScore"].astype(float),
            "Total": source["WScore"].astype(float) + source["LScore"].astype(float),
            "IsWin": 1,
        }
    )
    losers = pd.DataFrame(
        {
            "Season": source["Season"].astype(int),
            "TeamID": source["LTeamID"].astype(int),
            "DayNum": source["DayNum"].astype(int),
            "PointsFor": source["LScore"].astype(float),
            "PointsAgainst": source["WScore"].astype(float),
            "Margin": source["LScore"].astype(float) - source["WScore"].astype(float),
            "Total": source["WScore"].astype(float) + source["LScore"].astype(float),
            "IsWin": 0,
        }
    )
    team_rows = pd.concat([winners, losers], ignore_index=True).sort_values(["Season", "TeamID", "DayNum"]).reset_index(drop=True)

    agg = (
        team_rows.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            CurrTourneyGames=("IsWin", "size"),
            CurrTourneyWins=("IsWin", "sum"),
            CurrTourneyPointsForMean=("PointsFor", "mean"),
            CurrTourneyPointsAgainstMean=("PointsAgainst", "mean"),
            CurrTourneyMarginMean=("Margin", "mean"),
            CurrTourneyTotalMean=("Total", "mean"),
        )
    )
    agg["CurrTourneyWinPct"] = np.where(
        pd.to_numeric(agg["CurrTourneyGames"], errors="coerce").fillna(0) > 0,
        pd.to_numeric(agg["CurrTourneyWins"], errors="coerce") / pd.to_numeric(agg["CurrTourneyGames"], errors="coerce"),
        0.0,
    )

    last = (
        team_rows.sort_values(["Season", "TeamID", "DayNum"])
        .groupby(["Season", "TeamID"], as_index=False)
        .tail(1)[["Season", "TeamID", "PointsFor", "PointsAgainst", "Margin", "Total"]]
        .rename(
            columns={
                "PointsFor": "CurrTourneyLastPointsFor",
                "PointsAgainst": "CurrTourneyLastPointsAgainst",
                "Margin": "CurrTourneyLastMargin",
                "Total": "CurrTourneyLastTotal",
            }
        )
    )
    out = agg.merge(last, on=["Season", "TeamID"], how="left")
    return out[cols].sort_values(["Season", "TeamID"]).reset_index(drop=True)



def build_snapshot_config(
    season: int,
    cutoff_daynum: int,
    settings: dict[str, Any],
    optimize_target: str | None = None,
) -> TrainConfig:
    optimize_target = normalize_optimize_target(optimize_target or settings.get("default_optimize_for", "spread"))
    config = TrainConfig(
        target_season=int(season),
        min_train_season=max(int(settings.get("snapshot_training_min_season", 2003)), int(season) - int(settings.get("snapshot_training_prior_seasons", 1))),
        eval_start_season=max(int(settings.get("snapshot_training_min_season", 2003)), int(season) - int(settings.get("snapshot_training_prior_seasons", 1))),
        calibration_method="none",
        n_jobs=int(settings.get("model_n_jobs", 4)),
    )
    config.tourney_daynum_cutoff = int(cutoff_daynum)
    config.n_estimators_cls = int(settings.get("snapshot_n_estimators_cls", 450))
    config.n_estimators_quantile = int(settings.get("snapshot_n_estimators_quantile", 300))
    config.learning_rate = float(settings.get("snapshot_learning_rate", 0.04))
    config.max_depth = int(settings.get("snapshot_max_depth", 4))
    config.min_child_weight = float(settings.get("snapshot_min_child_weight", 4.0))
    config.subsample = float(settings.get("snapshot_subsample", 0.85))
    config.colsample_bytree = float(settings.get("snapshot_colsample_bytree", 0.85))
    config.reg_alpha = float(settings.get("snapshot_reg_alpha", 0.1))
    config.reg_lambda = float(settings.get("snapshot_reg_lambda", 2.0))
    config.gamma = float(settings.get("snapshot_gamma", 0.05))
    config.direct_market_min_rows = int(settings.get("snapshot_direct_market_min_rows", 12))
    config.direct_market_min_class_count = int(settings.get("snapshot_direct_market_min_class_count", 4))
    config.direct_market_history_lookback_seasons = int(settings.get("snapshot_direct_market_history_lookback_seasons", 1))
    config.direct_market_history_min_season = int(settings.get("snapshot_direct_market_history_min_season", 2021))

    if optimize_target == "spread":
        config.direct_cover_blend_weight = float(settings.get("snapshot_spread_direct_cover_blend_weight", 0.85))
        config.direct_total_blend_weight = float(settings.get("snapshot_spread_direct_total_blend_weight", 0.40))
        config.blend_margin_win_prob_weight = float(settings.get("snapshot_spread_win_blend_weight", 0.10))
    elif optimize_target == "total":
        config.direct_cover_blend_weight = float(settings.get("snapshot_total_direct_cover_blend_weight", 0.40))
        config.direct_total_blend_weight = float(settings.get("snapshot_total_direct_total_blend_weight", 0.85))
        config.blend_margin_win_prob_weight = float(settings.get("snapshot_total_win_blend_weight", 0.10))
    else:
        config.direct_cover_blend_weight = float(settings.get("snapshot_winner_direct_cover_blend_weight", 0.55))
        config.direct_total_blend_weight = float(settings.get("snapshot_winner_direct_total_blend_weight", 0.55))
        config.blend_margin_win_prob_weight = float(settings.get("snapshot_winner_win_blend_weight", 0.20))
    return config



def build_snapshot_model_key(
    season: int,
    cutoff_daynum: int,
    training_results: pd.DataFrame,
    tournament_results: pd.DataFrame,
    external_team_features: pd.DataFrame | None,
    market_history_rows: pd.DataFrame | None,
    direct_market_training_results: pd.DataFrame | None,
    direct_market_history_rows: pd.DataFrame | None,
    config: TrainConfig | None = None,
) -> str:
    training_sig = dataframe_signature(training_results, ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"])
    tournament_sig = dataframe_signature(tournament_results, ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"])
    external_sig = dataframe_signature(external_team_features) if external_team_features is not None else "noext"
    market_sig = dataframe_signature(market_history_rows) if market_history_rows is not None else "nomarket"
    direct_training_sig = dataframe_signature(direct_market_training_results, ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]) if direct_market_training_results is not None else "nodirectresults"
    direct_market_sig = dataframe_signature(direct_market_history_rows) if direct_market_history_rows is not None else "nodirectmarket"
    config_sig = hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()[:12] if config is not None else "noconfig"
    blob = json.dumps(
        {
            "version": SNAPSHOT_MODEL_VERSION,
            "season": int(season),
            "cutoff_daynum": int(cutoff_daynum),
            "training_sig": training_sig,
            "tournament_sig": tournament_sig,
            "external_sig": external_sig,
            "market_sig": market_sig,
            "direct_training_sig": direct_training_sig,
            "direct_market_sig": direct_market_sig,
            "config_sig": config_sig,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return f"day{int(cutoff_daynum):03d}_{digest}"



def load_or_train_season_snapshot_model(
    paths: AppPaths,
    settings: dict[str, Any],
    data,
    season: int,
    cutoff_daynum: int,
    training_results: pd.DataFrame,
    tournament_results: pd.DataFrame,
    external_team_features: pd.DataFrame | None,
    external_meta: dict[str, Any] | None,
    market_history_rows: pd.DataFrame | None = None,
    direct_market_training_results: pd.DataFrame | None = None,
    direct_market_history_rows: pd.DataFrame | None = None,
    force_retrain: bool = False,
    optimize_target: str | None = None,
) -> tuple[Path, pd.DataFrame, Any, dict[str, Any]]:
    optimize_target = normalize_optimize_target(optimize_target or settings.get("default_optimize_for", "spread"))
    config = build_snapshot_config(season, cutoff_daynum, settings, optimize_target=optimize_target)
    snapshot_key = build_snapshot_model_key(
        season,
        cutoff_daynum,
        training_results,
        tournament_results,
        external_team_features,
        market_history_rows,
        direct_market_training_results,
        direct_market_history_rows,
        config=config,
    )
    artifact_dir = season_snapshot_model_root(paths, season) / snapshot_key
    model_path = artifact_dir / "model_bundle.joblib"
    snapshot_meta_path = artifact_dir / "snapshot_meta.json"

    training_games_used = int(len(training_results))
    training_window = snapshot_training_seasons(season, settings)
    regular_games_used = int(
        len(
            data.regular.loc[
                data.regular["Season"].isin(training_window)
                & (
                    (data.regular["Season"] < int(season))
                    | (pd.to_numeric(data.regular["DayNum"], errors="coerce") <= int(cutoff_daynum))
                )
            ]
        )
    )
    tournament_games_used = max(0, int(training_games_used - regular_games_used))
    direct_training_games_used = int(len(direct_market_training_results)) if direct_market_training_results is not None else 0
    direct_market_rows_used = int(len(direct_market_history_rows)) if direct_market_history_rows is not None else 0
    direct_market_seasons_used = (
        sorted(int(x) for x in pd.to_numeric(direct_market_history_rows["Season"], errors="coerce").dropna().unique().tolist())
        if direct_market_history_rows is not None and not direct_market_history_rows.empty and "Season" in direct_market_history_rows.columns
        else []
    )

    if not force_retrain and model_path.exists():
        ok, reason = validate_bundle_runtime_compatibility(artifact_dir)
        if ok:
            team_features = load_team_features_from_artifacts(artifact_dir)
            bundle = load_bundle(artifact_dir)
            meta = json.loads(snapshot_meta_path.read_text(encoding="utf-8")) if snapshot_meta_path.exists() else {}
            meta.update(
                {
                    "artifact_dir": str(artifact_dir),
                    "snapshot_key": snapshot_key,
                    "model_cache_hit": True,
                    "runtime_compatible": True,
                    "training_games_used": training_games_used,
                    "regular_games_used": regular_games_used,
                    "tournament_games_used": tournament_games_used,
                    "direct_training_games_used": direct_training_games_used,
                    "direct_market_history_rows": direct_market_rows_used,
                    "direct_market_history_seasons": direct_market_seasons_used,
                    "optimize_for": optimize_target,
                    "training_window_seasons": snapshot_training_seasons(season, settings),
                }
            )
            return artifact_dir, team_features, bundle, meta

    team_features = build_multiseason_direct_team_features(
        data=data,
        target_season=season,
        cutoff_daynum=cutoff_daynum,
        direct_training_results=training_results,
        current_tournament_results=tournament_results,
        external_team_features=external_team_features,
    )

    training_rows = results_to_training_rows(training_results)
    if market_history_rows is not None and not market_history_rows.empty:
        training_rows = merge_market_history_into_rows(training_rows, market_history_rows)
    if training_rows.empty:
        raise RuntimeError(f"No training rows available for season {season} at cutoff day {cutoff_daynum}.")

    direct_training_rows: pd.DataFrame | None = None
    direct_team_features: pd.DataFrame | None = None
    if (
        direct_market_training_results is not None
        and not direct_market_training_results.empty
        and direct_market_history_rows is not None
        and not direct_market_history_rows.empty
    ):
        direct_training_rows = results_to_training_rows(direct_market_training_results)
        direct_training_rows = merge_market_history_into_rows(direct_training_rows, direct_market_history_rows)
        if not direct_training_rows.empty:
            direct_team_features = build_multiseason_direct_team_features(
                data=data,
                target_season=season,
                cutoff_daynum=cutoff_daynum,
                direct_training_results=direct_market_training_results,
                current_tournament_results=tournament_results,
                external_team_features=external_team_features,
            )

    bundle, training_frame = train_model_bundle(
        training_rows,
        team_features,
        config,
        direct_market_training_rows=direct_training_rows,
        direct_market_team_features=direct_team_features,
    )

    if external_team_features is not None and not external_team_features.empty:
        try:
            ext_models, ext_cols, ext_fill, ext_summary = fit_external_prior_models(
                regular_results=data.regular,
                team_features=team_features,
                config=config,
            )
            bundle.external_prior_models = ext_models
            bundle.external_prior_feature_columns = ext_cols
            bundle.external_prior_fill_values = ext_fill
            bundle.external_prior_summary = ext_summary
        except Exception as exc:
            bundle.external_prior_models = {}
            bundle.external_prior_feature_columns = []
            bundle.external_prior_fill_values = {}
            bundle.external_prior_summary = {"status": "error", "error": str(exc)}

    ensure_dir(artifact_dir)
    save_bundle(bundle, artifact_dir)
    team_features.to_csv(artifact_dir / "team_features.csv", index=False)
    training_frame.head(1000).to_csv(artifact_dir / "training_frame_head.csv", index=False)

    meta = {
        "season": int(season),
        "cutoff_daynum": int(cutoff_daynum),
        "artifact_dir": str(artifact_dir),
        "snapshot_key": snapshot_key,
        "model_cache_hit": False,
        "training_games_used": training_games_used,
        "regular_games_used": regular_games_used,
        "tournament_games_used": tournament_games_used,
        "direct_training_games_used": direct_training_games_used,
        "external_meta": external_meta or {},
        "training_rows_examples": int(len(training_rows)),
        "feature_columns": int(len(getattr(bundle, "feature_columns", []) or [])),
        "market_history_rows": int(len(market_history_rows)) if market_history_rows is not None else 0,
        "direct_market_history_rows": direct_market_rows_used,
        "direct_market_history_seasons": direct_market_seasons_used,
        "direct_cover_training_rows": int(getattr(bundle, "direct_cover_training_rows", 0) or 0),
        "direct_over_training_rows": int(getattr(bundle, "direct_over_training_rows", 0) or 0),
        "optimize_for": optimize_target,
        "training_window_seasons": snapshot_training_seasons(season, settings),
        "version": SNAPSHOT_MODEL_VERSION,
    }
    snapshot_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return artifact_dir, team_features, bundle, meta



def build_round_snapshot_plan(matched_consensus: pd.DataFrame) -> pd.DataFrame:
    if matched_consensus.empty:
        return pd.DataFrame(columns=["RoundLabel", "RoundStartDay", "Games", "CutoffDayNum"])
    plan = (
        matched_consensus.groupby("RoundLabel", dropna=False)
        .agg(
            RoundStartDay=("DayNum", "min"),
            Games=("EventID", "count"),
        )
        .reset_index()
    )
    order = {
        "First Four": 1,
        "Round of 64": 2,
        "Round of 32": 3,
        "Sweet 16": 4,
        "Elite 8": 5,
        "Final Four": 6,
        "Championship": 7,
    }
    plan["RoundOrder"] = plan["RoundLabel"].map(order).fillna(999)
    plan["CutoffDayNum"] = pd.to_numeric(plan["RoundStartDay"], errors="coerce").fillna(0).astype(int) - 1
    return plan.sort_values(["RoundOrder", "RoundStartDay", "RoundLabel"]).reset_index(drop=True)


def build_current_market_source_frames(
    paths: AppPaths,
    settings: dict[str, Any],
    data,
    aliases: dict[str, int],
    season: int,
    market_source: str,
    api_key: str | None,
    use_cache: bool,
    refresh_cache: bool,
    live_cache_ttl_minutes: int,
    completed_event_ids: set[str],
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    live_rows = pd.DataFrame()
    live_meta: dict[str, Any] = {"status": "not_requested"}

    if market_source in {"live", "hybrid"}:
        if not api_key:
            raise ValueError("Live or hybrid market source requires an Odds API key.")
        live_flat, live_meta = fetch_live_odds(
            api_key=api_key,
            sport=str(settings.get("sport", "basketball_ncaab")),
            regions=str(settings.get("regions", "us")),
            markets=str(settings.get("markets", "h2h,spreads,totals")),
            cache_dir=paths.api_cache,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            max_age_seconds=max(0, int(live_cache_ttl_minutes)) * 60 if live_cache_ttl_minutes is not None else None,
            timeout=int(settings.get("request_timeout_seconds", 60)),
            max_retries=int(settings.get("request_max_retries", 4)),
            retry_backoff_seconds=float(settings.get("request_retry_backoff_seconds", 3.0)),
        )
        resolved_live, unresolved = safe_resolve_market_team_ids(live_flat, aliases, season=season)
        live_rows = filter_market_rows_to_tournament_field(resolved_live, data, season, paths)
        if not live_rows.empty:
            live_rows = live_rows.loc[~live_rows["EventID"].astype(str).isin(completed_event_ids)].copy()
        live_meta["unresolved_names"] = unresolved
        live_meta["tournament_rows"] = int(len(live_rows))

    cached_rows = read_cached_frame(observed_market_cache_path(paths, season))
    if not cached_rows.empty:
        cached_rows = filter_market_rows_to_tournament_field(cached_rows, data, season, paths)
        cached_rows = cached_rows.loc[~cached_rows["EventID"].astype(str).isin(completed_event_ids)].copy()

    prep_rows = pd.DataFrame()
    if int(season) == 2026 and paths.prep_dir.exists():
        prep_rows = prep_raw_market_rows(paths).copy()
        prep_rows = filter_market_rows_to_tournament_field(prep_rows, data, season, paths)
        prep_rows = prep_rows.loc[~prep_rows["EventID"].astype(str).isin(completed_event_ids)].copy()

    def with_pair(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        work = df.copy()
        work["PairKey"] = work.apply(lambda r: pair_key(r["TeamAID"], r["TeamBID"]), axis=1)
        return work

    live_pairs = set(with_pair(live_rows)["PairKey"].tolist()) if not live_rows.empty else set()
    cached_pairs = set(with_pair(cached_rows)["PairKey"].tolist()) if not cached_rows.empty else set()

    if market_source == "live":
        if not live_rows.empty:
            return live_rows, "live", {"live_meta": live_meta, "cached_rows": int(len(cached_rows))}
        if not cached_rows.empty:
            return cached_rows, "market_cache_fallback", {"live_meta": live_meta, "cached_rows": int(len(cached_rows))}
        raise RuntimeError("No live or cached current tournament market rows were found.")

    if market_source == "prep":
        if prep_rows.empty:
            raise RuntimeError("No bundled prep rows are available for this season.")
        return prep_rows, "prep_bundle", {"prep_rows": int(len(prep_rows))}

    # hybrid
    parts = []
    source_bits = []
    if not live_rows.empty:
        parts.append(live_rows)
        source_bits.append("live")
    cached_missing = pd.DataFrame()
    if not cached_rows.empty:
        cached_with_pair = with_pair(cached_rows)
        cached_missing = cached_with_pair.loc[~cached_with_pair["PairKey"].isin(live_pairs)].drop(columns=["PairKey"])
        if not cached_missing.empty:
            parts.append(cached_missing)
            source_bits.append("market_cache")
    prep_missing = pd.DataFrame()
    if not prep_rows.empty:
        prep_with_pair = with_pair(prep_rows)
        seen_pairs = live_pairs | set(with_pair(cached_missing)["PairKey"].tolist()) if not cached_missing.empty else live_pairs
        prep_missing = prep_with_pair.loc[~prep_with_pair["PairKey"].isin(seen_pairs)].drop(columns=["PairKey"])
        if not prep_missing.empty:
            parts.append(prep_missing)
            source_bits.append("prep_bundle")

    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if combined.empty:
        raise RuntimeError("No current tournament market rows were found from live, cache, or prep sources.")
    return combined, "+".join(source_bits), {
        "live_meta": live_meta,
        "cached_rows": int(len(cached_rows)),
        "prep_rows": int(len(prep_rows)),
        "combined_rows": int(len(combined)),
    }


def enrich_current_predictions_general(
    predictions: pd.DataFrame,
    paths: AppPaths,
    team_names: dict[int, str],
) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    pred = predictions.copy()
    pred["TeamAName"] = pred["TeamAID"].map(team_names)
    pred["TeamBName"] = pred["TeamBID"].map(team_names)
    pred["PredWinnerTeamName"] = pred["PredWinnerTeamID"].map(team_names)
    pred["CommenceTimeParsed"] = pd.to_datetime(pred["CommenceTime"], utc=True, errors="coerce")
    pred["GameDateET_Fallback"] = pred["CommenceTimeParsed"].dt.tz_convert(ET).dt.date.astype(str)
    pred["TipTimeET_Fallback"] = pred["CommenceTimeParsed"].dt.tz_convert(ET).dt.strftime("%-I:%M %p")

    try:
        meta = official_tournament_metadata(paths)[["Round", "GameDateET", "TipTimeET", "Network", "Region", "PairKey"]].copy()
    except Exception:
        meta = pd.DataFrame(columns=["Round", "GameDateET", "TipTimeET", "Network", "Region", "PairKey"])

    pred["PairKey"] = pred.apply(lambda r: pair_key(r["TeamAID"], r["TeamBID"]), axis=1)
    pred = pred.merge(meta, on="PairKey", how="left")
    pred["Round"] = pred["Round"].fillna("Current / Upcoming")
    pred["GameDateET"] = pred["GameDateET"].fillna(pred["GameDateET_Fallback"])
    pred["TipTimeET"] = pred["TipTimeET"].fillna(pred["TipTimeET_Fallback"])
    return pred.drop(columns=["PairKey", "CommenceTimeParsed", "GameDateET_Fallback", "TipTimeET_Fallback"], errors="ignore")


def build_current_snapshot_summary(
    season: int,
    artifact_dir: Path,
    market_source: str,
    predictions: pd.DataFrame,
    training_meta: dict[str, Any],
    market_meta: dict[str, Any],
    score_meta: dict[str, Any],
    observed_market_rows: int,
    observed_score_rows: int,
) -> dict[str, Any]:
    return {
        "season": int(season),
        "artifact_dir": str(artifact_dir),
        "training_mode": "selected-season snapshot",
        "market_source": market_source,
        "games_scored": int(len(predictions)),
        "model_cache_hit": bool(training_meta.get("model_cache_hit", False)),
        "snapshot_key": training_meta.get("snapshot_key"),
        "snapshot_cutoff_daynum": int(training_meta.get("cutoff_daynum", 0) or 0),
        "regular_season_games_used": int(training_meta.get("regular_games_used", 0) or 0),
        "completed_tournament_games_used": int(training_meta.get("tournament_games_used", 0) or 0),
        "market_history_rows_used": int(training_meta.get("market_history_rows", 0) or 0),
        "direct_market_history_rows_used": int(training_meta.get("direct_market_history_rows", 0) or 0),
        "direct_market_history_seasons": training_meta.get("direct_market_history_seasons", []),
        "direct_cover_training_rows": int(training_meta.get("direct_cover_training_rows", 0) or 0),
        "direct_over_training_rows": int(training_meta.get("direct_over_training_rows", 0) or 0),
        "observed_market_cache_rows": int(observed_market_rows),
        "observed_score_cache_rows": int(observed_score_rows),
        "market_meta": market_meta,
        "score_meta": score_meta,
    }


def build_current_snapshot_markdown(
    summary: dict[str, Any],
    predictions: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append(f"# {summary['season']} Current Tournament Market Snapshot")
    lines.append("")
    lines.append("Selected-season snapshot model: regular-season games from the chosen season plus any cached / recently observed completed NCAA tournament games for that same season.")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Market source: {summary.get('market_source')}")
    lines.append(f"- Games scored: {summary.get('games_scored', 0)}")
    lines.append(f"- Snapshot cutoff day number: {summary.get('snapshot_cutoff_daynum')}")
    lines.append(f"- Regular-season games used for training: {summary.get('regular_season_games_used')}")
    lines.append(f"- Completed tournament games used for training: {summary.get('completed_tournament_games_used')}")
    lines.append(f"- Snapshot model cache hit: {bool(summary.get('model_cache_hit'))}")
    lines.append(f"- Same-season market history rows used by the core snapshot model: {summary.get('market_history_rows_used')}")
    lines.append(f"- Multi-season market history rows used by direct ATS/O-U heads: {summary.get('direct_market_history_rows_used')}")
    lines.append(f"- Direct ATS/O-U history seasons: {summary.get('direct_market_history_seasons')}")
    lines.append(f"- Direct cover training rows: {summary.get('direct_cover_training_rows')}")
    lines.append(f"- Direct over/under training rows: {summary.get('direct_over_training_rows')}")
    lines.append(f"- Observed market cache rows: {summary.get('observed_market_cache_rows')}")
    lines.append(f"- Observed score cache rows: {summary.get('observed_score_cache_rows')}")
    lines.append("")
    if predictions.empty:
        lines.append("(no current tournament market rows)")
        return "\n".join(lines)

    display = predictions.copy().sort_values(["GameDateET", "TipTimeET", "CommenceTime", "TeamAName", "TeamBName"])
    for col in ["WinProbTeamA", "ModelProbCoverTeamA", "ModelProbOver"]:
        if col in display.columns:
            display[col] = display[col].map(format_pct)
    for col in ["FairSpreadTeamA", "MarketSpreadTeamA", "FairTotal", "MarketTotal", "PredMarginQ50", "PredTotalQ50"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: format_line(x, signed=(col != "FairTotal" and col != "MarketTotal")) if "Spread" in col or "Margin" in col else format_num(x))
    for col in ["FairMoneylineTeamA", "FairMoneylineTeamB", "MarketMoneylineTeamA", "MarketMoneylineTeamB"]:
        if col in display.columns:
            display[col] = display[col].map(format_moneyline)
    lines.append("## Game-by-game model output")
    lines.append("")
    lines.append(
        render_simple_table(
            display[
                [
                    "Round",
                    "GameDateET",
                    "TipTimeET",
                    "TeamAName",
                    "TeamBName",
                    "PredWinnerTeamName",
                    "WinProbTeamA",
                    "PredictedScore",
                    "FairMoneylineTeamA",
                    "FairMoneylineTeamB",
                    "MarketMoneylineTeamA",
                    "MarketMoneylineTeamB",
                    "FairSpreadTeamA",
                    "MarketSpreadTeamA",
                    "ModelProbCoverTeamA",
                    "FairTotal",
                    "MarketTotal",
                    "ModelProbOver",
                ]
            ]
        )
    )
    lines.append("")
    return "\n".join(lines)


def command_backtest(args: argparse.Namespace, paths: AppPaths, settings: dict[str, Any]) -> int:
    bootstrap(paths)
    year = int(args.year)
    scope = str(args.scope)
    optimize_for = normalize_optimize_target(getattr(args, "optimize_for", None) or settings.get("default_optimize_for", "spread"))
    if scope != "tournament":
        raise ValueError("This round-by-round season snapshot backtest is implemented for --scope tournament only.")

    output_dir = Path(args.output_dir) if args.output_dir else (paths.outputs / "backtests" / f"{year}_{scope}")
    ensure_dir(output_dir)

    raw_odds_path = Path(args.raw_odds_csv) if args.raw_odds_csv else (paths.outputs / "odds_history" / f"{year}_{scope}" / "raw_snapshots.csv")
    if not raw_odds_path.exists():
        if not getattr(args, "auto_fetch", False):
            raise FileNotFoundError(f"Historical raw odds file not found: {raw_odds_path}. Re-run with --auto-fetch or pass --raw-odds-csv.")
        pull_args = argparse.Namespace(
            year=year,
            scope=scope,
            interval_hours=args.interval_hours or int(settings.get("default_historical_interval_hours", 6)),
            sport=args.sport or settings.get("sport", "basketball_ncaab"),
            regions=args.regions or settings.get("regions", "us"),
            markets=args.markets or settings.get("markets", "h2h,spreads,totals"),
            output_dir=str(paths.outputs / "odds_history" / f"{year}_{scope}"),
            force=False,
            api_key=args.api_key,
            bookmakers=args.bookmakers,
            start_date=args.start_date,
            end_date=args.end_date,
            commence_time_from=None,
            commence_time_to=None,
            no_cache=bool(getattr(args, "no_cache", False)),
            refresh_cache=bool(getattr(args, "refresh_cache", False)),
            timeout_seconds=getattr(args, "timeout_seconds", None),
            max_retries=getattr(args, "max_retries", None),
            retry_backoff_seconds=getattr(args, "retry_backoff_seconds", None),
            fail_fast=bool(getattr(args, "fail_fast", False)),
        )
        command_pull_odds(pull_args, paths, settings)

    raw_snapshots = pd.read_csv(raw_odds_path, low_memory=False)
    if raw_snapshots.empty:
        raise RuntimeError(f"No rows in historical odds file: {raw_odds_path}")

    data = load_kaggle_data(paths.kaggle_data)
    aliases = build_alias_lookup(data)
    raw_snapshots, unresolved_names = safe_resolve_market_team_ids(raw_snapshots, aliases, season=year)
    raw_snapshots.to_csv(raw_odds_path, index=False)

    unresolved_rows_path = output_dir / "unresolved_market_rows.csv"
    unresolved_count = write_unresolved_market_rows(raw_snapshots, unresolved_rows_path)
    if unresolved_count:
        preview = ", ".join(unresolved_names[:10])
        more = "" if len(unresolved_names) <= 10 else f" (+{len(unresolved_names) - 10} more)"
        log(
            f"Found {unresolved_count} unresolved historical market row(s). "
            f"They will be excluded from consensus matching. See {unresolved_rows_path}. "
            f"Unresolved names: {preview}{more}"
        )

    closing_raw = select_latest_pregame_snapshot(
        raw_snapshots,
        pregame_buffer_minutes=int(settings.get("historical_pregame_buffer_minutes", 5)),
    )
    consensus = build_consensus_matchups(closing_raw)
    actual_games = build_actual_games(paths, year, scope)
    matched_consensus, unmatched = match_market_events_to_actuals(consensus, actual_games)

    if matched_consensus.empty:
        summary = build_backtest_summary(
            year=year,
            scope=scope,
            artifact_dir=season_snapshot_model_root(paths, year),
            raw_odds_path=raw_odds_path,
            consensus=consensus,
            matched_consensus=matched_consensus,
            unmatched=unmatched,
            game_report=pd.DataFrame(),
        )
        summary["training_note"] = (
            f"Round-by-round selected-season snapshot model for {year}. "
            f"For each NCAA tournament round, the winner and score models use completed {year} games before that round, "
            f"while direct ATS/O-U heads can also reuse prior seasons of cached market-labeled games when available."
        )
        (output_dir / "backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_text(output_dir / "backtest_report.md", build_backtest_markdown(summary, pd.DataFrame(), pd.DataFrame(), {"moneyline": pd.DataFrame(), "spread": pd.DataFrame(), "total": pd.DataFrame()}))
        log(json.dumps(summary, indent=2))
        return 0

    round_plan = build_round_snapshot_plan(matched_consensus)
    external_team_features, external_meta = load_optional_external_team_features_for_season(paths, data, year)
    full_market_training_history = load_same_season_market_training_history(
        paths=paths,
        data=data,
        season=year,
        cutoff_daynum=None,
        observed_scores_df=pd.DataFrame(),
        aliases=aliases,
    )
    direct_market_history_full, direct_market_meta = load_multiseason_market_training_history(
        paths=paths,
        data=data,
        target_season=year,
        settings=settings,
        cutoff_daynum=None,
        observed_scores_df=pd.DataFrame(),
        aliases=aliases,
    )
    direct_training_results_full = combine_simple_results_for_direct_market_training(
        data=data,
        paths=paths,
        target_season=year,
        settings=settings,
        cutoff_daynum=None,
        observed_scores_df=pd.DataFrame(),
    )

    log(
        f"Snapshot training window seasons: {snapshot_training_seasons(year, settings)} | "
        "Direct ATS/O-U market history requested seasons: "
        f"{direct_market_meta.get('requested_seasons', [])} | used cached seasons: {direct_market_meta.get('used_seasons', [])}"
        + (
            f" | missing cached seasons: {direct_market_meta.get('missing_cached_seasons', [])}"
            if direct_market_meta.get('missing_cached_seasons') else ""
        )
    )

    predictions_parts: list[pd.DataFrame] = []
    snapshot_rows: list[dict[str, Any]] = []

    for round_row in round_plan.to_dict("records"):
        round_label = str(round_row["RoundLabel"])
        cutoff_daynum = int(round_row["CutoffDayNum"])
        round_events = matched_consensus.loc[matched_consensus["RoundLabel"] == round_label].copy()
        training_results, tournament_results = combine_windowed_results_for_snapshot(
            data=data,
            paths=paths,
            season=year,
            cutoff_daynum=cutoff_daynum,
            settings=settings,
            observed_scores_df=pd.DataFrame(),
        )
        round_market_history = (
            full_market_training_history.loc[
                pd.to_numeric(full_market_training_history.get("DayNum"), errors="coerce") <= int(cutoff_daynum)
            ].copy()
            if not full_market_training_history.empty
            else pd.DataFrame(columns=market_history_frame_columns())
        )
        round_direct_market_history = (
            direct_market_history_full.loc[
                (pd.to_numeric(direct_market_history_full.get("Season"), errors="coerce") < int(year))
                | (pd.to_numeric(direct_market_history_full.get("DayNum"), errors="coerce") <= int(cutoff_daynum))
            ].copy()
            if not direct_market_history_full.empty
            else pd.DataFrame(columns=market_history_frame_columns())
        )
        round_direct_training_results = (
            direct_training_results_full.loc[
                (pd.to_numeric(direct_training_results_full.get("Season"), errors="coerce") < int(year))
                | (pd.to_numeric(direct_training_results_full.get("DayNum"), errors="coerce") <= int(cutoff_daynum))
            ].copy()
            if not direct_training_results_full.empty
            else pd.DataFrame(columns=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"])
        )

        artifact_dir, team_features, bundle, train_meta = load_or_train_season_snapshot_model(
            paths=paths,
            settings=settings,
            data=data,
            season=year,
            cutoff_daynum=cutoff_daynum,
            training_results=training_results,
            tournament_results=tournament_results,
            external_team_features=external_team_features,
            external_meta=external_meta,
            market_history_rows=round_market_history,
            direct_market_training_results=round_direct_training_results,
            direct_market_history_rows=round_direct_market_history,
            force_retrain=bool(getattr(args, "force_retrain", False)),
            optimize_target=optimize_for,
        )
        preds = predict_market_event_frame(round_events, team_features, bundle)
        preds["SnapshotRoundLabel"] = round_label
        preds["SnapshotCutoffDayNum"] = cutoff_daynum
        preds["SnapshotArtifactDir"] = str(artifact_dir)
        preds["SnapshotModelCacheHit"] = bool(train_meta.get("model_cache_hit", False))
        predictions_parts.append(preds)

        snapshot_rows.append(
            {
                "RoundLabel": round_label,
                "RoundStartDay": int(round_row["RoundStartDay"]),
                "CutoffDayNum": cutoff_daynum,
                "Games": int(len(round_events)),
                "TrainingGamesUsed": int(train_meta.get("training_games_used", 0)),
                "RegularSeasonGamesUsed": int(train_meta.get("regular_games_used", 0)),
                "CompletedTournamentGamesUsed": int(train_meta.get("tournament_games_used", 0)),
                "MarketHistoryRowsUsed": int(train_meta.get("market_history_rows", 0)),
                "DirectMarketHistoryRowsUsed": int(train_meta.get("direct_market_history_rows", 0)),
                "DirectMarketHistorySeasons": ",".join(str(x) for x in train_meta.get("direct_market_history_seasons", []) or []),
                "DirectTrainingGamesUsed": int(train_meta.get("direct_training_games_used", 0)),
                "DirectCoverTrainingRows": int(train_meta.get("direct_cover_training_rows", 0)),
                "DirectOverTrainingRows": int(train_meta.get("direct_over_training_rows", 0)),
                "ModelCacheHit": bool(train_meta.get("model_cache_hit", False)),
                "ArtifactDir": str(artifact_dir),
            }
        )

    predictions = pd.concat(predictions_parts, ignore_index=True) if predictions_parts else pd.DataFrame()
    names = team_name_map(data.teams)
    game_report = compute_backtest_game_report(matched_consensus, predictions, names)
    round_summary = build_backtest_round_summary(game_report)
    side_breakdown = build_backtest_side_breakdown(game_report)
    summary = build_backtest_summary(
        year=year,
        scope=scope,
        artifact_dir=season_snapshot_model_root(paths, year),
        raw_odds_path=raw_odds_path,
        consensus=consensus,
        matched_consensus=matched_consensus,
        unmatched=unmatched,
        game_report=game_report,
    )
    summary["training_note"] = (
        f"Round-by-round selected-season snapshot model for {year}. "
        f"Before each NCAA tournament round, the winner and score models are retrained using completed {year} regular-season games "
        f"plus the full {year - 1} season and tournament. Direct ATS/O-U heads reuse cached market-labeled games from {year - 1} plus {year} games available before that round. "
        f"Optimization target for this run: {optimize_for}. Predictions are graded against the matched closing market consensus."
    )
    summary["snapshot_rounds_trained"] = int(len(snapshot_rows))
    summary["optimize_for"] = optimize_for
    summary["snapshot_training_window_seasons"] = snapshot_training_seasons(year, settings)
    summary["external_feature_status"] = external_meta.get("status") if isinstance(external_meta, dict) else None
    summary["optimize_for"] = optimize_for
    summary["snapshot_training_window_seasons"] = snapshot_training_seasons(year, settings)
    summary["direct_market_history_requested_seasons"] = direct_market_meta.get("requested_seasons", [])
    summary["direct_market_history_used_seasons"] = direct_market_meta.get("used_seasons", [])
    summary["direct_market_history_missing_cached_seasons"] = direct_market_meta.get("missing_cached_seasons", [])

    snapshot_df = pd.DataFrame(snapshot_rows)
    if not snapshot_df.empty:
        summary["market_history_rows_used_max"] = int(pd.to_numeric(snapshot_df.get("MarketHistoryRowsUsed"), errors="coerce").max())
        summary["direct_market_history_rows_used_max"] = int(pd.to_numeric(snapshot_df.get("DirectMarketHistoryRowsUsed"), errors="coerce").max())
        summary["direct_training_games_used_max"] = int(pd.to_numeric(snapshot_df.get("DirectTrainingGamesUsed"), errors="coerce").max())
        summary["direct_cover_training_rows_max"] = int(pd.to_numeric(snapshot_df.get("DirectCoverTrainingRows"), errors="coerce").max())
        summary["direct_over_training_rows_max"] = int(pd.to_numeric(snapshot_df.get("DirectOverTrainingRows"), errors="coerce").max())
    summary_path = output_dir / "backtest_summary.json"
    matched_consensus.to_csv(output_dir / "matched_market_consensus.csv", index=False)
    predictions.to_csv(output_dir / "backtest_predictions.csv", index=False)
    game_report.to_csv(output_dir / "backtest_game_report.csv", index=False)
    round_summary.to_csv(output_dir / "backtest_round_summary.csv", index=False)
    snapshot_df.to_csv(output_dir / "backtest_round_training_snapshots.csv", index=False)
    side_breakdown.get("moneyline", pd.DataFrame()).to_csv(output_dir / "backtest_moneyline_breakdown.csv", index=False)
    side_breakdown.get("spread", pd.DataFrame()).to_csv(output_dir / "backtest_spread_breakdown.csv", index=False)
    side_breakdown.get("total", pd.DataFrame()).to_csv(output_dir / "backtest_total_breakdown.csv", index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    accuracy_table = build_backtest_accuracy_table(summary)
    accuracy_table.to_csv(output_dir / "backtest_accuracy_table.csv", index=False)
    round_accuracy = build_backtest_round_accuracy_table(round_summary)
    round_accuracy.to_csv(output_dir / "backtest_round_accuracy_table.csv", index=False)

    write_text(output_dir / "backtest_report.md", build_backtest_markdown(summary, game_report, round_summary, side_breakdown))
    log(json.dumps(summary, indent=2))
    log(f"Saved backtest outputs to: {output_dir}")
    return 0

def command_predict_current(args: argparse.Namespace, paths: AppPaths, settings: dict[str, Any]) -> int:
    bootstrap(paths)
    season = int(args.season)
    optimize_for = normalize_optimize_target(getattr(args, "optimize_for", None) or settings.get("default_optimize_for", "spread"))
    market_source = str(args.market_source or settings.get("default_market_source", "hybrid"))
    use_cache = not bool(getattr(args, "no_cache", False))
    refresh_cache = bool(getattr(args, "refresh_cache", False))
    live_cache_ttl_minutes = int(
        getattr(args, "live_cache_ttl_minutes", None)
        or settings.get("live_cache_ttl_minutes", 10)
    )

    data = load_kaggle_data(paths.kaggle_data)
    aliases = build_alias_lookup(data)
    names = team_name_map(data.teams)
    api_key = None if market_source == "prep" else read_api_key(settings, getattr(args, "api_key", None))

    observed_scores = read_cached_frame(observed_scores_cache_path(paths, season))
    scores_meta: dict[str, Any] = {"status": "not_requested"}
    if market_source != "prep":
        flat_scores, fetch_meta = fetch_recent_scores(
            api_key=api_key,
            sport=str(settings.get("sport", "basketball_ncaab")),
            days_from=3,
            cache_dir=paths.api_cache,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            max_age_seconds=max(0, live_cache_ttl_minutes) * 60 if live_cache_ttl_minutes is not None else None,
            timeout=int(settings.get("request_timeout_seconds", 60)),
            max_retries=int(settings.get("request_max_retries", 4)),
            retry_backoff_seconds=float(settings.get("request_retry_backoff_seconds", 3.0)),
        )
        resolved_scores, unresolved_scores = safe_resolve_market_team_ids(flat_scores, aliases, season=season)
        resolved_scores = filter_market_rows_to_tournament_field(resolved_scores, data, season, paths)
        resolved_scores = resolved_scores.loc[
            resolved_scores["Completed"].fillna(False).astype(bool)
            & pd.to_numeric(resolved_scores["ScoreTeamA"], errors="coerce").notna()
            & pd.to_numeric(resolved_scores["ScoreTeamB"], errors="coerce").notna()
        ].copy()
        if not resolved_scores.empty:
            resolved_scores["ScoreSource"] = "odds_api_scores"
            observed_scores = update_observed_score_cache(paths, season, resolved_scores)
        scores_meta = {
            **fetch_meta,
            "unresolved_names": unresolved_scores,
            "observed_score_rows": int(len(observed_scores)),
        }

    completed_event_ids = set()
    if not observed_scores.empty and "Completed" in observed_scores.columns:
        completed_scores = observed_scores.loc[observed_scores["Completed"].fillna(False).astype(bool)].copy()
        if "EventID" in completed_scores.columns:
            completed_event_ids = set(completed_scores["EventID"].astype(str).tolist())

    raw_market, source_label, market_meta = build_current_market_source_frames(
        paths=paths,
        settings=settings,
        data=data,
        aliases=aliases,
        season=season,
        market_source=market_source,
        api_key=api_key,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        live_cache_ttl_minutes=live_cache_ttl_minutes,
        completed_event_ids=completed_event_ids,
    )
    if raw_market.empty:
        raise RuntimeError("No current tournament market rows found after filtering.")

    observed_market = update_observed_market_cache(paths, season, raw_market, source_label)

    external_team_features, external_meta = load_optional_external_team_features_for_season(paths, data, season)
    regular_max_day = int(pd.to_numeric(data.regular.loc[data.regular["Season"] == int(season), "DayNum"], errors="coerce").max())
    completed_tourney_results = scores_frame_to_results(paths, season, observed_scores)
    latest_tourney_day = int(pd.to_numeric(completed_tourney_results["DayNum"], errors="coerce").max()) if not completed_tourney_results.empty else regular_max_day
    cutoff_daynum = max(regular_max_day, latest_tourney_day)

    training_results, tournament_results = combine_windowed_results_for_snapshot(
        data=data,
        paths=paths,
        season=season,
        cutoff_daynum=cutoff_daynum,
        settings=settings,
        observed_scores_df=observed_scores,
    )
    market_training_history = load_same_season_market_training_history(
        paths=paths,
        data=data,
        season=season,
        cutoff_daynum=cutoff_daynum,
        observed_scores_df=observed_scores,
        aliases=aliases,
    )
    direct_market_history_full, direct_market_meta = load_multiseason_market_training_history(
        paths=paths,
        data=data,
        target_season=season,
        settings=settings,
        cutoff_daynum=None,
        observed_scores_df=observed_scores,
        aliases=aliases,
    )
    direct_training_results_full = combine_simple_results_for_direct_market_training(
        data=data,
        paths=paths,
        target_season=season,
        settings=settings,
        cutoff_daynum=None,
        observed_scores_df=observed_scores,
    )
    direct_market_history = (
        direct_market_history_full.loc[
            (pd.to_numeric(direct_market_history_full.get("Season"), errors="coerce") < int(season))
            | (pd.to_numeric(direct_market_history_full.get("DayNum"), errors="coerce") <= int(cutoff_daynum))
        ].copy()
        if not direct_market_history_full.empty
        else pd.DataFrame(columns=market_history_frame_columns())
    )
    direct_training_results = (
        direct_training_results_full.loc[
            (pd.to_numeric(direct_training_results_full.get("Season"), errors="coerce") < int(season))
            | (pd.to_numeric(direct_training_results_full.get("DayNum"), errors="coerce") <= int(cutoff_daynum))
        ].copy()
        if not direct_training_results_full.empty
        else pd.DataFrame(columns=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"])
    )

    log(
        f"Snapshot training window seasons: {snapshot_training_seasons(season, settings)} | "
        "Direct ATS/O-U market history requested seasons: "
        f"{direct_market_meta.get('requested_seasons', [])} | used cached seasons: {direct_market_meta.get('used_seasons', [])}"
        + (
            f" | missing cached seasons: {direct_market_meta.get('missing_cached_seasons', [])}"
            if direct_market_meta.get('missing_cached_seasons') else ""
        )
    )

    artifact_dir, team_features, bundle, train_meta = load_or_train_season_snapshot_model(
        paths=paths,
        settings=settings,
        data=data,
        season=season,
        cutoff_daynum=cutoff_daynum,
        training_results=training_results,
        tournament_results=tournament_results,
        external_team_features=external_team_features,
        external_meta=external_meta,
        market_history_rows=market_training_history,
        direct_market_training_results=direct_training_results,
        direct_market_history_rows=direct_market_history,
        force_retrain=bool(getattr(args, "force_retrain", False)),
        optimize_target=optimize_for,
    )

    consensus = build_consensus_matchups(raw_market)
    predictions = predict_market_event_frame(consensus, team_features, bundle)
    predictions = enrich_current_predictions_general(predictions, paths, names)
    predictions["ActualScore"] = ""
    predictions["PredictedScore"] = safe_score_pair_series(predictions["PredScoreTeamA"], predictions["PredScoreTeamB"])

    output_dir = Path(args.output_dir) if args.output_dir else (paths.outputs / "current_market" / str(season))
    ensure_dir(output_dir)
    summary = build_current_snapshot_summary(
        season=season,
        artifact_dir=artifact_dir,
        market_source=source_label,
        predictions=predictions,
        training_meta=train_meta,
        market_meta=market_meta,
        score_meta=scores_meta,
        observed_market_rows=int(len(observed_market)),
        observed_score_rows=int(len(observed_scores)),
    )
    summary["external_feature_status"] = external_meta.get("status") if isinstance(external_meta, dict) else None
    summary["optimize_for"] = optimize_for
    summary["snapshot_training_window_seasons"] = snapshot_training_seasons(season, settings)
    summary["direct_market_history_requested_seasons"] = direct_market_meta.get("requested_seasons", [])
    summary["direct_market_history_used_seasons"] = direct_market_meta.get("used_seasons", [])
    summary["direct_market_history_missing_cached_seasons"] = direct_market_meta.get("missing_cached_seasons", [])

    (output_dir / "market_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    raw_market.to_csv(output_dir / "raw_market_rows.csv", index=False)
    consensus.to_csv(output_dir / "market_consensus.csv", index=False)
    predictions.to_csv(output_dir / "market_predictions.csv", index=False)
    write_text(output_dir / "market_report.md", build_current_snapshot_markdown(summary, predictions))

    log(json.dumps(summary, indent=2))
    log(f"Saved current market outputs to: {output_dir}")
    return 0

def command_predict_bracket(args: argparse.Namespace, paths: AppPaths, settings: dict[str, Any]) -> int:
    bootstrap(paths)
    season = int(args.season)
    optimize_for = normalize_optimize_target(getattr(args, "optimize_for", None) or settings.get("default_optimize_for", "spread"))
    data = load_kaggle_data(paths.kaggle_data)
    observed_scores = read_cached_frame(observed_scores_cache_path(paths, season))
    external_team_features, external_meta = load_optional_external_team_features_for_season(paths, data, season)
    regular_max_day = int(pd.to_numeric(data.regular.loc[data.regular["Season"] == int(season), "DayNum"], errors="coerce").max())
    completed_tourney_results = scores_frame_to_results(paths, season, observed_scores)
    latest_tourney_day = int(pd.to_numeric(completed_tourney_results["DayNum"], errors="coerce").max()) if not completed_tourney_results.empty else regular_max_day
    cutoff_daynum = max(regular_max_day, latest_tourney_day)
    training_results, tournament_results = combine_windowed_results_for_snapshot(
        data=data,
        paths=paths,
        season=season,
        cutoff_daynum=cutoff_daynum,
        settings=settings,
        observed_scores_df=observed_scores,
    )
    aliases = build_alias_lookup(data)
    market_training_history = load_same_season_market_training_history(
        paths=paths,
        data=data,
        season=season,
        cutoff_daynum=cutoff_daynum,
        observed_scores_df=observed_scores,
        aliases=aliases,
    )
    direct_market_history_full, direct_market_meta = load_multiseason_market_training_history(
        paths=paths,
        data=data,
        target_season=season,
        settings=settings,
        cutoff_daynum=None,
        observed_scores_df=observed_scores,
        aliases=aliases,
    )
    direct_training_results_full = combine_simple_results_for_direct_market_training(
        data=data,
        paths=paths,
        target_season=season,
        settings=settings,
        cutoff_daynum=None,
        observed_scores_df=observed_scores,
    )
    direct_market_history = (
        direct_market_history_full.loc[
            (pd.to_numeric(direct_market_history_full.get("Season"), errors="coerce") < int(season))
            | (pd.to_numeric(direct_market_history_full.get("DayNum"), errors="coerce") <= int(cutoff_daynum))
        ].copy()
        if not direct_market_history_full.empty
        else pd.DataFrame(columns=market_history_frame_columns())
    )
    direct_training_results = (
        direct_training_results_full.loc[
            (pd.to_numeric(direct_training_results_full.get("Season"), errors="coerce") < int(season))
            | (pd.to_numeric(direct_training_results_full.get("DayNum"), errors="coerce") <= int(cutoff_daynum))
        ].copy()
        if not direct_training_results_full.empty
        else pd.DataFrame(columns=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"])
    )

    log(
        f"Snapshot training window seasons: {snapshot_training_seasons(season, settings)} | "
        "Direct ATS/O-U market history requested seasons: "
        f"{direct_market_meta.get('requested_seasons', [])} | used cached seasons: {direct_market_meta.get('used_seasons', [])}"
        + (
            f" | missing cached seasons: {direct_market_meta.get('missing_cached_seasons', [])}"
            if direct_market_meta.get('missing_cached_seasons') else ""
        )
    )

    artifact_dir, team_features, bundle, _ = load_or_train_season_snapshot_model(
        paths=paths,
        settings=settings,
        data=data,
        season=season,
        cutoff_daynum=cutoff_daynum,
        training_results=training_results,
        tournament_results=tournament_results,
        external_team_features=external_team_features,
        external_meta=external_meta,
        market_history_rows=market_training_history,
        direct_market_training_results=direct_training_results,
        direct_market_history_rows=direct_market_history,
        force_retrain=bool(getattr(args, "force_retrain", False)),
        optimize_target=optimize_for,
    )

    bracket = simulate_bracket(season, data.seeds, data.slots, team_features, bundle)
    names = team_name_map(data.teams)
    bracket = bracket.copy()
    bracket["TeamAName"] = bracket["TeamAID"].map(names)
    bracket["TeamBName"] = bracket["TeamBID"].map(names)
    bracket["PredWinnerTeamName"] = bracket["PredWinnerTeamID"].map(names)

    output_dir = Path(args.output_dir) if args.output_dir else (paths.outputs / "brackets" / str(season))
    ensure_dir(output_dir)
    bracket.to_csv(output_dir / "bracket_simulation.csv", index=False)
    write_text(output_dir / "bracket_simulation.md", build_bracket_markdown(season, bracket))
    log(f"Saved bracket simulation outputs to: {output_dir}")
    return 0

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-file March Madness market workflow with selected-season plus prior-season training for backtests and current tournament scoring."
    )
    sub = parser.add_subparsers(dest="command")

    bootstrap_cmd = sub.add_parser("bootstrap", help="Extract bundled Kaggle data into the local workspace.")
    bootstrap_cmd.add_argument("--force", action="store_true")
    bootstrap_cmd.set_defaults(func=command_bootstrap)

    cache_info = sub.add_parser("cache-info", help="Show persistent cache stats, including season-state cache.")
    cache_info.set_defaults(func=command_cache_info)

    clear_cache = sub.add_parser("clear-cache", help="Delete the persistent Odds API cache. Use --include-season-state for a clean current-market reset.")
    clear_cache.add_argument("--include-season-state", action="store_true", help="Also delete cached observed market/scores under workspace/cache/season_state.")
    clear_cache.set_defaults(func=command_clear_cache)

    pull = sub.add_parser("pull-odds", help="Pull historical Odds API snapshots for a year.")
    pull.add_argument("--year", type=int, required=True)
    pull.add_argument("--scope", choices=["season", "tournament"], default="tournament")
    pull.add_argument("--interval-hours", type=int, default=6)
    pull.add_argument("--sport", default=None)
    pull.add_argument("--regions", default=None)
    pull.add_argument("--markets", default=None)
    pull.add_argument("--bookmakers", default=None)
    pull.add_argument("--api-key", default=None)
    pull.add_argument("--start-date", default=None, help="Optional UTC ISO timestamp, overrides default season window.")
    pull.add_argument("--end-date", default=None, help="Optional UTC ISO timestamp, overrides default season window.")
    pull.add_argument("--commence-time-from", default=None)
    pull.add_argument("--commence-time-to", default=None)
    pull.add_argument("--output-dir", default=None)
    pull.add_argument("--force", action="store_true")
    pull.add_argument("--no-cache", action="store_true", help="Disable persistent Odds API response caching for this run.")
    pull.add_argument("--refresh-cache", action="store_true", help="Ignore cached Odds API responses and re-fetch them.")
    pull.add_argument("--timeout-seconds", type=int, default=None, help="HTTP timeout for Odds API requests.")
    pull.add_argument("--max-retries", type=int, default=None, help="Retry count for timeout / transient server errors.")
    pull.add_argument("--retry-backoff-seconds", type=float, default=None, help="Base backoff between request retries.")
    pull.add_argument("--fail-fast", action="store_true", help="Abort immediately on the first failed historical snapshot request.")
    pull.set_defaults(func=command_pull_odds)

    backtest = sub.add_parser("backtest", help="Backtest a selected NCAA tournament using selected-season plus prior-season training.")
    backtest.add_argument("--year", type=int, default=2025)
    backtest.add_argument("--scope", choices=["tournament"], default="tournament")
    backtest.add_argument("--raw-odds-csv", default=None)
    backtest.add_argument("--auto-fetch", action="store_true")
    backtest.add_argument("--interval-hours", type=int, default=6)
    backtest.add_argument("--sport", default=None)
    backtest.add_argument("--regions", default=None)
    backtest.add_argument("--markets", default=None)
    backtest.add_argument("--bookmakers", default=None)
    backtest.add_argument("--api-key", default=None)
    backtest.add_argument("--start-date", default=None)
    backtest.add_argument("--end-date", default=None)
    backtest.add_argument("--bankroll", type=float, default=None)
    backtest.add_argument("--fractional-kelly", type=float, default=None)
    backtest.add_argument("--max-stake-fraction", type=float, default=None)
    backtest.add_argument("--min-moneyline-ev", type=float, default=0.015)
    backtest.add_argument("--min-spread-ev", type=float, default=0.010)
    backtest.add_argument("--min-total-ev", type=float, default=0.010)
    backtest.add_argument("--min-edge-prob", type=float, default=0.015)
    backtest.add_argument("--min-market-books", type=int, default=1)
    backtest.add_argument("--force-retrain", action="store_true")
    backtest.add_argument("--optimize-for", choices=["winner", "spread", "total"], default=None)
    backtest.add_argument("--output-dir", default=None)
    backtest.add_argument("--no-cache", action="store_true", help="Disable persistent Odds API response caching for any auto-fetch work.")
    backtest.add_argument("--refresh-cache", action="store_true", help="Ignore cached Odds API responses and re-fetch them during auto-fetch.")
    backtest.add_argument("--timeout-seconds", type=int, default=None, help="HTTP timeout for Odds API requests during auto-fetch.")
    backtest.add_argument("--max-retries", type=int, default=None, help="Retry count for timeout / transient server errors during auto-fetch.")
    backtest.add_argument("--retry-backoff-seconds", type=float, default=None, help="Base backoff between request retries during auto-fetch.")
    backtest.add_argument("--fail-fast", action="store_true", help="Abort immediately on the first failed historical snapshot request during auto-fetch.")
    backtest.set_defaults(func=command_backtest)

    current = sub.add_parser("predict-current", help="Score the current tournament market using selected-season plus prior-season training.")
    current.add_argument("--season", type=int, default=2026)
    current.add_argument("--market-source", choices=["hybrid", "live", "prep"], default=None)
    current.add_argument("--api-key", default=None)
    current.add_argument("--bankroll", type=float, default=None)
    current.add_argument("--fractional-kelly", type=float, default=None)
    current.add_argument("--max-stake-fraction", type=float, default=None)
    current.add_argument("--min-moneyline-ev", type=float, default=0.015)
    current.add_argument("--min-spread-ev", type=float, default=0.010)
    current.add_argument("--min-total-ev", type=float, default=0.010)
    current.add_argument("--min-edge-prob", type=float, default=0.015)
    current.add_argument("--min-market-books", type=int, default=1)
    current.add_argument("--force-retrain", action="store_true")
    current.add_argument("--optimize-for", choices=["winner", "spread", "total"], default=None)
    current.add_argument("--output-dir", default=None)
    current.add_argument("--no-cache", action="store_true", help="Disable persistent live Odds API response caching for this run.")
    current.add_argument("--refresh-cache", action="store_true", help="Ignore cached live Odds API responses and re-fetch them.")
    current.add_argument("--live-cache-ttl-minutes", type=int, default=None, help="How long a cached live odds response stays reusable.")
    current.set_defaults(func=command_predict_current)

    bracket = sub.add_parser("predict-bracket", help="Simulate the full bracket from the selected-season plus prior-season model.")
    bracket.add_argument("--season", type=int, default=2026)
    bracket.add_argument("--force-retrain", action="store_true")
    bracket.add_argument("--optimize-for", choices=["winner", "spread", "total"], default=None)
    bracket.add_argument("--output-dir", default=None)
    bracket.set_defaults(func=command_predict_bracket)

    smoke = sub.add_parser("smoke-test", help="Run an offline smoke test against the bundled 2026 prep lines.")
    smoke.set_defaults(func=command_smoke_test)

    interactive = sub.add_parser("interactive", help="Open the interactive menu.")
    interactive.set_defaults(func=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    paths = build_paths()
    settings = load_settings(paths)
    parser = build_arg_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return interactive_menu(paths, settings)
    args = parser.parse_args(argv)
    if args.command == "interactive":
        return interactive_menu(paths, settings)
    return int(args.func(args, paths, settings))


if __name__ == "__main__":
    raise SystemExit(main())
