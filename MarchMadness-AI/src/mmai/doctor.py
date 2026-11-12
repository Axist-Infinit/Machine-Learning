"""Preflight checks (``mmai doctor``). M0 scope: data, calendar, team IDs, API key, odds logger.

Later milestones add: no-NaN features for the tournament field, artifact hashes, schedule sanity.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone

import pandas as pd

from mmai.calendar import DAY_ZERO_OVERRIDES, day_zero, daynum, season_of
from mmai.config import Paths
from mmai.ids.crosswalk import unmapped_division_one
from mmai.ids.resolver import TeamResolver, is_provisional
from mmai.sources.kaggle import KaggleData
from mmai.store import read_parquets


def run_doctor(paths: Paths, today: date | None = None) -> list[tuple[str, str, str]]:
    today = today or datetime.now(timezone.utc).date()
    season = season_of(today)
    checks: list[tuple[str, str, str]] = []

    kd = None
    try:
        kd = KaggleData.latest(paths.kaggle_zips)
        problems = kd.validate()
        checks.append(("FAIL" if problems else "OK", "kaggle data",
                       kd.zip_path.name + (f": {problems[:3]}" if problems else "")))
    except FileNotFoundError as exc:
        checks.append(("FAIL", "kaggle data", str(exc)))

    try:
        seasons = kd.seasons("M") if kd else None
        dz = day_zero(season, seasons)
        detail = f"season {season}: DayZero {dz}, today is DayNum {daynum(today, season, seasons)}"
        status = "OK"
        published = seasons is not None and season in set(seasons["Season"])
        if published and season in DAY_ZERO_OVERRIDES and day_zero(season, seasons) != DAY_ZERO_OVERRIDES[season]:
            status, detail = "FAIL", detail + f" (override {DAY_ZERO_OVERRIDES[season]} disagrees with Kaggle)"
        checks.append((status, "season calendar", detail))
    except KeyError as exc:
        checks.append(("FAIL", "season calendar", str(exc)))

    resolver = TeamResolver.load(paths.store / "ids" / "team_crosswalk.csv", paths.config / "team_aliases.csv")
    for g in ("M", "W"):
        games = read_parquets(paths.store / "espn" / "games" / f"gender={g}", f"season={season}.parquet")
        if games.empty:
            checks.append(("WARN", f"espn games [{g}]", f"none for season {season}; run `mmai ingest-espn`"))
            continue
        done = games[games["completed"]]
        last = max(done["date_et"]) if len(done) else "never"
        stale = len(done) == 0 or (today - date.fromisoformat(str(last))).days > 2
        checks.append(("WARN" if stale else "OK", f"espn games [{g}]",
                       f"{len(done)} completed this season, latest {last}"))
        mapped = set(resolver.keys(g, "espn"))
        if not mapped:
            checks.append(("FAIL", f"team ids [{g}]", "no crosswalk; run `mmai crosswalk`"))
            continue
        kaggle_last = int(kd.seasons(g)["Season"].max()) if kd else season - 1
        gaps = unmapped_division_one(done, mapped, kaggle_last)
        n_prov = sum(is_provisional(resolver.resolve(g, "espn", k)) for k in mapped)
        detail = f"{len(mapped)} ESPN teams mapped ({n_prov} provisional)"
        if len(gaps):
            detail += f"; unmapped Division I teams (run `mmai crosswalk`): {list(gaps['espn_name'])[:8]}"
        checks.append(("WARN" if len(gaps) else "OK", f"team ids [{g}]", detail))

    key = os.environ.get("ODDS_API_KEY")
    checks.append(("OK" if key else "WARN", "odds api key", "set" if key else "ODDS_API_KEY not set (logger can't run)"))
    ledger = paths.store / "odds_api_ledger.csv"
    if ledger.exists():
        led = pd.read_csv(ledger)
        month = led[led["ts_utc"].astype(str).str.startswith(today.strftime("%Y-%m"))]
        used = pd.to_numeric(month["cost_reported"], errors="coerce").fillna(month["cost_estimate"]).sum()
        remaining = month["remaining"].dropna().iloc[-1] if month["remaining"].notna().any() else "?"
        checks.append(("OK", "odds api credits", f"{int(used)} used this month; {remaining} remaining per API"))

    snaps = sorted((paths.store / "odds_snapshots").glob("gender=*/date=*/*.parquet"))
    if snaps:
        newest = max(snaps, key=lambda p: p.stat().st_mtime)
        age_h = (datetime.now().timestamp() - newest.stat().st_mtime) / 3600
        checks.append(("WARN" if age_h > 26 else "OK", "odds logger", f"newest snapshot {age_h:.1f} h old"))
    else:
        checks.append(("WARN", "odds logger", "no snapshots yet"))
    return checks
