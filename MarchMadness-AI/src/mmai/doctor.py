"""Preflight checks (``mmai doctor``). M0 scope: data, calendar, team IDs, API key, odds logger.

Later milestones add: no-NaN features for the tournament field, artifact hashes, schedule sanity.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone

import pandas as pd

from mmai.calendar import DAY_ZERO_OVERRIDES, day_zero, daynum, season_of
from mmai.config import Paths
from mmai.ids.resolver import TeamResolver
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
        # Division I teams this season = teams with >= 3 games against crosswalked teams.
        mapped = set(resolver.keys(g, "espn"))
        counts: dict[str, int] = {}
        for h, a in zip(done["home_id"].astype(str), done["away_id"].astype(str)):
            if a in mapped:
                counts[h] = counts.get(h, 0) + 1
            if h in mapped:
                counts[a] = counts.get(a, 0) + 1
        names = dict(zip(done["home_id"].astype(str), done["home_name"])) | dict(
            zip(done["away_id"].astype(str), done["away_name"]))
        missing = sorted(t for t, n in counts.items() if n >= 3 and t not in mapped)
        if not mapped:
            checks.append(("FAIL", f"team ids [{g}]", "no crosswalk; run `mmai crosswalk`"))
        else:
            checks.append(("WARN" if missing else "OK", f"team ids [{g}]",
                           f"{len(mapped)} ESPN teams mapped" + (
                               f"; unmapped D1-looking teams: {[names.get(t, t) for t in missing][:8]}" if missing else "")))

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
