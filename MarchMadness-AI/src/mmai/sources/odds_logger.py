"""Snapshot current pre-game lines into an append-only parquet archive.

Layout: ``store/odds_snapshots/gender=M/date=YYYY-MM-DD/HHMMSS.parquet`` (UTC). The closing line
for a game is later taken as the last snapshot strictly before its tip.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mmai.sources.odds_api import OddsAPI, flatten_odds
from mmai.store import write_parquet


def log_snapshot(
    api: OddsAPI,
    store_root: str | Path,
    genders: Sequence[str] = ("M", "W"),
    *,
    bookmakers: Sequence[str] | None = None,
) -> list[Path]:
    paths = []
    for g in genders:
        events, snap = api.odds(g, bookmakers=bookmakers)
        df = flatten_odds(events, snapshot_utc=snap, gender=g)
        path = Path(store_root) / "odds_snapshots" / f"gender={g}" / f"date={snap:%Y-%m-%d}" / f"{snap:%H%M%S}.parquet"
        paths.append(write_parquet(df, path))
    return paths
