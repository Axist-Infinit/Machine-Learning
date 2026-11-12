"""Filesystem layout. Everything generated lives under ``workspace/`` (gitignored)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def kaggle_zips(self) -> Path:
        """Kaggle competition zips. Never committed: the competition rules forbid redistribution."""
        return self.workspace / "data" / "kaggle_zips"

    @property
    def raw(self) -> Path:
        """Raw API responses (JSON), kept forever so a vanished endpoint can't take history away."""
        return self.workspace / "raw"

    @property
    def store(self) -> Path:
        """Tidy tables (parquet), always rebuildable from ``raw``."""
        return self.workspace / "store"

    @property
    def legacy_odds_cache(self) -> Path:
        """Odds API responses captured by the legacy pipeline (2025 tournament, March 2026)."""
        return self.workspace / "cache" / "odds_api"

    @property
    def config(self) -> Path:
        return self.root / "config"


def get_paths(root: str | os.PathLike | None = None) -> Paths:
    return Paths(Path(root or os.environ.get("MMAI_ROOT") or PROJECT_ROOT))
