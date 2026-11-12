"""Kaggle March Machine Learning Mania data, read straight from the newest competition zip.

The zip stays in the gitignored workspace (``workspace/data/kaggle_zips``); the competition
rules forbid redistributing it, so nothing derived from it in bulk is committed either.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd

ZIP_PATTERN = re.compile(r"march-machine-learning-mania-(\d{4})\.zip$")

_RESULT_COLS = ["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc", "NumOT"]
_BOX = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]

#: Files and columns the pipeline relies on, per gender prefix (M/W).
REQUIRED: dict[str, list[str]] = {
    "{g}Teams.csv": ["TeamID", "TeamName"],
    "{g}Seasons.csv": ["Season", "DayZero"],
    "{g}TeamSpellings.csv": ["TeamNameSpelling", "TeamID"],
    "{g}RegularSeasonCompactResults.csv": _RESULT_COLS,
    "{g}RegularSeasonDetailedResults.csv": _RESULT_COLS + [s + c for s in "WL" for c in _BOX],
    "{g}NCAATourneyCompactResults.csv": _RESULT_COLS,
    "{g}NCAATourneySeeds.csv": ["Season", "Seed", "TeamID"],
    "{g}NCAATourneySlots.csv": ["Season", "Slot", "StrongSeed", "WeakSeed"],
}


def find_latest_zip(directory: str | Path) -> Path:
    found = []
    for p in Path(directory).glob("*.zip"):
        m = ZIP_PATTERN.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    if not found:
        raise FileNotFoundError(
            f"No march-machine-learning-mania-YYYY.zip in {directory}. Download it from the "
            "Kaggle competition page (data tab) and place it there."
        )
    return max(found)[1]


class KaggleData:
    """Lazy, cached access to the CSVs inside one competition zip. Returned frames are shallow
    copies of a cache: add columns freely, but don't modify values in place."""

    def __init__(self, zip_path: str | Path):
        self.zip_path = Path(zip_path)
        m = ZIP_PATTERN.search(self.zip_path.name)
        self.year = int(m.group(1)) if m else None
        with zipfile.ZipFile(self.zip_path) as zf:
            self._members = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".csv")}
        self._cache: dict[str, pd.DataFrame] = {}

    @classmethod
    def latest(cls, directory: str | Path) -> KaggleData:
        return cls(find_latest_zip(directory))

    @property
    def files(self) -> list[str]:
        return sorted(self._members)

    def has(self, name: str) -> bool:
        return name in self._members

    def read(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            if name not in self._members:
                raise FileNotFoundError(f"{name} is not in {self.zip_path.name}")
            with zipfile.ZipFile(self.zip_path) as zf, zf.open(self._members[name]) as fh:
                self._cache[name] = pd.read_csv(fh)
        return self._cache[name].copy(deep=False)

    def columns(self, name: str) -> list[str]:
        with zipfile.ZipFile(self.zip_path) as zf, zf.open(self._members[name]) as fh:
            return list(pd.read_csv(fh, nrows=0).columns)

    # Convenience accessors -------------------------------------------------------------
    def teams(self, g: str) -> pd.DataFrame:
        return self.read(f"{g}Teams.csv")

    def seasons(self, g: str) -> pd.DataFrame:
        return self.read(f"{g}Seasons.csv")

    def spellings(self, g: str) -> pd.DataFrame:
        return self.read(f"{g}TeamSpellings.csv")

    def compact(self, g: str, phase: str = "RegularSeason") -> pd.DataFrame:
        """``phase``: RegularSeason (incl. conference tournaments), NCAATourney or SecondaryTourney."""
        return self.read(f"{g}{phase}CompactResults.csv")

    def detailed(self, g: str, phase: str = "RegularSeason") -> pd.DataFrame:
        return self.read(f"{g}{phase}DetailedResults.csv")

    def seeds(self, g: str) -> pd.DataFrame:
        return self.read(f"{g}NCAATourneySeeds.csv")

    def slots(self, g: str) -> pd.DataFrame:
        return self.read(f"{g}NCAATourneySlots.csv")

    def massey(self) -> pd.DataFrame:
        return self.read("MMasseyOrdinals.csv")

    def validate(self) -> list[str]:
        """Return a list of problems (missing files/columns); empty means the schema is as expected."""
        problems = []
        for g in ("M", "W"):
            for template, cols in REQUIRED.items():
                name = template.format(g=g)
                if name not in self._members:
                    problems.append(f"missing {name}")
                    continue
                missing = sorted(set(cols) - set(self.columns(name)))
                if missing:
                    problems.append(f"{name} lacks columns {missing}")
        return problems
