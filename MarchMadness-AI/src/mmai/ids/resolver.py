"""Strict team resolution: every (gender, source, key) maps to exactly one Kaggle TeamID or raises.

Sources: ``espn`` (ESPN team id) and ``odds_api`` (Odds API team name). The learned crosswalk
lives in ``workspace/store/ids/team_crosswalk.csv`` (regenerable). Manual overrides live in
``config/team_aliases.csv`` (committed) and always win.

Teams new to Division I have no Kaggle TeamID until Kaggle's next data release; they get a
provisional id (``PROVISIONAL_BASE + ESPN id``) so they can be rated, and are remapped later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROVISIONAL_BASE = 10_000_000
TABLE_COLUMNS = ["gender", "source", "key", "kaggle_id"]


class UnresolvedTeamError(KeyError):
    pass


def provisional_id(espn_id: str | int) -> int:
    return PROVISIONAL_BASE + int(espn_id)


def is_provisional(team_id: int) -> bool:
    return int(team_id) >= PROVISIONAL_BASE


@dataclass
class TeamResolver:
    table: pd.DataFrame
    _map: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._map = {
            (str(r.gender), str(r.source), str(r.key)): int(r.kaggle_id)
            for r in self.table.itertuples(index=False)
        }

    def resolve(self, gender: str, source: str, key: str | int) -> int:
        try:
            return self._map[(gender, source, str(key))]
        except KeyError:
            raise UnresolvedTeamError(
                f"[{gender}] no Kaggle TeamID for {source} key {key!r}. Rebuild the crosswalk "
                "(`mmai crosswalk`) or add a row to config/team_aliases.csv."
            ) from None

    def resolve_many(self, gender: str, source: str, keys) -> tuple[dict[str, int], list[str]]:
        found, missing = {}, []
        for k in dict.fromkeys(str(k) for k in keys):
            v = self._map.get((gender, source, k))
            if v is None:
                missing.append(k)
            else:
                found[k] = v
        return found, missing

    def keys(self, gender: str, source: str) -> list[str]:
        return [k for (g, s, k) in self._map if g == gender and s == source]

    @classmethod
    def load(cls, crosswalk_csv: str | Path | None, aliases_csv: str | Path | None = None) -> TeamResolver:
        frames = []
        for path in (crosswalk_csv, aliases_csv):  # aliases last, so they override
            if path is not None and Path(path).exists():
                df = pd.read_csv(path, dtype={"key": str, "source": str, "gender": str}, comment="#")
                if len(df):
                    frames.append(df[TABLE_COLUMNS])
        if not frames:
            return cls(pd.DataFrame(columns=TABLE_COLUMNS))
        table = pd.concat(frames, ignore_index=True).dropna(subset=["kaggle_id"])
        table = table.drop_duplicates(["gender", "source", "key"], keep="last")
        return cls(table.reset_index(drop=True))
