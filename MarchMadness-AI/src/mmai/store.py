"""Small parquet helpers: atomic writes and directory reads."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


def write_parquet(df: pd.DataFrame, path: str | os.PathLike) -> Path:
    """Write ``df`` to ``path`` atomically (temp file + rename), creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def read_parquets(root: str | os.PathLike, pattern: str = "**/*.parquet") -> pd.DataFrame:
    """Concatenate every parquet file under ``root`` matching ``pattern`` (empty frame if none)."""
    files = sorted(Path(root).glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    columns = list(dict.fromkeys(c for f in frames for c in f.columns))
    # All-NA columns (e.g. open/close line fields before 2025) must not decide the result dtype
    # (pandas deprecates that); drop them per file and restore the full column set afterwards.
    trimmed = [f.dropna(axis=1, how="all") for f in frames if len(f)]
    if not trimmed:
        return pd.DataFrame(columns=columns)
    return pd.concat(trimmed, ignore_index=True).reindex(columns=columns)
