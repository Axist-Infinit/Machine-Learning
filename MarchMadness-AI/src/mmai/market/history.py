"""Historical closing (and opening) lines per game, built from ESPN's per-event odds.

Provider choice per game, in order:

1. an item with structured ``open``/``close`` blocks (ESPN BET / DraftKings, 2025 on): its
   ``close_*`` values are the closing line by construction;
2. ESPN's ``consensus`` item;
3. the remaining item with the best provider priority.

Pre-2025 items carry a single "final" line whose closing-line semantics still have to be
verified (plan M0: a paid Odds API sample of 2021-24 tournaments). ``line_kind`` records which
case each row is, so evaluations can be restricted to verified closers.

The home spread comes from ESPN's ``spread`` field, cross-checked against the ``details`` string
('DUKE -28.5'); if their signs disagree the details string wins and the row is flagged.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from mmai.calendar import et_date
from mmai.ids.resolver import TeamResolver

LINE_COLUMNS = [
    "event_id", "provider", "line_kind", "n_providers", "spread_home", "total", "ml_home", "ml_away",
    "open_spread_home", "open_total", "spread_sign_conflict",
]


def choose_lines(odds: pd.DataFrame) -> pd.DataFrame:
    """One line per event from ESPN odds rows (see module docstring for the rules)."""
    if odds.empty:
        return pd.DataFrame(columns=LINE_COLUMNS)
    o = odds.copy()
    has_close = o["close_spread_home"].notna() | o["close_total"].notna()
    is_consensus = o["provider"].fillna("").str.lower().eq("consensus")
    o["_rank"] = np.where(has_close, 0, np.where(is_consensus, 1, 2))
    o["_prio"] = pd.to_numeric(o["provider_priority"], errors="coerce").fillna(99)
    o["_has_close"] = has_close
    n_providers = o.groupby("event_id")["provider_id"].nunique()
    best = o.sort_values(["event_id", "_rank", "_prio", "provider"]).drop_duplicates("event_id").copy()

    raw_spread = best["close_spread_home"].where(best["_has_close"] & best["close_spread_home"].notna(),
                                                 best["spread_home"])
    details = best["spread_home_from_details"]
    conflict = details.notna() & raw_spread.notna() & (details != 0) & (np.sign(details) != np.sign(raw_spread))
    out = pd.DataFrame({
        "event_id": best["event_id"].astype(str),
        "provider": best["provider"],
        "line_kind": np.where(best["_has_close"], "close", np.where(best["_rank"] == 1, "consensus_final", "book_final")),
        "n_providers": best["event_id"].map(n_providers).astype(int),
        "spread_home": raw_spread.where(~conflict, details),
        "total": best["close_total"].where(best["_has_close"] & best["close_total"].notna(), best["total"]),
        "ml_home": best["close_ml_home"].where(best["close_ml_home"].notna(), best["ml_home"]),
        "ml_away": best["close_ml_away"].where(best["close_ml_away"].notna(), best["ml_away"]),
        "open_spread_home": best["open_spread_home"],
        "open_total": best["open_total"],
        "spread_sign_conflict": conflict.to_numpy(),
    })
    return out.reset_index(drop=True)


def build_line_table(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Completed games joined to their chosen line (games without a line are kept, with NaNs)."""
    keep = ["event_id", "season", "date_et", "start_utc", "neutral", "home_id", "away_id",
            "home_name", "away_name", "home_score", "away_score"]
    g = games.loc[games["completed"], keep].copy()
    g["event_id"] = g["event_id"].astype(str)
    return g.merge(choose_lines(odds), on="event_id", how="left")


def compare_to_reference(lines: pd.DataFrame, ref: pd.DataFrame, resolver: TeamResolver, gender: str = "M") -> dict:
    """Check ESPN lines against a reference closing table with Kaggle ``TeamAID``/``TeamBID``,
    ``CommenceTime``, ``MarketSpreadTeamA`` and ``MarketTotal`` (e.g. the cached 2025 Odds API
    closing consensus). Games are matched by Eastern date (±1 day) and the Kaggle team pair."""
    found_h, _ = resolver.resolve_many(gender, "espn", lines["home_id"])
    found_a, _ = resolver.resolve_many(gender, "espn", lines["away_id"])
    index: dict[tuple, pd.Series] = {}
    for _, r in lines.dropna(subset=["spread_home", "total"]).iterrows():
        h, a = found_h.get(str(r["home_id"])), found_a.get(str(r["away_id"]))
        if h is not None and a is not None:
            index[(str(r["date_et"]), frozenset((h, a)))] = (r, h)
    ds, dt = [], []
    for _, ref_row in ref.dropna(subset=["MarketSpreadTeamA", "MarketTotal"]).iterrows():
        d = et_date(ref_row["CommenceTime"])
        pair = frozenset((int(ref_row["TeamAID"]), int(ref_row["TeamBID"])))
        hit = next((index[k] for k in ((str(d + timedelta(days=s)), pair) for s in (0, -1, 1)) if k in index), None)
        if hit is None:
            continue
        r, home_kaggle = hit
        spread_a = r["spread_home"] if home_kaggle == int(ref_row["TeamAID"]) else -r["spread_home"]
        ds.append(spread_a - ref_row["MarketSpreadTeamA"])
        dt.append(r["total"] - ref_row["MarketTotal"])
    ds, dt = np.abs(np.array(ds, float)), np.abs(np.array(dt, float))
    return {
        "reference_games": len(ref),
        "matched": len(ds),
        "spread_mae": round(float(ds.mean()), 3) if len(ds) else None,
        "spread_within_1": round(float((ds <= 1).mean()), 3) if len(ds) else None,
        "total_mae": round(float(dt.mean()), 3) if len(dt) else None,
        "total_within_1": round(float((dt <= 1).mean()), 3) if len(dt) else None,
    }
