"""Historical closing (and opening) lines per game, built from ESPN's per-event odds.

Only pre-game market feeds count (``market_odds``): ESPN's in-play feeds ("... - Live Odds"),
the 2017 "Opening" feed and Sportradar's internal feed are dropped. Each game then takes the
first tier that has a spread:

1. ``close``: the median, across books, of structured ``close`` lines (2023-24 for some books,
   every book from 2025). Close equals the book's final line ~98.5% of the time in 2024-26;
2. ``consensus_final``: ESPN's ``consensus`` item (2017-23);
3. ``book_final``: the median of the books' final lines.

A missing total falls through the tiers the same way. Pre-2025 final lines are what the books
showed last; whether that is the true close is checked separately (plan M0: a paid Odds API
sample of 2021-24 tournaments), so evaluations can restrict on ``line_kind``.

Each book's home spread comes from ESPN's ``spread`` field, cross-checked against its
``details`` string ('DUKE -28.5'); if the signs disagree the details string wins and the game is
flagged. Moneylines are medianed in decimal odds, which (unlike American odds) are continuous
across -100/+100.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from mmai.calendar import et_date
from mmai.ids.resolver import TeamResolver
from mmai.market.odds_math import american_to_decimal, decimal_to_american
from mmai.sources.espn import MAX_ABS_SPREAD, TOTAL_RANGE

LINE_COLUMNS = [
    "event_id", "provider", "line_kind", "n_providers", "spread_home", "total", "ml_home", "ml_away",
    "open_spread_home", "open_total", "spread_sign_conflict",
]
CONSENSUS_PROVIDER_ID = "1004"
# "Opening" (2017) is the opening line by definition; Sportradar's internal feed is 6 rows of
# unknown meaning. accuscore, teamrankings and numberfire relay market lines (median gap 0.0
# from ESPN's consensus, all on the half-point grid) and are kept.
EXCLUDED_PROVIDER_IDS = frozenset({"23", "37"})
_PRICE_COLUMNS = ("ml_home", "ml_away")


def market_odds(odds: pd.DataFrame) -> pd.DataFrame:
    """The rows of ESPN odds that are pre-game market lines."""
    live = odds["provider"].fillna("").str.contains("live", case=False)
    return odds[~live & ~odds["provider_id"].astype(str).isin(EXCLUDED_PROVIDER_IDS)]


def _median_price(s: pd.Series) -> float:
    dec = [american_to_decimal(float(x)) for x in s.dropna() if abs(float(x)) >= 100]  # else not a price
    return float(decimal_to_american(float(np.median(dec)))) if dec else np.nan


def _tier(rows: pd.DataFrame, spread: str, total: str, ml_home: str, ml_away: str) -> pd.DataFrame:
    """Per-event medians across the books in ``rows``."""
    if rows.empty:
        return pd.DataFrame(columns=["spread_home", "total", *_PRICE_COLUMNS, "provider"])
    g = rows.groupby("event_id")
    n = g["provider_id"].nunique()
    first = g["provider"].first()
    return pd.DataFrame({
        "spread_home": g[spread].median(),
        "total": g[total].median(),
        "ml_home": g[ml_home].agg(_median_price),
        "ml_away": g[ml_away].agg(_median_price),
        "provider": first.where(n == 1, "median of " + n.astype(str) + " books"),
    })


def choose_lines(odds: pd.DataFrame) -> pd.DataFrame:
    """One line per event (see the module docstring for the rules)."""
    o = market_odds(odds).copy() if len(odds) else odds
    if o.empty:
        return pd.DataFrame(columns=LINE_COLUMNS)
    o["event_id"] = o["event_id"].astype(str)
    o["provider_id"] = o["provider_id"].astype(str)
    numeric = ["spread_home", "spread_home_from_details", "total", "ml_home", "ml_away", "open_spread_home",
               "open_total", "close_spread_home", "close_total", "close_ml_home", "close_ml_away"]
    o[numeric] = o[numeric].apply(pd.to_numeric, errors="coerce").astype(float)
    # Same bounds the parser applies, for rows stored before it did (2023 close blocks hold prices).
    for col in ("spread_home", "spread_home_from_details", "open_spread_home", "close_spread_home"):
        o[col] = o[col].where(o[col].abs() <= MAX_ABS_SPREAD)
    for col in ("total", "open_total", "close_total"):
        o[col] = o[col].where(o[col].between(*TOTAL_RANGE))
    det, raw = o["spread_home_from_details"], o["spread_home"]
    o["_conflict"] = (det.notna() & raw.notna() & (det != 0) & (np.sign(det) != np.sign(raw))).astype(bool)
    o["_final_spread"] = raw.where(~o["_conflict"], det)
    is_cons = o["provider_id"].eq(CONSENSUS_PROVIDER_ID) | o["provider"].fillna("").str.lower().eq("consensus")
    books = o[~is_cons]
    has_close = books["close_spread_home"].notna() | books["close_total"].notna()
    tiers = [
        ("close", _tier(books[has_close], "close_spread_home", "close_total", "close_ml_home", "close_ml_away")),
        ("consensus_final", _tier(o[is_cons].drop_duplicates("event_id"), "_final_spread", "total", "ml_home", "ml_away")),
        ("book_final", _tier(books, "_final_spread", "total", "ml_home", "ml_away")),
    ]
    idx = pd.Index(sorted(o["event_id"].unique()), name="event_id")
    out = pd.DataFrame(index=idx, columns=["line_kind", "provider", "spread_home", "total", *_PRICE_COLUMNS])
    out[["spread_home", "total", *_PRICE_COLUMNS]] = np.nan
    for kind, t in tiers:
        t = t.reindex(idx)
        take = out["spread_home"].isna() & t["spread_home"].notna()
        out.loc[take, "line_kind"] = kind
        out.loc[take, "provider"] = t.loc[take, "provider"]
        for col in ("spread_home", "total", *_PRICE_COLUMNS):
            out[col] = out[col].astype(float).fillna(t[col].astype(float))
    g = o.groupby("event_id")
    out["n_providers"] = g["provider_id"].nunique().reindex(idx).astype(int)
    opens = books.groupby("event_id")[["open_spread_home", "open_total"]].median().reindex(idx)
    out["open_spread_home"], out["open_total"] = opens["open_spread_home"], opens["open_total"]
    out["spread_sign_conflict"] = g["_conflict"].any().reindex(idx).astype(bool)
    return out.reset_index()[LINE_COLUMNS]


def build_line_table(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Completed games joined to their chosen line (games without a line are kept, with NaNs)."""
    keep = ["event_id", "season", "date_et", "start_utc", "neutral", "home_id", "away_id",
            "home_name", "away_name", "home_score", "away_score"]
    g = games.loc[games["completed"], keep].copy()
    g["event_id"] = g["event_id"].astype(str)
    return g.merge(choose_lines(odds), on="event_id", how="left")


def attach_lines(games: pd.DataFrame, lines: pd.DataFrame, resolver: TeamResolver, gender: str) -> pd.DataFrame:
    """Closing lines for Kaggle-oriented games (``game_id, date, team_a, team_b``), matched by
    date (±1 day) and Kaggle team pair. Returns game_id, line_margin_a (= -spread for team A),
    line_total, line_kind."""
    found_h, _ = resolver.resolve_many(gender, "espn", lines["home_id"])
    found_a, _ = resolver.resolve_many(gender, "espn", lines["away_id"])
    index: dict[tuple, tuple] = {}
    for r in lines.dropna(subset=["spread_home"]).itertuples(index=False):
        h, a = found_h.get(str(r.home_id)), found_a.get(str(r.away_id))
        if h is not None and a is not None:
            index[(str(r.date_et), frozenset((h, a)))] = (h, r.spread_home, r.total, r.line_kind)
    out = []
    for g in games.itertuples(index=False):
        pair = frozenset((int(g.team_a), int(g.team_b)))
        for shift in (0, -1, 1):
            hit = index.get((str(g.date + timedelta(days=shift)), pair))
            if hit is not None:
                home, spread_home, total, kind = hit
                spread_a = spread_home if home == int(g.team_a) else -spread_home
                out.append((g.game_id, -spread_a, total, kind))
                break
    return pd.DataFrame(out, columns=["game_id", "line_margin_a", "line_total", "line_kind"])


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
