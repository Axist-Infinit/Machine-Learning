"""The Odds API v4 (https://the-odds-api.com/liveapi/guides/v4/).

Credit costs: live odds = markets x regions; historical odds = 10 x markets x regions;
scores = 1 (2 with ``daysFrom``). Up to 10 bookmakers passed via ``bookmakers=`` count as one
region. Every billable call is checked against a monthly budget *before* it is sent and
written to a ledger afterwards. Live pulls use ``commenceTimeFrom=now`` and drop any event
that has already tipped, so in-play lines are never stored as pre-game lines.
"""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mmai.calendar import to_utc
from mmai.sources.http import CachedJSONClient, Fetched

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEYS = {"M": "basketball_ncaab", "W": "basketball_wncaab"}
FEATURED_MARKETS = ("h2h", "spreads", "totals")
LEDGER_FIELDS = ["ts_utc", "endpoint", "cost_estimate", "cost_reported", "remaining", "used"]


class MissingAPIKey(RuntimeError):
    pass


class CreditBudgetExceeded(RuntimeError):
    pass


def estimate_cost(
    markets: Sequence[str],
    *,
    regions: Sequence[str] = ("us",),
    bookmakers: Sequence[str] | None = None,
    historical: bool = False,
) -> int:
    n_regions = max(1, math.ceil(len(bookmakers) / 10)) if bookmakers else max(1, len(regions))
    return (10 if historical else 1) * len(markets) * n_regions


def iso_z(ts: datetime) -> str:
    return to_utc(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class OddsAPI:
    def __init__(
        self,
        client: CachedJSONClient,
        ledger_path: str | os.PathLike,
        *,
        api_key: str | None = None,
        monthly_budget: int = 20_000,
        now: Callable[[], datetime] = _now_utc,
    ):
        self.client = client
        self.ledger_path = Path(ledger_path)
        self._api_key = api_key
        self.monthly_budget = monthly_budget
        self.now = now

    @property
    def api_key(self) -> str:
        key = self._api_key or os.environ.get("ODDS_API_KEY")
        if not key:
            raise MissingAPIKey(
                "ODDS_API_KEY is not set. Get a key at https://the-odds-api.com, then "
                "`export ODDS_API_KEY=...` (e.g. in ~/.bashrc)."
            )
        return key

    # --- ledger ----------------------------------------------------------------------------
    def ledger(self) -> pd.DataFrame:
        if not self.ledger_path.exists():
            return pd.DataFrame(columns=LEDGER_FIELDS)
        return pd.read_csv(self.ledger_path)

    def spent_this_month(self) -> int:
        led = self.ledger()
        if led.empty:
            return 0
        rows = led[led["ts_utc"].astype(str).str.startswith(self.now().strftime("%Y-%m"))]
        cost = pd.to_numeric(rows["cost_reported"], errors="coerce").fillna(rows["cost_estimate"])
        return int(cost.sum())

    def _log(self, endpoint: str, cost_estimate: int, headers: dict[str, str]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.ledger_path.exists()
        with self.ledger_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
            if new:
                w.writeheader()
            w.writerow({
                "ts_utc": iso_z(self.now()),
                "endpoint": endpoint,
                "cost_estimate": cost_estimate,
                "cost_reported": headers.get("x-requests-last", ""),
                "remaining": headers.get("x-requests-remaining", ""),
                "used": headers.get("x-requests-used", ""),
            })

    def _billable(self, endpoint: str, params: dict[str, Any], cost: int, *, ttl_s: float | None) -> Fetched:
        url = BASE_URL + endpoint
        hit = self.client.lookup(url, params, ttl_s=ttl_s) if ttl_s is None or ttl_s > 0 else None
        if hit is not None:
            return hit
        spent = self.spent_this_month()
        if spent + cost > self.monthly_budget:
            raise CreditBudgetExceeded(
                f"{endpoint} would cost ~{cost} credits; {spent} of the {self.monthly_budget} monthly "
                "budget already used. Raise --budget deliberately if this is intended."
            )
        fetched = self.client.get(url, {**params, "apiKey": self.api_key}, use_cache=False)
        self._log(endpoint, cost, fetched.headers)
        return fetched

    # --- endpoints ---------------------------------------------------------------------------
    def odds(
        self,
        gender: str,
        *,
        markets: Sequence[str] = FEATURED_MARKETS,
        regions: Sequence[str] = ("us",),
        bookmakers: Sequence[str] | None = None,
    ) -> tuple[list[dict], datetime]:
        """Current pre-game odds. Returns (events, snapshot time)."""
        now = self.now()
        params: dict[str, Any] = {
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "dateFormat": "iso",
            "commenceTimeFrom": iso_z(now),
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        else:
            params["regions"] = ",".join(regions)
        cost = estimate_cost(markets, regions=regions, bookmakers=bookmakers)
        fetched = self._billable(f"/sports/{SPORT_KEYS[gender]}/odds", params, cost, ttl_s=0)
        events = [e for e in fetched.body or [] if to_utc(e["commence_time"]) > now]
        return events, now

    def historical_odds(
        self,
        gender: str,
        at: datetime,
        *,
        markets: Sequence[str] = FEATURED_MARKETS,
        regions: Sequence[str] = ("us",),
        bookmakers: Sequence[str] | None = None,
    ) -> dict:
        """The odds board as of ``at`` (paid plans only). Cached forever."""
        params: dict[str, Any] = {
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "dateFormat": "iso",
            "date": iso_z(at),
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        else:
            params["regions"] = ",".join(regions)
        cost = estimate_cost(markets, regions=regions, bookmakers=bookmakers, historical=True)
        return self._billable(f"/historical/sports/{SPORT_KEYS[gender]}/odds", params, cost, ttl_s=None).body or {}

    def scores(self, gender: str, *, days_from: int = 3) -> list[dict]:
        params = {"daysFrom": days_from, "dateFormat": "iso"}
        cost = 2 if days_from else 1
        return self._billable(f"/sports/{SPORT_KEYS[gender]}/scores", params, cost, ttl_s=600).body or []


ODDS_COLUMNS = [
    "snapshot_utc", "gender", "event_id", "commence_utc", "home_team", "away_team",
    "book", "book_last_update", "market", "outcome", "price", "point",
]


def flatten_odds(events: list[dict], *, snapshot_utc: datetime, gender: str) -> pd.DataFrame:
    """One row per (event, book, market, outcome)."""
    rows = []
    for e in events:
        for b in e.get("bookmakers") or []:
            for m in b.get("markets") or []:
                for o in m.get("outcomes") or []:
                    rows.append({
                        "snapshot_utc": to_utc(snapshot_utc),
                        "gender": gender,
                        "event_id": e.get("id"),
                        "commence_utc": to_utc(e["commence_time"]),
                        "home_team": e.get("home_team"),
                        "away_team": e.get("away_team"),
                        "book": b.get("key"),
                        "book_last_update": b.get("last_update"),
                        "market": m.get("key"),
                        "outcome": o.get("name"),
                        "price": o.get("price"),
                        "point": o.get("point"),
                    })
    return pd.DataFrame(rows, columns=ODDS_COLUMNS)
