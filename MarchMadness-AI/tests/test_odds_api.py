from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from mmai.sources.http import Fetched
from mmai.sources.odds_api import (
    CreditBudgetExceeded,
    MissingAPIKey,
    OddsAPI,
    estimate_cost,
    flatten_odds,
)
from mmai.sources.odds_logger import log_snapshot

NOW = datetime(2026, 11, 12, 15, 0, tzinfo=timezone.utc)


def _event(eid, tip):
    return {"id": eid, "sport_key": "basketball_ncaab", "commence_time": tip.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_team": "Duke Blue Devils", "away_team": "Siena Saints",
            "bookmakers": [{"key": "draftkings", "last_update": "2026-11-12T14:59:00Z", "markets": [
                {"key": "spreads", "outcomes": [{"name": "Duke Blue Devils", "price": -110, "point": -7.5},
                                                {"name": "Siena Saints", "price": -110, "point": 7.5}]},
                {"key": "totals", "outcomes": [{"name": "Over", "price": -105, "point": 141.5},
                                               {"name": "Under", "price": -115, "point": 141.5}]}]}]}


class FakeClient:
    def __init__(self, body):
        self.body = body
        self.get_calls = []

    def lookup(self, url, params=None, ttl_s=None):
        return None

    def get(self, url, params=None, ttl_s=None, use_cache=True):
        self.get_calls.append((url, dict(params)))
        return Fetched(self.body, {"x-requests-last": "3", "x-requests-remaining": "19997", "x-requests-used": "3"},
                       200, False)


def _api(tmp_path, body, budget=20_000):
    return OddsAPI(FakeClient(body), tmp_path / "ledger.csv", api_key="TEST", monthly_budget=budget, now=lambda: NOW)


def test_estimate_cost():
    assert estimate_cost(["h2h", "spreads", "totals"]) == 3
    assert estimate_cost(["h2h", "spreads", "totals"], historical=True) == 30
    assert estimate_cost(["spreads"], bookmakers=[f"b{i}" for i in range(12)]) == 2


def test_live_odds_drop_in_play_events_and_log_reported_cost(tmp_path):
    body = [_event("pre", NOW + timedelta(hours=3)), _event("live", NOW - timedelta(minutes=20))]
    api = _api(tmp_path, body)
    events, snap = api.odds("M")
    assert [e["id"] for e in events] == ["pre"] and snap == NOW
    _, params = api.client.get_calls[0]
    assert params["commenceTimeFrom"] == "2026-11-12T15:00:00Z" and params["regions"] == "us"
    led = api.ledger()
    assert int(led["cost_reported"].iloc[0]) == 3 and int(led["remaining"].iloc[0]) == 19997
    assert api.spent_this_month() == 3


def test_budget_is_checked_before_the_request_is_sent(tmp_path):
    api = _api(tmp_path, [], budget=2)
    with pytest.raises(CreditBudgetExceeded):
        api.odds("M")
    assert api.client.get_calls == []


def test_missing_key_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    api = OddsAPI(FakeClient([]), tmp_path / "ledger.csv", now=lambda: NOW)
    with pytest.raises(MissingAPIKey, match="ODDS_API_KEY"):
        api.odds("M")


def test_bookmakers_replace_regions(tmp_path):
    api = _api(tmp_path, [])
    api.odds("W", bookmakers=["pinnacle", "draftkings"])
    url, params = api.client.get_calls[0]
    assert "basketball_wncaab" in url and params["bookmakers"] == "pinnacle,draftkings" and "regions" not in params


def test_flatten_odds_one_row_per_outcome():
    df = flatten_odds([_event("e1", NOW + timedelta(hours=3))], snapshot_utc=NOW, gender="M")
    assert len(df) == 4 and set(df["market"]) == {"spreads", "totals"}
    assert df.loc[df["outcome"] == "Over", "point"].iloc[0] == 141.5


def test_log_snapshot_writes_partitioned_parquet(tmp_path):
    api = _api(tmp_path, [_event("e1", NOW + timedelta(hours=3))])
    (path,) = log_snapshot(api, tmp_path / "store", ["M"])
    assert "gender=M" in str(path) and "date=2026-11-12" in str(path)
    assert len(pd.read_parquet(path)) == 4
