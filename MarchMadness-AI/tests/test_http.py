import json

import pytest

from mmai.sources import http
from mmai.sources.http import CachedJSONClient, HTTPError, cache_key


class FakeResp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body if body is not None else {"ok": True}
        self.headers = headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


def make_client(tmp_path, responses, **kw):
    session = FakeSession(responses)
    client = CachedJSONClient(tmp_path, session=session, min_interval_s=0, sleep=lambda s: None, **kw)
    return client, session


def test_cache_key_ignores_secret_params():
    assert cache_key("u", {"a": 1, "apiKey": "X"}, ("apiKey",)) == cache_key("u", {"a": 1}, ("apiKey",))


def test_second_get_is_served_from_cache_and_key_never_stored(tmp_path):
    client, session = make_client(tmp_path, [FakeResp(body={"n": 1})])
    first = client.get("https://x/y", {"q": 1, "apiKey": "SECRET"})
    second = client.get("https://x/y", {"q": 1, "apiKey": "SECRET"})
    assert first.body == second.body == {"n": 1}
    assert not first.from_cache and second.from_cache
    assert len(session.calls) == 1
    stored = "".join(p.read_text() for p in tmp_path.rglob("*.json"))
    assert "SECRET" not in stored


def test_headers_are_lower_cased(tmp_path):
    client, _ = make_client(tmp_path, [FakeResp(headers={"X-Requests-Remaining": "123"})])
    assert client.get("https://x/y").headers["x-requests-remaining"] == "123"


def test_retries_429_then_succeeds(tmp_path):
    client, session = make_client(tmp_path, [FakeResp(429, {"e": 1}), FakeResp(200, {"n": 2})])
    assert client.get("https://x/y").body == {"n": 2}
    assert len(session.calls) == 2


def test_404_is_negative_cached(tmp_path):
    client, session = make_client(tmp_path, [FakeResp(404, {"e": "missing"})])
    assert client.get("https://x/y").body is None
    assert client.get("https://x/y").body is None
    assert len(session.calls) == 1


def test_other_4xx_raises(tmp_path):
    client, _ = make_client(tmp_path, [FakeResp(401, {"e": "bad key"})])
    with pytest.raises(HTTPError):
        client.get("https://x/y")


def test_ttl_expiry_refetches(tmp_path, monkeypatch):
    client, session = make_client(tmp_path, [FakeResp(body={"v": 1}), FakeResp(body={"v": 2})])
    monkeypatch.setattr(http.time, "time", lambda: 1000.0)
    client.get("https://x/y", ttl_s=500)
    monkeypatch.setattr(http.time, "time", lambda: 1400.0)
    assert client.get("https://x/y", ttl_s=500).body == {"v": 1}  # still fresh
    monkeypatch.setattr(http.time, "time", lambda: 2000.0)
    assert client.get("https://x/y", ttl_s=500).body == {"v": 2}  # stale -> refetched
    assert len(session.calls) == 2
