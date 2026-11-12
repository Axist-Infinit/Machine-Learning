"""Polite, cached JSON-over-HTTP client shared by the ESPN and Odds API sources.

- Responses are cached on disk as JSON, keyed by URL + params with secret params removed,
  so API keys never land in the cache. Historical data uses ``ttl_s=None`` (cache forever).
- Requests are spaced at least ``min_interval_s`` apart; 429/5xx/timeouts back off and retry.
- Response headers are stored lower-cased, so quota headers such as ``X-Requests-Remaining``
  are always found under ``x-requests-remaining``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; mmai/0.1; personal research)"
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class HTTPError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        super().__init__(f"HTTP {status} for {url}: {body[:200]}")
        self.status = status


@dataclass
class Fetched:
    body: Any
    headers: dict[str, str]
    status: int
    from_cache: bool


def cache_key(url: str, params: Mapping[str, Any] | None, secret_params: tuple[str, ...] = ()) -> str:
    clean = {k: v for k, v in sorted((params or {}).items()) if k not in secret_params}
    return hashlib.sha256(json.dumps([url, clean], sort_keys=True, default=str).encode()).hexdigest()


class CachedJSONClient:
    def __init__(
        self,
        cache_dir: str | os.PathLike,
        *,
        min_interval_s: float = 0.5,
        timeout_s: float = 25.0,
        max_retries: int = 4,
        session: Any = None,
        user_agent: str = DEFAULT_USER_AGENT,
        secret_params: tuple[str, ...] = ("apiKey",),
        negative_cache_status: tuple[int, ...] = (404,),
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cache_dir = Path(cache_dir)
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.user_agent = user_agent
        self.secret_params = secret_params
        self.negative_cache_status = negative_cache_status
        self._sleep = sleep
        self._clock = clock
        self._last_request = float("-inf")
        self.requests_made = 0

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def lookup(self, url: str, params: Mapping[str, Any] | None = None, *, ttl_s: float | None = None) -> Fetched | None:
        """Return the cached response if present and fresh, without touching the network."""
        path = self._path(cache_key(url, params, self.secret_params))
        if not path.exists():
            return None
        rec = json.loads(path.read_text())
        if ttl_s is not None and time.time() - rec["fetched_epoch"] > ttl_s:
            return None
        return Fetched(rec["body"], rec["headers"], rec["status"], True)

    def get(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        ttl_s: float | None = None,
        use_cache: bool = True,
    ) -> Fetched:
        """GET ``url``; serve from cache when fresh (``ttl_s=None`` means never stale)."""
        path = self._path(cache_key(url, params, self.secret_params))
        if use_cache:
            hit = self.lookup(url, params, ttl_s=ttl_s)
            if hit is not None:
                return hit
        fetched = self._fetch(url, params)
        clean = {k: v for k, v in (params or {}).items() if k not in self.secret_params}
        record = {
            "url": url,
            "params": clean,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fetched_epoch": time.time(),
            "status": fetched.status,
            "headers": fetched.headers,
            "body": fetched.body,
        }
        _atomic_write_json(path, record)
        return fetched

    def _fetch(self, url: str, params: Mapping[str, Any] | None) -> Fetched:
        attempt = 0
        while True:
            wait = self.min_interval_s - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = self._clock()
            self.requests_made += 1
            try:
                resp = self.session.get(
                    url, params=params, timeout=self.timeout_s, headers={"User-Agent": self.user_agent}
                )
            except (requests.Timeout, requests.ConnectionError):
                if attempt >= self.max_retries:
                    raise
                attempt += 1
                self._sleep(min(60.0, 2.0**attempt))
                continue
            status = resp.status_code
            if status in RETRY_STATUS and attempt < self.max_retries:
                attempt += 1
                retry_after = resp.headers.get("Retry-After", "")
                self._sleep(float(retry_after) if retry_after.isdigit() else min(60.0, 2.0**attempt))
                continue
            headers = {k.lower(): v for k, v in resp.headers.items()}
            if status in self.negative_cache_status:
                return Fetched(None, headers, status, False)
            if status >= 400:
                raise HTTPError(status, url, resp.text)
            return Fetched(resp.json(), headers, status, False)


def _atomic_write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(record, fh)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
