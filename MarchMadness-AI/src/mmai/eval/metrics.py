"""Scoring rules and a paired bootstrap."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def rmse(err) -> float:
    e = np.asarray(err, float)
    return float(np.sqrt(np.mean(e**2)))


def mae(err) -> float:
    return float(np.mean(np.abs(np.asarray(err, float))))


def bias(err) -> float:
    return float(np.mean(np.asarray(err, float)))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss(p, y, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def crps_normal(y, mu, sigma) -> float:
    """Mean CRPS of Normal(mu, sigma) forecasts (closed form)."""
    y, mu, sigma = (np.asarray(v, float) for v in (y, mu, sigma))
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def coverage(y, mu, sigma, level: float) -> float:
    """Share of outcomes inside the central ``level`` interval of Normal(mu, sigma)."""
    half = norm.ppf(0.5 + level / 2.0) * np.asarray(sigma, float)
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(mu, float)) <= half))


def paired_bootstrap(stat, a, b, *, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """stat(a) - stat(b) and its 95% CI, resampling the same games for both."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    point = stat(a) - stat(b)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = np.array([stat(a[i]) - stat(b[i]) for i in idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)
