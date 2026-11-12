"""Odds conversion, expected value, Kelly sizing and payouts.

Ported unchanged from the legacy ``mm26_xgb/markets.py`` (its tests came along in
``tests/test_odds_math.py``). A sign error here corrupts P&L directly, so keep these pinned.
"""

from __future__ import annotations

import numpy as np


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_to_implied_prob(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def fair_prob_from_two_way_moneyline(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Proportional (multiplicative) de-vig of a two-way market."""
    p_a = american_to_implied_prob(odds_a)
    p_b = american_to_implied_prob(odds_b)
    total = p_a + p_b
    if total <= 0:
        return 0.5, 0.5
    return p_a / total, p_b / total


def fair_prob_to_american(prob: float) -> int:
    p = float(np.clip(prob, 1e-6, 1 - 1e-6))
    if p >= 0.5:
        odds = -(100.0 * p / (1.0 - p))
    else:
        odds = 100.0 * (1.0 - p) / p
    return round(odds)


def decimal_to_american(decimal_odds: float) -> int:
    dec = float(decimal_odds)
    if dec <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    if dec >= 2.0:
        return round((dec - 1.0) * 100.0)
    return round(-100.0 / (dec - 1.0))


def expected_value_per_unit(win_prob: float, american_odds: float) -> float:
    profit_multiple = american_to_decimal(american_odds) - 1.0
    p = float(np.clip(win_prob, 0.0, 1.0))
    return p * profit_multiple - (1.0 - p)


def break_even_prob(american_odds: float) -> float:
    return american_to_implied_prob(american_odds)


def kelly_fraction(win_prob: float, american_odds: float) -> float:
    b = american_to_decimal(american_odds) - 1.0
    p = float(np.clip(win_prob, 0.0, 1.0))
    return max(0.0, (b * p - (1.0 - p)) / b)


def american_profit(stake: float, american_odds: float) -> float:
    return float(stake) * (american_to_decimal(american_odds) - 1.0)


def compute_hold_two_way(price_a: float | None, price_b: float | None) -> float | None:
    if price_a is None or price_b is None:
        return None
    return float(american_to_implied_prob(price_a) + american_to_implied_prob(price_b) - 1.0)
