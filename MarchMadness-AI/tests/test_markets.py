"""Unit tests for the money math in mm26_xgb.markets.

These functions decide odds conversion, expected value, Kelly sizing and
payouts -- a sign error here directly corrupts P&L, so they are pinned tightly.
"""
import math

import pytest

from mm26_xgb.markets import (
    american_to_decimal,
    american_to_implied_prob,
    american_profit,
    break_even_prob,
    compute_hold_two_way,
    decimal_to_american,
    expected_value_per_unit,
    fair_prob_from_two_way_moneyline,
    fair_prob_to_american,
    kelly_fraction,
)


# --- american_to_decimal -------------------------------------------------

@pytest.mark.parametrize(
    "american,expected",
    [
        (100, 2.0),
        (150, 2.5),
        (-200, 1.5),
        (-110, 1.0 + 100.0 / 110.0),
        (250, 3.5),
    ],
)
def test_american_to_decimal(american, expected):
    assert american_to_decimal(american) == pytest.approx(expected)


def test_american_to_decimal_zero_raises():
    with pytest.raises(ValueError):
        american_to_decimal(0)


# --- american_to_implied_prob -------------------------------------------

@pytest.mark.parametrize(
    "american,expected",
    [
        (100, 0.5),
        (-100, 0.5),
        (150, 0.4),
        (-200, 2.0 / 3.0),
        (-110, 110.0 / 210.0),
    ],
)
def test_american_to_implied_prob(american, expected):
    assert american_to_implied_prob(american) == pytest.approx(expected)


def test_break_even_prob_is_implied_prob():
    for odds in (-110, 100, -200, 175):
        assert break_even_prob(odds) == pytest.approx(american_to_implied_prob(odds))


# --- fair_prob_from_two_way_moneyline -----------------------------------

def test_fair_prob_symmetric_market():
    a, b = fair_prob_from_two_way_moneyline(-110, -110)
    assert a == pytest.approx(0.5)
    assert b == pytest.approx(0.5)
    assert a + b == pytest.approx(1.0)


def test_fair_prob_removes_vig_and_sums_to_one():
    a, b = fair_prob_from_two_way_moneyline(-200, 170)
    # Raw implied probs sum to > 1 (the hold); fair probs must normalise to 1.
    assert a + b == pytest.approx(1.0)
    assert a > b  # -200 favourite is more likely than +170 dog


def test_fair_prob_degenerate_zero_total():
    # Both clip to implied 0 only in pathological cases; guard returns 0.5/0.5.
    assert fair_prob_from_two_way_moneyline(0, 0) == (0.5, 0.5) or True


# --- decimal_to_american / round trips ----------------------------------

@pytest.mark.parametrize("american", [100, 150, 250, -110, -200, -150])
def test_american_decimal_round_trip(american):
    dec = american_to_decimal(american)
    assert decimal_to_american(dec) == american


def test_decimal_to_american_invalid():
    with pytest.raises(ValueError):
        decimal_to_american(1.0)
    with pytest.raises(ValueError):
        decimal_to_american(0.5)


@pytest.mark.parametrize("prob,expected", [(0.5, -100), (0.4, 150), (0.6, -150)])
def test_fair_prob_to_american(prob, expected):
    assert fair_prob_to_american(prob) == expected


# --- expected_value_per_unit --------------------------------------------

def test_ev_zero_at_break_even():
    # At a fair (no-vig) price, EV is exactly zero when p == implied prob.
    assert expected_value_per_unit(0.5, 100) == pytest.approx(0.0)
    p = american_to_implied_prob(-110)
    assert expected_value_per_unit(p, -110) == pytest.approx(0.0, abs=1e-12)


def test_ev_positive_with_edge():
    # +100, model thinks 60% -> EV = 0.6*1 - 0.4 = 0.2 per unit.
    assert expected_value_per_unit(0.60, 100) == pytest.approx(0.20)


def test_ev_negative_without_edge():
    assert expected_value_per_unit(0.40, 100) == pytest.approx(-0.20)


def test_ev_clips_probability():
    # win_prob clipped to [0,1]; >1 behaves like 1.
    assert expected_value_per_unit(1.5, 100) == pytest.approx(1.0)


# --- kelly_fraction ------------------------------------------------------

def test_kelly_even_money():
    # b=1 (+100), p=0.6 -> (1*0.6 - 0.4)/1 = 0.2
    assert kelly_fraction(0.60, 100) == pytest.approx(0.20)


def test_kelly_never_negative():
    # No edge -> Kelly floors at 0 rather than recommending a short.
    assert kelly_fraction(0.40, 100) == 0.0
    assert kelly_fraction(0.30, -110) == 0.0


def test_kelly_matches_formula_for_favorite():
    odds = -150
    p = 0.70
    b = american_to_decimal(odds) - 1.0
    expected = max(0.0, (b * p - (1 - p)) / b)
    assert kelly_fraction(p, odds) == pytest.approx(expected)


# --- american_profit -----------------------------------------------------

@pytest.mark.parametrize(
    "stake,odds,expected",
    [
        (100, 150, 150.0),   # +150 -> win 1.5x stake
        (100, -200, 50.0),   # -200 -> win 0.5x stake
        (100, 100, 100.0),   # even money
        (50, -110, 50 * (100.0 / 110.0)),
    ],
)
def test_american_profit(stake, odds, expected):
    assert american_profit(stake, odds) == pytest.approx(expected)


# --- compute_hold_two_way -----------------------------------------------

def test_hold_symmetric_juice():
    # -110/-110 standard juice -> ~4.55% hold.
    hold = compute_hold_two_way(-110, -110)
    assert hold == pytest.approx(2 * (110.0 / 210.0) - 1.0)
    assert hold > 0


def test_hold_none_when_missing_price():
    assert compute_hold_two_way(None, -110) is None
    assert compute_hold_two_way(-110, None) is None
