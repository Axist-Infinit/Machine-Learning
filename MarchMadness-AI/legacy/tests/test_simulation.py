"""Unit tests for the probability/score plumbing in mm26_xgb.simulation."""
import numpy as np
import pytest

from mm26_xgb.simulation import (
    cover_prob_team_a,
    enforce_non_crossing_quantiles,
    normal_cdf,
    over_prob,
    quantile_columns,
    quantile_label,
    reconcile_scores,
    robust_sigma_from_quantiles,
)


# --- normal_cdf ----------------------------------------------------------

def test_normal_cdf_at_mean_is_half():
    assert float(normal_cdf(0.0, 0.0, 1.0)) == pytest.approx(0.5)
    assert float(normal_cdf(150.0, 150.0, 12.0)) == pytest.approx(0.5)


def test_normal_cdf_known_quantile():
    # ~97.5% of mass below mean + 1.96 sigma.
    assert float(normal_cdf(1.96, 0.0, 1.0)) == pytest.approx(0.975, abs=1e-3)


def test_normal_cdf_guards_nonpositive_std():
    # std <= 0 is replaced with 1.0 instead of dividing by zero.
    val = float(normal_cdf(0.0, 0.0, 0.0))
    assert val == pytest.approx(0.5)


def test_normal_cdf_vectorized():
    out = normal_cdf(np.array([-1.0, 0.0, 1.0]), 0.0, 1.0)
    assert out.shape == (3,)
    assert out[0] < out[1] < out[2]


# --- robust_sigma_from_quantiles ----------------------------------------

def test_sigma_from_iqr():
    # 0.25/0.75 of a unit normal are +/-0.6744898; scaled by 1.34898 -> sigma 1.
    q_map = {
        0.25: np.array([-0.6744897501960817]),
        0.75: np.array([0.6744897501960817]),
    }
    sigma = robust_sigma_from_quantiles(q_map)
    assert sigma[0] == pytest.approx(1.0, abs=1e-6)


def test_sigma_clipped_to_minimum_one():
    q_map = {0.25: np.array([5.0]), 0.75: np.array([5.0])}  # zero spread
    sigma = robust_sigma_from_quantiles(q_map)
    assert sigma[0] == pytest.approx(1.0)


def test_sigma_default_when_no_pairs():
    q_map = {0.50: np.array([0.0, 0.0])}  # no usable quantile pair
    sigma = robust_sigma_from_quantiles(q_map)
    assert sigma.tolist() == [10.0, 10.0]


# --- reconcile_scores ----------------------------------------------------

def test_reconcile_basic():
    a, b, margin, total = reconcile_scores(
        np.array([0.6]), np.array([10.0]), np.array([140.0]), 0, 200
    )
    assert a[0] == 75 and b[0] == 65
    assert margin[0] == pytest.approx(10.0)
    assert total[0] == pytest.approx(140.0)


def test_reconcile_corrects_sign_for_favorite():
    # Model says TeamA wins (p=0.6) but margin came back negative -> flip sign.
    a, b, margin, _ = reconcile_scores(
        np.array([0.6]), np.array([-4.0]), np.array([140.0]), 0, 200
    )
    assert margin[0] > 0
    assert a[0] > b[0]


def test_reconcile_underdog_orientation():
    a, b, margin, _ = reconcile_scores(
        np.array([0.3]), np.array([8.0]), np.array([140.0]), 0, 200
    )
    # p < 0.5 means TeamB should be the predicted winner.
    assert b[0] > a[0]
    assert margin[0] < 0


def test_reconcile_breaks_ties():
    a, b, _, _ = reconcile_scores(
        np.array([0.6]), np.array([0.0]), np.array([140.0]), 0, 200
    )
    assert a[0] != b[0]
    assert a[0] > b[0]  # favorite gets the extra point


# --- enforce_non_crossing_quantiles -------------------------------------

def test_enforce_non_crossing_sorts_each_row():
    out = enforce_non_crossing_quantiles(np.array([[3.0, 1.0, 2.0], [9.0, 8.0, 10.0]]))
    assert out[0].tolist() == [1.0, 2.0, 3.0]
    assert out[1].tolist() == [8.0, 9.0, 10.0]


# --- cover_prob_team_a / over_prob --------------------------------------

def test_cover_prob_pickem_is_half():
    p = cover_prob_team_a(np.array([0.0]), np.array([10.0]), np.array([0.0]))
    assert p[0] == pytest.approx(0.5)


def test_cover_prob_when_model_matches_line():
    # Line has A favored by 10 (spread -10); model mean margin also 10 -> 50/50 to cover.
    p = cover_prob_team_a(np.array([10.0]), np.array([10.0]), np.array([-10.0]))
    assert p[0] == pytest.approx(0.5)


def test_cover_prob_strong_favorite_above_line():
    # Model mean margin (20) far exceeds the line (A -5) -> high cover prob.
    p = cover_prob_team_a(np.array([20.0]), np.array([10.0]), np.array([-5.0]))
    assert p[0] > 0.9


def test_over_prob_at_line_is_half():
    p = over_prob(np.array([150.0]), np.array([12.0]), np.array([150.0]))
    assert p[0] == pytest.approx(0.5)


def test_over_prob_high_when_model_above_line():
    p = over_prob(np.array([170.0]), np.array([10.0]), np.array([150.0]))
    assert p[0] > 0.9


# --- quantile label helpers ---------------------------------------------

def test_quantile_columns():
    assert quantile_columns([0.1, 0.5, 0.9], "Margin") == ["MarginQ10", "MarginQ50", "MarginQ90"]


@pytest.mark.parametrize("alpha,label", [(0.1, "Q10"), (0.25, "Q25"), (0.5, "Q50"), (0.9, "Q90")])
def test_quantile_label(alpha, label):
    assert quantile_label(alpha) == label
