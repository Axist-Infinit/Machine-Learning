import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.stats import norm

from mmai.eval import metrics as M
from mmai.model.baselines import elo, massey_consensus, raw_net_rating, seed_numbers
from tests.synthetic import make_league


@pytest.fixture(scope="module")
def league():
    return make_league(n_teams=20, seasons=(2020, 2021), n_days=50, games_per_day=6, seed=2)[0]


def _time_travel(fn, games, cut, cols):
    full = fn(games)
    trunc = fn(games[(games["season"] < 2021) | (games["daynum"] <= cut)])
    m = trunc.merge(full, on="game_id", suffixes=("_t", "_f"))
    for c in cols:
        assert np.allclose(m[f"{c}_t"], m[f"{c}_f"], equal_nan=True)


def test_raw_net_rating_is_as_of(league):
    _time_travel(raw_net_rating, league, 20, ["raw_pts_a", "raw_pts_b"])
    first_day = raw_net_rating(league).merge(league, on="game_id")
    d0 = first_day[(first_day["season"] == 2020) & (first_day["daynum"] == 0)]
    assert d0["raw_poss40"].eq(68.0).all()  # nothing known yet -> league defaults


def test_elo_is_as_of_and_rewards_winners(league):
    _time_travel(elo, league, 20, ["elo_diff_a"])
    e = elo(league).merge(league, on="game_id")
    first = e.iloc[0]
    assert first["elo_diff_a"] == pytest.approx(65.0 * first["venue_a"])


def test_massey_uses_latest_ranking_on_or_before_the_day():
    massey = pd.DataFrame({
        "Season": [2021] * 4, "SystemName": ["POM"] * 4, "RankingDayNum": [10, 10, 20, 20],
        "TeamID": [1, 2, 1, 2], "OrdinalRank": [1, 2, 2, 1]})
    games = pd.DataFrame({"game_id": ["g15", "g20"], "season": [2021, 2021], "daynum": [15, 20],
                          "team_a": [1, 1], "team_b": [2, 2]})
    out = massey_consensus(massey, games).set_index("game_id")["massey_diff_a"]
    assert out["g15"] > 0 and out["g20"] < 0  # day-20 ranking applies on day 20, not before


def test_seed_numbers():
    s = seed_numbers(pd.DataFrame({"Season": [2025, 2025], "TeamID": [1, 2], "Seed": ["W01", "X16a"]}))
    assert list(s["seed"]) == [1, 16]


def test_point_metrics():
    e = np.array([1.0, -1.0, 3.0])
    assert M.rmse(e) == pytest.approx(np.sqrt(11 / 3)) and M.mae(e) == pytest.approx(5 / 3) and M.bias(e) == pytest.approx(1.0)


def test_crps_normal_matches_numeric_integral():
    y, mu, s = 3.0, 1.0, 2.0
    numeric = integrate.quad(lambda x: (norm.cdf(x, mu, s) - (x >= y)) ** 2, -60, 60, points=[y], limit=200)[0]
    assert M.crps_normal([y], [mu], [s]) == pytest.approx(numeric, rel=1e-4)


def test_coverage_of_a_well_specified_normal():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 20000)
    assert M.coverage(y, 0.0, 1.0, 0.8) == pytest.approx(0.8, abs=0.01)


def test_paired_bootstrap_detects_a_better_model():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 10, 3000)
    d, lo, _hi = M.paired_bootstrap(M.rmse, noise + rng.normal(0, 3, 3000), noise, n=300)
    assert d > 0 and lo > 0
