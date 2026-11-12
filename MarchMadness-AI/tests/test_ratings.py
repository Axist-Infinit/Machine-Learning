import numpy as np
import pytest

from mmai.model.ratings import RatingsConfig, SeasonRater, asof_season, recency_weights, run_ratings
from tests.synthetic import HCA, MU, make_league


@pytest.fixture(scope="module")
def league():
    return make_league(n_teams=30, seasons=(2020,), n_days=110, games_per_day=10, seed=1)


def test_recency_weights_shapes():
    age = np.array([0, 20, 40, 60, 80, 120])
    assert np.allclose(recency_weights(age, RatingsConfig()), 1.0)
    assert recency_weights([60], RatingsConfig(recency="exp", half_life=60))[0] == pytest.approx(0.5)
    tr = recency_weights(age, RatingsConfig(recency="trank"))
    assert tr[0] == tr[2] == 1.0 and tr[3] == pytest.approx(0.8) and tr[4] == tr[5] == pytest.approx(0.6)


def test_recovers_true_ratings_and_league_constants(league):
    games, teams, truth = league
    _, final = asof_season(games, RatingsConfig())
    order = {t: i for i, t in enumerate(final.teams)}
    idx = [order[t] for t in teams]
    assert np.corrcoef(final.off[idx], truth[2020]["off"])[0, 1] > 0.95
    assert np.corrcoef(final.deff[idx], truth[2020]["deff"])[0, 1] > 0.95
    assert np.corrcoef(final.tempo[idx], truth[2020]["tempo"])[0, 1] > 0.9
    assert final.hca == pytest.approx(HCA, abs=1.0)
    assert final.mu + final.off.mean() + final.deff.mean() == pytest.approx(MU, abs=1.0)


def test_late_season_predictions_track_true_margins(league):
    games, teams, truth = league
    preds = run_ratings(games, RatingsConfig())
    late = games[games["daynum"] >= 80].merge(preds, on="game_id")
    ti = {t: i for i, t in enumerate(teams)}
    t = truth[2020]
    a, b = late["team_a"].map(ti).to_numpy(), late["team_b"].map(ti).to_numpy()
    true_margin = (t["off"][a] + t["deff"][b] - t["off"][b] - t["deff"][a] + 2 * HCA * late["venue_a"]) * 0.68
    assert np.corrcoef(late["pred_margin_a"], true_margin)[0, 1] > 0.9


@pytest.mark.parametrize("cfg", [RatingsConfig(), RatingsConfig(recency="trank"), RatingsConfig(target="points")])
def test_time_travel_identical(league, cfg):
    """Deleting every game after the cut must not change any prediction up to the cut."""
    games, _, _ = league
    cut = 60
    full, _ = asof_season(games, cfg)
    trunc, _ = asof_season(games[games["daynum"] <= cut], cfg)
    m = trunc.merge(full, on="game_id", suffixes=("_t", "_f"))
    assert len(m) == len(trunc) > 0
    assert np.allclose(m["pred_pts_a_t"], m["pred_pts_a_f"], atol=1e-10, equal_nan=True)


def test_first_day_uses_only_the_prior(league):
    games, _, _ = league
    _, prior_full = asof_season(games, RatingsConfig())
    prior = prior_full.as_prior(0.6)
    rater = SeasonRater(games, RatingsConfig(), prior)
    r0 = rater.solve(0)
    assert r0.mu == prior.mu and r0.hca == prior.hca
    pos = {t: i for i, t in enumerate(prior.teams)}
    assert np.allclose(r0.off, [prior.off[pos[t]] for t in r0.teams])


def test_points_target_has_no_tempo(league):
    games, _, _ = league
    preds = run_ratings(games, RatingsConfig(target="points"))
    assert preds["pred_poss40"].isna().all() and preds["pred_total_reg"].between(100, 200).all()


def test_priors_carry_across_seasons():
    games, _, _ = make_league(n_teams=24, seasons=(2020, 2021), n_days=60, games_per_day=8, seed=3)
    with_prior = run_ratings(games, RatingsConfig(carryover=0.7))
    no_prior = run_ratings(games, RatingsConfig(carryover=0.0, prior_games=0.0))
    early = games[(games["season"] == 2021) & (games["daynum"] < 5)]
    err = {}
    for name, p in (("with", with_prior), ("without", no_prior)):
        e = early.merge(p, on="game_id")
        err[name] = np.sqrt(np.mean((e["margin_a"] - e["pred_margin_a"]) ** 2))
    assert err["with"] < err["without"]
