import numpy as np
import yaml

from mmai.eval import PREREG_PATH
from mmai.eval.walkforward import (
    assemble,
    choose_shipped,
    gate_g1,
    per_season,
    score,
    time_travel_check,
    walk_forward,
)
from mmai.model.ratings import RatingsConfig
from tests.synthetic import make_league


def test_end_to_end_on_a_synthetic_league():
    games, _, _ = make_league(n_teams=30, seasons=(2019, 2020, 2021, 2022), n_days=110, games_per_day=10, seed=4)
    cfg = RatingsConfig()
    df = assemble(games, cfg)
    test, maps = walk_forward(df, [2021, 2022])
    assert set(test["season"]) == {2021, 2022} and set(maps) == {2021, 2022}
    # mappings for 2022 may use 2021 but never 2022 itself
    assert maps[2021]["sigma_cal"] != maps[2022]["sigma_cal"]
    prereg = yaml.safe_load(PREREG_PATH.read_text())
    test, g2, shipped = choose_shipped(test, prereg)
    assert shipped in ("engine_ctx", "engine_cal") and g2[-1]["check"].endswith("ships")
    pooled = score(test)
    eng = pooled[(pooled["set"] == "after_jan15") & (pooled["model"] == "engine")].iloc[0]
    raw = pooled[(pooled["set"] == "after_jan15") & (pooled["model"] == "raw_net")].iloc[0]
    assert eng["margin_rmse"] < raw["margin_rmse"]  # opponent adjustment helps
    assert abs(eng["cov80"] - 0.8) < 0.05
    tt = time_travel_check(games, cfg, season=2022, cut_day=60)
    assert tt < 1e-9
    gates = gate_g1(test, prereg, tt)
    assert gates[0]["passed"] and any("closing line" in g["check"] and g["passed"] is None for g in gates)
    assert np.isfinite(per_season(test)["margin_rmse"]).all()


def test_calibration_corrects_overconfident_ratings():
    games, _, _ = make_league(n_teams=24, seasons=(2020, 2021, 2022), n_days=110, games_per_day=8, seed=5)
    df = assemble(games, RatingsConfig())
    df["eng_margin_raw"] *= 1.3  # make the ratings deliberately overconfident
    test, maps = walk_forward(df, [2021, 2022])
    b, _home = maps[2022]["cal_margin"]
    assert 0.65 < b < 0.9  # ~1/1.3, learned from earlier seasons only
    assert len(maps[2022]["ctx_total"]) == 5 and len(maps[2022]["cal_total"]) == 2
    late = test[test["daynum"] >= 60]
    rmse_plain = np.sqrt(np.mean((late["margin_a"] - late["plain_margin"]) ** 2))
    rmse_cal = np.sqrt(np.mean((late["margin_a"] - late["cal_margin"]) ** 2))
    assert rmse_cal < rmse_plain - 0.1
