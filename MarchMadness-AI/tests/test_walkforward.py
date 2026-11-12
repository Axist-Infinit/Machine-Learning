import numpy as np
import yaml

from mmai.eval import PREREG_PATH
from mmai.eval.walkforward import assemble, gate_g1, per_season, score, time_travel_check, walk_forward
from mmai.model.ratings import RatingsConfig
from tests.synthetic import make_league


def test_end_to_end_on_a_synthetic_league():
    games, _, _ = make_league(n_teams=30, seasons=(2019, 2020, 2021, 2022), n_days=110, games_per_day=10, seed=4)
    cfg = RatingsConfig()
    df = assemble(games, cfg)
    test, maps = walk_forward(df, [2021, 2022])
    assert set(test["season"]) == {2021, 2022} and set(maps) == {2021, 2022}
    # mappings for 2022 may use 2021 but never 2022 itself
    assert maps[2021]["sigma"] != maps[2022]["sigma"]
    pooled = score(test)
    eng = pooled[(pooled["set"] == "after_jan15") & (pooled["model"] == "engine")].iloc[0]
    raw = pooled[(pooled["set"] == "after_jan15") & (pooled["model"] == "raw_net")].iloc[0]
    assert eng["margin_rmse"] < raw["margin_rmse"]  # opponent adjustment helps
    assert abs(eng["cov80"] - 0.8) < 0.05
    tt = time_travel_check(games, cfg, season=2022, cut_day=60)
    assert tt < 1e-9
    gates = gate_g1(test, yaml.safe_load(PREREG_PATH.read_text()), tt)
    assert gates[0]["passed"] and any("closing line" in g["check"] and g["passed"] is None for g in gates)
    assert np.isfinite(per_season(test)["margin_rmse"]).all()


def test_gate_g2_runs_and_context_is_inert_without_context_columns():
    from mmai.eval.walkforward import gate_g2

    games, _, _ = make_league(n_teams=24, seasons=(2020, 2021, 2022), n_days=80, games_per_day=8, seed=5)
    test, maps = walk_forward(assemble(games, RatingsConfig()), [2021, 2022])
    rows = gate_g2(test, yaml.safe_load(PREREG_PATH.read_text()))
    assert rows[-1]["check"].endswith("ships") and isinstance(rows[-1]["passed"], bool)
    assert len(maps[2022]["ctx_total"]) == 5 and len(maps[2022]["ctx_margin"]) == 5
