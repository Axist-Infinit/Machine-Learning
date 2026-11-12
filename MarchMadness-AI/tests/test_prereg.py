import yaml

from mmai.eval import PREREG_PATH


def test_prereg_is_v1_with_all_gates_and_holdout_rules():
    pr = yaml.safe_load(PREREG_PATH.read_text())
    assert pr["version"] == 1
    assert set(pr["gates"]) == {"G1_ratings", "G2_added_component", "G3_probabilities_and_kaggle",
                                "G4_scores_vs_market", "G5_betting"}
    assert pr["holdout"]["one_shot"] == [2026]
    assert pr["gates"]["G5_betting"]["on_fail"] == "fair lines only"
    assert pr["ridge_floor"]["brier"] == 0.190
