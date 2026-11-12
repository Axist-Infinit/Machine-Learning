from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

DEFAULT_RANDOM_STATE = 42
DEFAULT_TARGET_SEASON = 2026
DEFAULT_MIN_TRAIN_SEASON = 2003
DEFAULT_EVAL_START_SEASON = 2018
DEFAULT_TOURNEY_DAYNUM_CUTOFF = 133

REQUIRED_FILES = {
    "teams": "MTeams.csv",
    "regular": "MRegularSeasonDetailedResults.csv",
    "tourney": "MNCAATourneyDetailedResults.csv",
    "seeds": "MNCAATourneySeeds.csv",
}

OPTIONAL_FILES = {
    "massey": "MMasseyOrdinals.csv",
    "slots": "MNCAATourneySlots.csv",
    "seed_round_slots": "MNCAATourneySeedRoundSlots.csv",
    "seasons": "MSeasons.csv",
    "spellings": "MTeamSpellings.csv",
}


@dataclass(slots=True)
class TrainConfig:
    target_season: int = DEFAULT_TARGET_SEASON
    min_train_season: int = DEFAULT_MIN_TRAIN_SEASON
    eval_start_season: int = DEFAULT_EVAL_START_SEASON
    random_state: int = DEFAULT_RANDOM_STATE

    n_estimators_cls: int = 700
    n_estimators_quantile: int = 500
    learning_rate: float = 0.03
    max_depth: int = 4
    min_child_weight: float = 4.0
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.1
    reg_lambda: float = 2.0
    gamma: float = 0.05
    tree_method: str = "hist"
    n_jobs: int = 4

    tourney_daynum_cutoff: int = DEFAULT_TOURNEY_DAYNUM_CUTOFF
    clip_score_min: int = 40
    clip_score_max: int = 105

    quantiles: list[float] = field(default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 0.90])
    calibration_method: str = "isotonic"
    calibration_holdout_seasons: int = 3
    min_calibration_samples: int = 100
    blend_margin_win_prob_weight: float = 0.15
    external_prior_blend_win: float = 0.20
    external_prior_blend_margin: float = 0.25
    external_prior_blend_total: float = 0.25
    external_prior_min_rows: int = 250

    direct_market_min_rows: int = 12
    direct_market_min_class_count: int = 4
    direct_cover_blend_weight: float = 0.60
    direct_total_blend_weight: float = 0.60
    direct_market_history_lookback_seasons: int = 4
    direct_market_history_min_season: int = 2021

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "quantiles" in raw:
            raw["quantiles"] = [float(x) for x in raw["quantiles"]]
        return cls(**raw)
