from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .calibration import ProbabilityCalibrator, fit_probability_calibrator
from .config import TrainConfig
from .external_features import merge_external_team_features
from .features import build_matchup_frame, build_team_features
from .markets import add_reverse_market_rows, aggregate_market_history_to_consensus, fair_prob_to_american
from .simulation import (
    cover_prob_team_a,
    enforce_non_crossing_quantiles,
    normal_cdf,
    over_prob,
    quantile_label,
    reconcile_scores,
    robust_sigma_from_quantiles,
)


MARKET_FEATURE_COLUMNS = [
    "MarketProbTeamA",
    "MarketSpreadTeamA",
    "MarketTotal",
    "MarketBookCount",
    "MoneylineHold",
    "SpreadHold",
    "TotalHold",
    "HasMarketProb",
    "HasMarketSpread",
    "HasMarketTotal",
]

MARKET_MERGE_COLUMNS = [
    "MarketProbTeamA",
    "MarketProbTeamB",
    "MarketMoneylineTeamA",
    "MarketMoneylineTeamB",
    "MarketSpreadTeamA",
    "MarketSpreadPriceA",
    "MarketSpreadPriceB",
    "MarketTotal",
    "MarketOverPrice",
    "MarketUnderPrice",
    "MarketBookCount",
    "MoneylineHold",
    "SpreadHold",
    "TotalHold",
    "EventID",
    "CommenceTime",
]

WIN_MONOTONE_SPECS: dict[str, int] = {
    "MarketProbTeamA": 1,
    "MarketSpreadTeamA": -1,
    "Diff_EloEnd": 1,
    "Diff_NetRatingMean": 1,
    "Diff_SeedNum": -1,
}


@dataclass(slots=True)
class ModelBundle:
    win_model: Any
    win_calibrator: ProbabilityCalibrator | None
    margin_quantile_models: dict[str, Any]
    total_quantile_models: dict[str, Any]
    feature_columns: list[str]
    fill_values: dict[str, float]
    config: TrainConfig
    margin_total_corr: float = 0.0
    trained_with_market_prob: bool = False
    trained_with_market_spread: bool = False
    trained_with_market_total: bool = False
    cover_model: Any | None = None
    cover_calibrator: ProbabilityCalibrator | None = None
    over_model: Any | None = None
    over_calibrator: ProbabilityCalibrator | None = None
    direct_cover_training_rows: int = 0
    direct_over_training_rows: int = 0
    direct_market_history_rows: int = 0
    direct_market_training_seasons: list[int] = field(default_factory=list)
    external_prior_models: dict[str, Any] = field(default_factory=dict)
    external_prior_feature_columns: list[str] = field(default_factory=list)
    external_prior_fill_values: dict[str, float] = field(default_factory=dict)
    external_prior_summary: dict[str, Any] = field(default_factory=dict)



def make_tournament_training_rows(tourney_results: pd.DataFrame) -> pd.DataFrame:
    forward = pd.DataFrame(
        {
            "Season": tourney_results["Season"],
            "TeamAID": tourney_results["WTeamID"],
            "TeamBID": tourney_results["LTeamID"],
            "TeamAScore": tourney_results["WScore"],
            "TeamBScore": tourney_results["LScore"],
            "TeamAWin": 1,
        }
    )
    reverse = pd.DataFrame(
        {
            "Season": tourney_results["Season"],
            "TeamAID": tourney_results["LTeamID"],
            "TeamBID": tourney_results["WTeamID"],
            "TeamAScore": tourney_results["LScore"],
            "TeamBScore": tourney_results["WScore"],
            "TeamAWin": 0,
        }
    )
    rows = pd.concat([forward, reverse], ignore_index=True)
    rows["Margin"] = rows["TeamAScore"] - rows["TeamBScore"]
    rows["Total"] = rows["TeamAScore"] + rows["TeamBScore"]
    return rows.sort_values(["Season", "TeamAID", "TeamBID"]).reset_index(drop=True)



def merge_market_history_into_rows(
    rows: pd.DataFrame,
    market_history: pd.DataFrame | None,
) -> pd.DataFrame:
    if market_history is None or market_history.empty:
        return rows.copy()

    market = aggregate_market_history_to_consensus(market_history)
    if market.empty:
        return rows.copy()
    required = {"Season", "TeamAID", "TeamBID"}
    missing = required - set(market.columns)
    if missing:
        raise ValueError(
            "Market history must be resolved to TeamAID/TeamBID/Season before training. "
            f"Missing columns: {sorted(missing)}"
        )

    market = add_reverse_market_rows(market)
    market = market.drop(columns=[c for c in ["Orientation"] if c in market.columns])
    keep_cols = [c for c in ["Season", "TeamAID", "TeamBID", *MARKET_MERGE_COLUMNS] if c in market.columns]
    market = market[keep_cols].drop_duplicates(subset=["Season", "TeamAID", "TeamBID"])
    return rows.merge(market, on=["Season", "TeamAID", "TeamBID"], how="left")



def _build_monotone_constraints(feature_columns: list[str], specs: dict[str, int]) -> tuple[int, ...] | None:
    constraints = tuple(int(specs.get(col, 0)) for col in feature_columns)
    if any(value != 0 for value in constraints):
        return constraints
    return None



def _make_classifier(config: TrainConfig, feature_columns: list[str]) -> XGBClassifier:
    monotone_constraints = _build_monotone_constraints(feature_columns, WIN_MONOTONE_SPECS)
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=config.n_estimators_cls,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        gamma=config.gamma,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        tree_method=config.tree_method,
        monotone_constraints=monotone_constraints,
        verbosity=0,
    )



def _make_quantile_regressor(config: TrainConfig, alpha: float) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=float(alpha),
        n_estimators=config.n_estimators_quantile,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        gamma=config.gamma,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        tree_method=config.tree_method,
        verbosity=0,
    )



def _fit_fill_values(x: pd.DataFrame) -> dict[str, float]:
    medians = x.median(numeric_only=True).fillna(0.0)
    return {str(k): float(v) for k, v in medians.items()}



def _apply_fill_values(x: pd.DataFrame, fill_values: dict[str, float]) -> pd.DataFrame:
    out = x.copy()
    for col, value in fill_values.items():
        if col in out.columns:
            out[col] = out[col].fillna(value)
    out = out.fillna(0.0)
    return out



def _clip_prob(prob: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)



def _logit(prob: pd.Series | np.ndarray) -> np.ndarray:
    p = _clip_prob(prob)
    return np.log(p / (1.0 - p))



def _compute_hold(df: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
    if col_a not in df.columns or col_b not in df.columns:
        return pd.Series(np.nan, index=df.index)

    def _single(row: pd.Series) -> float:
        a = row[col_a]
        b = row[col_b]
        if pd.isna(a) or pd.isna(b):
            return np.nan
        a = float(a)
        b = float(b)
        if a > 0:
            pa = 100.0 / (a + 100.0)
        else:
            pa = abs(a) / (abs(a) + 100.0)
        if b > 0:
            pb = 100.0 / (b + 100.0)
        else:
            pb = abs(b) / (abs(b) + 100.0)
        return pa + pb - 1.0

    return df.apply(_single, axis=1)



def attach_market_inputs(matchup_frame: pd.DataFrame, source_rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = matchup_frame.reset_index(drop=True).copy()
    src = source_rows.reset_index(drop=True).copy()

    defaults = {
        "MarketProbTeamA": np.nan,
        "MarketProbTeamB": np.nan,
        "MarketMoneylineTeamA": np.nan,
        "MarketMoneylineTeamB": np.nan,
        "MarketSpreadTeamA": np.nan,
        "MarketSpreadPriceA": np.nan,
        "MarketSpreadPriceB": np.nan,
        "MarketTotal": np.nan,
        "MarketOverPrice": np.nan,
        "MarketUnderPrice": np.nan,
        "MarketBookCount": 0.0,
        "MoneylineHold": np.nan,
        "SpreadHold": np.nan,
        "TotalHold": np.nan,
        "EventID": "",
        "CommenceTime": "",
    }

    for col, default in defaults.items():
        if col in src.columns:
            out[col] = src[col].reset_index(drop=True)
        else:
            out[col] = default

    numeric_cols = [
        "MarketProbTeamA",
        "MarketProbTeamB",
        "MarketMoneylineTeamA",
        "MarketMoneylineTeamB",
        "MarketSpreadTeamA",
        "MarketSpreadPriceA",
        "MarketSpreadPriceB",
        "MarketTotal",
        "MarketOverPrice",
        "MarketUnderPrice",
        "MarketBookCount",
        "MoneylineHold",
        "SpreadHold",
        "TotalHold",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out["MarketProbTeamA"].isna().any() and {"MarketMoneylineTeamA", "MarketMoneylineTeamB"}.issubset(out.columns):
        mask = out["MarketProbTeamA"].isna() & out["MarketMoneylineTeamA"].notna() & out["MarketMoneylineTeamB"].notna()
        if mask.any():
            pa = out.loc[mask, "MarketMoneylineTeamA"].astype(float)
            pb = out.loc[mask, "MarketMoneylineTeamB"].astype(float)
            pa_imp = np.where(pa > 0, 100.0 / (pa + 100.0), np.abs(pa) / (np.abs(pa) + 100.0))
            pb_imp = np.where(pb > 0, 100.0 / (pb + 100.0), np.abs(pb) / (np.abs(pb) + 100.0))
            total = pa_imp + pb_imp
            out.loc[mask, "MarketProbTeamA"] = pa_imp / total
            out.loc[mask, "MarketProbTeamB"] = pb_imp / total

    mask_b = out["MarketProbTeamA"].notna() & out["MarketProbTeamB"].isna()
    out.loc[mask_b, "MarketProbTeamB"] = 1.0 - out.loc[mask_b, "MarketProbTeamA"]

    if out["MoneylineHold"].isna().any():
        computed = _compute_hold(out, "MarketMoneylineTeamA", "MarketMoneylineTeamB")
        mask = out["MoneylineHold"].isna()
        out.loc[mask, "MoneylineHold"] = computed.loc[mask]
    if out["SpreadHold"].isna().any():
        computed = _compute_hold(out, "MarketSpreadPriceA", "MarketSpreadPriceB")
        mask = out["SpreadHold"].isna()
        out.loc[mask, "SpreadHold"] = computed.loc[mask]
    if out["TotalHold"].isna().any():
        computed = _compute_hold(out, "MarketOverPrice", "MarketUnderPrice")
        mask = out["TotalHold"].isna()
        out.loc[mask, "TotalHold"] = computed.loc[mask]

    out["HasMarketProb"] = out["MarketProbTeamA"].notna().astype(int)
    out["HasMarketSpread"] = out["MarketSpreadTeamA"].notna().astype(int)
    out["HasMarketTotal"] = out["MarketTotal"].notna().astype(int)

    out["WinBaseMargin"] = np.where(out["HasMarketProb"] == 1, _logit(out["MarketProbTeamA"]), 0.0)
    out["MarginBase"] = np.where(out["HasMarketSpread"] == 1, out["MarketSpreadTeamA"], 0.0)
    out["TotalBase"] = np.where(out["HasMarketTotal"] == 1, out["MarketTotal"], 0.0)

    return out, MARKET_FEATURE_COLUMNS.copy()



def build_training_matrix(
    training_rows: pd.DataFrame,
    team_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    frame, base_feature_cols = build_matchup_frame(training_rows, team_features)
    frame, market_feature_cols = attach_market_inputs(frame, training_rows)
    feature_cols = base_feature_cols + market_feature_cols
    return frame, feature_cols



def add_direct_market_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    spread_residual = np.where(
        out["HasMarketSpread"].fillna(0).astype(int) == 1,
        pd.to_numeric(out["Margin"], errors="coerce") + pd.to_numeric(out["MarketSpreadTeamA"], errors="coerce"),
        np.nan,
    )
    total_residual = np.where(
        out["HasMarketTotal"].fillna(0).astype(int) == 1,
        pd.to_numeric(out["Total"], errors="coerce") - pd.to_numeric(out["MarketTotal"], errors="coerce"),
        np.nan,
    )
    out["SpreadResidual"] = spread_residual
    out["TotalResidual"] = total_residual
    out["CoverTarget"] = np.where(
        pd.to_numeric(out["SpreadResidual"], errors="coerce").notna()
        & (pd.to_numeric(out["SpreadResidual"], errors="coerce").abs() > 1e-8),
        (pd.to_numeric(out["SpreadResidual"], errors="coerce") > 0).astype(int),
        np.nan,
    )
    out["OverTarget"] = np.where(
        pd.to_numeric(out["TotalResidual"], errors="coerce").notna()
        & (pd.to_numeric(out["TotalResidual"], errors="coerce").abs() > 1e-8),
        (pd.to_numeric(out["TotalResidual"], errors="coerce") > 0).astype(int),
        np.nan,
    )
    return out



def _align_direct_feature_matrix(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float],
) -> pd.DataFrame:
    x_direct = training_frame.copy()
    for col in feature_columns:
        if col not in x_direct.columns:
            x_direct[col] = np.nan
    x_direct = x_direct[feature_columns].copy()
    return _apply_fill_values(x_direct, fill_values)



def _fit_win_model_and_calibrator(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    seasons: pd.Series,
    base_margin: np.ndarray,
    config: TrainConfig,
    feature_columns: list[str],
) -> tuple[XGBClassifier, ProbabilityCalibrator | None]:
    calibrator: ProbabilityCalibrator | None = None

    method = str(config.calibration_method or "none").lower().strip()
    if method not in {"", "none"}:
        cutoff = max(config.min_train_season + 1, config.target_season - config.calibration_holdout_seasons)
        calib_mask = seasons.to_numpy(dtype=int) >= cutoff
        train_mask = ~calib_mask

        if (
            calib_mask.sum() >= config.min_calibration_samples
            and train_mask.sum() >= max(config.min_calibration_samples, 200)
            and y_train.loc[calib_mask].nunique() == 2
            and y_train.loc[train_mask].nunique() == 2
        ):
            pre_model = _make_classifier(config, feature_columns)
            pre_model.fit(
                x_train.loc[train_mask],
                y_train.loc[train_mask],
                base_margin=base_margin[train_mask],
            )
            raw_prob = pre_model.predict_proba(
                x_train.loc[calib_mask],
                base_margin=base_margin[calib_mask],
            )[:, 1]
            calibrator = fit_probability_calibrator(raw_prob, y_train.loc[calib_mask].to_numpy(), method=method)

    final_model = _make_classifier(config, feature_columns)
    final_model.fit(x_train, y_train, base_margin=base_margin)
    return final_model, calibrator



def _fit_direct_binary_market_model(
    training_frame: pd.DataFrame,
    x_train: pd.DataFrame,
    seasons: pd.Series,
    target_col: str,
    config: TrainConfig,
    feature_columns: list[str],
) -> tuple[Any | None, ProbabilityCalibrator | None, int]:
    if target_col not in training_frame.columns:
        return None, None, 0

    y = pd.to_numeric(training_frame[target_col], errors="coerce")
    valid_mask = y.notna()
    row_count = int(valid_mask.sum())
    if row_count < int(getattr(config, "direct_market_min_rows", 12)):
        return None, None, row_count

    y_valid = y.loc[valid_mask].astype(int)
    class_counts = y_valid.value_counts()
    if len(class_counts) < 2 or int(class_counts.min()) < int(getattr(config, "direct_market_min_class_count", 4)):
        return None, None, row_count

    x_valid = x_train.loc[valid_mask].copy()
    seasons_valid = seasons.loc[valid_mask].astype(int)
    base_margin = np.zeros(len(x_valid), dtype=float)
    model, calibrator = _fit_win_model_and_calibrator(
        x_train=x_valid,
        y_train=y_valid,
        seasons=seasons_valid,
        base_margin=base_margin,
        config=config,
        feature_columns=feature_columns,
    )
    return model, calibrator, row_count



def _predict_binary_probabilities(model: Any | None, calibrator: ProbabilityCalibrator | None, x_pred: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.full(len(x_pred), np.nan, dtype=float)
    prob = np.asarray(model.predict_proba(x_pred)[:, 1], dtype=float)
    if calibrator is not None:
        prob = np.asarray(calibrator.predict(prob), dtype=float)
    return np.clip(prob, 1e-6, 1.0 - 1e-6)



def train_model_bundle(
    training_rows: pd.DataFrame,
    team_features: pd.DataFrame,
    config: TrainConfig,
    direct_market_training_rows: pd.DataFrame | None = None,
    direct_market_team_features: pd.DataFrame | None = None,
) -> tuple[ModelBundle, pd.DataFrame]:
    training_frame, feature_cols = build_training_matrix(training_rows, team_features)
    training_frame = add_direct_market_targets(training_frame)

    x_train = training_frame[feature_cols].copy()
    fill_values = _fit_fill_values(x_train)
    x_train = _apply_fill_values(x_train, fill_values)

    y_cls = training_frame["TeamAWin"].astype(int)
    y_margin = (training_frame["Margin"].astype(float) - training_frame["MarginBase"].astype(float))
    y_total = (training_frame["Total"].astype(float) - training_frame["TotalBase"].astype(float))
    base_margin = training_frame["WinBaseMargin"].astype(float).to_numpy()
    seasons = training_frame["Season"].astype(int)

    direct_training_frame = training_frame
    direct_x_train = x_train
    direct_seasons = seasons
    direct_history_rows = int(len(training_frame))
    direct_training_seasons = sorted(int(x) for x in pd.to_numeric(training_frame["Season"], errors="coerce").dropna().unique().tolist())

    if direct_market_training_rows is not None and not direct_market_training_rows.empty:
        source_team_features = direct_market_team_features if direct_market_team_features is not None and not direct_market_team_features.empty else team_features
        direct_training_frame, _ = build_training_matrix(direct_market_training_rows, source_team_features)
        direct_training_frame = add_direct_market_targets(direct_training_frame)
        direct_x_train = _align_direct_feature_matrix(direct_training_frame, feature_cols, fill_values)
        direct_seasons = pd.to_numeric(direct_training_frame["Season"], errors="coerce").fillna(config.target_season).astype(int)
        direct_history_rows = int(len(direct_training_frame))
        direct_training_seasons = sorted(int(x) for x in pd.to_numeric(direct_training_frame["Season"], errors="coerce").dropna().unique().tolist())

    win_model, win_calibrator = _fit_win_model_and_calibrator(
        x_train=x_train,
        y_train=y_cls,
        seasons=seasons,
        base_margin=base_margin,
        config=config,
        feature_columns=feature_cols,
    )

    cover_model, cover_calibrator, cover_rows = _fit_direct_binary_market_model(
        training_frame=direct_training_frame,
        x_train=direct_x_train,
        seasons=direct_seasons,
        target_col="CoverTarget",
        config=config,
        feature_columns=feature_cols,
    )
    over_model, over_calibrator, over_rows = _fit_direct_binary_market_model(
        training_frame=direct_training_frame,
        x_train=direct_x_train,
        seasons=direct_seasons,
        target_col="OverTarget",
        config=config,
        feature_columns=feature_cols,
    )

    margin_quantile_models: dict[str, Any] = {}
    total_quantile_models: dict[str, Any] = {}
    for alpha in config.quantiles:
        label = quantile_label(alpha)
        margin_model = _make_quantile_regressor(config, alpha)
        total_model = _make_quantile_regressor(config, alpha)
        margin_model.fit(x_train, y_margin)
        total_model.fit(x_train, y_total)
        margin_quantile_models[label] = margin_model
        total_quantile_models[label] = total_model

    corr = float(training_frame[["Margin", "Total"]].corr().iloc[0, 1]) if len(training_frame) >= 3 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    corr = float(np.clip(corr, -0.75, 0.75))

    bundle = ModelBundle(
        win_model=win_model,
        win_calibrator=win_calibrator,
        margin_quantile_models=margin_quantile_models,
        total_quantile_models=total_quantile_models,
        feature_columns=feature_cols,
        fill_values=fill_values,
        config=config,
        margin_total_corr=corr,
        trained_with_market_prob=bool(training_frame["HasMarketProb"].fillna(0).sum() > 0),
        trained_with_market_spread=bool(training_frame["HasMarketSpread"].fillna(0).sum() > 0),
        trained_with_market_total=bool(training_frame["HasMarketTotal"].fillna(0).sum() > 0),
        cover_model=cover_model,
        cover_calibrator=cover_calibrator,
        over_model=over_model,
        over_calibrator=over_calibrator,
        direct_cover_training_rows=int(cover_rows),
        direct_over_training_rows=int(over_rows),
        direct_market_history_rows=int(direct_history_rows),
        direct_market_training_seasons=direct_training_seasons,
    )
    return bundle, training_frame



def _make_game_training_rows(results: pd.DataFrame) -> pd.DataFrame:
    forward = pd.DataFrame(
        {
            "Season": results["Season"],
            "TeamAID": results["WTeamID"],
            "TeamBID": results["LTeamID"],
            "TeamAScore": results["WScore"],
            "TeamBScore": results["LScore"],
            "TeamAWin": 1,
        }
    )
    reverse = pd.DataFrame(
        {
            "Season": results["Season"],
            "TeamAID": results["LTeamID"],
            "TeamBID": results["WTeamID"],
            "TeamAScore": results["LScore"],
            "TeamBScore": results["WScore"],
            "TeamAWin": 0,
        }
    )
    rows = pd.concat([forward, reverse], ignore_index=True)
    rows["Margin"] = rows["TeamAScore"] - rows["TeamBScore"]
    rows["Total"] = rows["TeamAScore"] + rows["TeamBScore"]
    return rows



def fit_external_prior_models(
    regular_results: pd.DataFrame,
    team_features: pd.DataFrame,
    config: TrainConfig,
) -> tuple[dict[str, Any], list[str], dict[str, float], dict[str, Any]]:
    season_games = regular_results.loc[
        (regular_results["Season"] == int(config.target_season))
        & (regular_results["DayNum"] <= int(config.tourney_daynum_cutoff))
    ].copy()
    if season_games.empty:
        return {}, [], {}, {"status": "no_target_season_regular_games"}

    neutral_games = season_games.loc[season_games["WLoc"].astype(str).eq("N")].copy()
    source_games = neutral_games if len(neutral_games) >= int(config.external_prior_min_rows) else season_games
    source_name = "neutral_regular" if len(neutral_games) >= int(config.external_prior_min_rows) else "regular_all_sites"

    rows = _make_game_training_rows(source_games)
    if rows.empty:
        return {}, [], {}, {"status": "no_rows_after_external_prior_filter"}

    frame, _ = build_matchup_frame(rows, team_features)
    ext_feature_columns = [
        col
        for col in frame.columns
        if col.startswith("Diff_Ext") or col.startswith("Avg_Ext") or col in {"Diff_ExtCompositeSide", "Avg_ExtCompositeSide", "Diff_ExtCompositeTotal", "Avg_ExtCompositeTotal"}
    ]
    ext_feature_columns = sorted(set(ext_feature_columns))
    if not ext_feature_columns:
        return {}, [], {}, {"status": "no_external_columns_available"}

    x = frame[ext_feature_columns].copy()
    non_null_features = int((x.notna().sum() > 0).sum())
    if non_null_features == 0 or len(rows) < int(config.external_prior_min_rows):
        return {}, ext_feature_columns, {}, {
            "status": "insufficient_external_signal",
            "rows": int(len(rows)),
            "non_null_features": non_null_features,
            "source": source_name,
        }

    fill_values = _fit_fill_values(x)
    x = _apply_fill_values(x, fill_values)
    y_cls = rows["TeamAWin"].astype(int).to_numpy()
    y_margin = rows["Margin"].astype(float).to_numpy()
    y_total = rows["Total"].astype(float).to_numpy()

    win_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.7))
    margin_model = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
    total_model = make_pipeline(StandardScaler(), Ridge(alpha=2.0))

    win_model.fit(x, y_cls)
    margin_model.fit(x, y_margin)
    total_model.fit(x, y_total)

    win_prob = win_model.predict_proba(x)[:, 1]
    margin_pred = margin_model.predict(x)
    total_pred = total_model.predict(x)

    summary = {
        "status": "ok",
        "source": source_name,
        "rows": int(len(rows)),
        "features": int(len(ext_feature_columns)),
        "win_accuracy": float(accuracy_score(y_cls, (win_prob >= 0.5).astype(int))),
        "win_logloss": float(log_loss(y_cls, np.clip(win_prob, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "margin_mae": float(mean_absolute_error(y_margin, margin_pred)),
        "total_mae": float(mean_absolute_error(y_total, total_pred)),
    }
    models = {"win": win_model, "margin": margin_model, "total": total_model}
    return models, ext_feature_columns, fill_values, summary



def _score_predictions(
    actual_win: np.ndarray,
    win_prob: np.ndarray,
    actual_margin: np.ndarray,
    pred_margin: np.ndarray,
    actual_total: np.ndarray,
    pred_total: np.ndarray,
) -> dict[str, float]:
    clipped = np.clip(win_prob, 1e-6, 1 - 1e-6)
    pred_label = (clipped >= 0.5).astype(int)
    return {
        "brier": float(mean_squared_error(actual_win, clipped)),
        "logloss": float(log_loss(actual_win, clipped, labels=[0, 1])),
        "accuracy": float(accuracy_score(actual_win, pred_label)),
        "margin_mae": float(mean_absolute_error(actual_margin, pred_margin)),
        "total_mae": float(mean_absolute_error(actual_total, pred_total)),
    }



def _blend_probabilities(classifier_prob: np.ndarray, center_margin: np.ndarray, sigma_margin: np.ndarray, weight: float) -> np.ndarray:
    p_margin = 1.0 - normal_cdf(0.0, center_margin, sigma_margin)
    p_margin = np.clip(p_margin, 1e-6, 1 - 1e-6)
    weight = float(np.clip(weight, 0.0, 1.0))
    return np.clip((1.0 - weight) * classifier_prob + weight * p_margin, 1e-6, 1 - 1e-6)



def predict_matchups(
    matchups: pd.DataFrame,
    team_features: pd.DataFrame,
    bundle: ModelBundle,
) -> pd.DataFrame:
    matchup_frame, feature_cols = build_matchup_frame(matchups, team_features)
    matchup_frame, market_feature_cols = attach_market_inputs(matchup_frame, matchups)
    full_feature_cols = feature_cols + market_feature_cols

    if full_feature_cols != bundle.feature_columns:
        missing = sorted(set(bundle.feature_columns) - set(full_feature_cols))
        extra = sorted(set(full_feature_cols) - set(bundle.feature_columns))
        if missing or extra:
            raise ValueError(f"Feature mismatch. Missing={missing[:10]} Extra={extra[:10]}")

    x_pred = _apply_fill_values(matchup_frame[bundle.feature_columns], bundle.fill_values)
    win_base_margin = matchup_frame["WinBaseMargin"].astype(float).to_numpy() if bundle.trained_with_market_prob else np.zeros(len(matchup_frame), dtype=float)
    margin_base = matchup_frame["MarginBase"].astype(float).to_numpy() if bundle.trained_with_market_spread else np.zeros(len(matchup_frame), dtype=float)
    total_base = matchup_frame["TotalBase"].astype(float).to_numpy() if bundle.trained_with_market_total else np.zeros(len(matchup_frame), dtype=float)

    raw_win_prob = bundle.win_model.predict_proba(x_pred, base_margin=win_base_margin)[:, 1]
    if bundle.win_calibrator is not None:
        raw_win_prob = bundle.win_calibrator.predict(raw_win_prob)

    margin_preds: dict[float, np.ndarray] = {}
    total_preds: dict[float, np.ndarray] = {}
    quantile_alphas = [float(alpha) for alpha in bundle.config.quantiles]

    for alpha in quantile_alphas:
        label = quantile_label(alpha)
        margin_resid = bundle.margin_quantile_models[label].predict(x_pred)
        total_resid = bundle.total_quantile_models[label].predict(x_pred)
        margin_preds[alpha] = np.asarray(margin_resid, dtype=float) + margin_base
        total_preds[alpha] = np.asarray(total_resid, dtype=float) + total_base

    margin_matrix = np.column_stack([margin_preds[alpha] for alpha in quantile_alphas])
    total_matrix = np.column_stack([total_preds[alpha] for alpha in quantile_alphas])
    margin_matrix = enforce_non_crossing_quantiles(margin_matrix)
    total_matrix = enforce_non_crossing_quantiles(total_matrix)

    margin_preds = {alpha: margin_matrix[:, idx] for idx, alpha in enumerate(sorted(quantile_alphas))}
    total_preds = {alpha: total_matrix[:, idx] for idx, alpha in enumerate(sorted(quantile_alphas))}

    center_margin = np.asarray(margin_preds.get(0.50, next(iter(margin_preds.values()))), dtype=float)
    center_total = np.asarray(total_preds.get(0.50, next(iter(total_preds.values()))), dtype=float)
    sigma_margin = robust_sigma_from_quantiles(margin_preds)
    sigma_total = robust_sigma_from_quantiles(total_preds)

    ext_win_prob = np.full(len(matchup_frame), np.nan)
    ext_margin = np.full(len(matchup_frame), np.nan)
    ext_total = np.full(len(matchup_frame), np.nan)
    if bundle.external_prior_models and bundle.external_prior_feature_columns:
        ext_x = matchup_frame[bundle.external_prior_feature_columns].copy()
        has_ext_signal = ext_x.notna().any(axis=1).to_numpy(dtype=bool)
        ext_x = _apply_fill_values(ext_x, bundle.external_prior_fill_values)
        if has_ext_signal.any():
            ext_win_prob = np.asarray(bundle.external_prior_models["win"].predict_proba(ext_x)[:, 1], dtype=float)
            ext_margin = np.asarray(bundle.external_prior_models["margin"].predict(ext_x), dtype=float)
            ext_total = np.asarray(bundle.external_prior_models["total"].predict(ext_x), dtype=float)

            margin_weight = float(np.clip(bundle.config.external_prior_blend_margin, 0.0, 1.0))
            total_weight = float(np.clip(bundle.config.external_prior_blend_total, 0.0, 1.0))
            center_margin_orig = center_margin.copy()
            center_total_orig = center_total.copy()
            center_margin = np.where(has_ext_signal, (1.0 - margin_weight) * center_margin + margin_weight * ext_margin, center_margin)
            center_total = np.where(has_ext_signal, (1.0 - total_weight) * center_total + total_weight * ext_total, center_total)
            margin_shift = center_margin - center_margin_orig
            total_shift = center_total - center_total_orig
            for alpha in margin_preds:
                margin_preds[alpha] = np.asarray(margin_preds[alpha], dtype=float) + margin_shift
            for alpha in total_preds:
                total_preds[alpha] = np.asarray(total_preds[alpha], dtype=float) + total_shift
            sigma_margin = robust_sigma_from_quantiles(margin_preds)
            sigma_total = robust_sigma_from_quantiles(total_preds)

    win_prob = _blend_probabilities(
        classifier_prob=np.asarray(raw_win_prob, dtype=float),
        center_margin=np.asarray(center_margin, dtype=float),
        sigma_margin=np.asarray(sigma_margin, dtype=float),
        weight=bundle.config.blend_margin_win_prob_weight,
    )

    if bundle.external_prior_models and bundle.external_prior_feature_columns:
        has_ext_signal = matchup_frame[bundle.external_prior_feature_columns].notna().any(axis=1).to_numpy(dtype=bool)
        win_weight = float(np.clip(bundle.config.external_prior_blend_win, 0.0, 1.0))
        win_prob = np.where(has_ext_signal, np.clip((1.0 - win_weight) * win_prob + win_weight * ext_win_prob, 1e-6, 1.0 - 1e-6), win_prob)

    score_a, score_b, margin_rec, total_rec = reconcile_scores(
        win_prob=win_prob,
        margin_pred=center_margin,
        total_pred=center_total,
        clip_min=bundle.config.clip_score_min,
        clip_max=bundle.config.clip_score_max,
    )

    out = matchup_frame[["Season", "TeamAID", "TeamBID"]].copy()
    if "EventID" in matchup_frame.columns:
        out["EventID"] = matchup_frame["EventID"]
    if "CommenceTime" in matchup_frame.columns:
        out["CommenceTime"] = matchup_frame["CommenceTime"]
    out["WinProbTeamA"] = win_prob
    out["WinProbTeamB"] = 1.0 - win_prob
    out["PredScoreTeamA"] = score_a
    out["PredScoreTeamB"] = score_b
    out["PredMargin"] = margin_rec
    out["PredTotal"] = total_rec
    out["PredWinnerTeamID"] = np.where(win_prob >= 0.5, out["TeamAID"], out["TeamBID"])
    out["MarginSigma"] = sigma_margin
    out["TotalSigma"] = sigma_total
    out["FairMoneylineTeamA"] = [fair_prob_to_american(p) for p in out["WinProbTeamA"]]
    out["FairMoneylineTeamB"] = [fair_prob_to_american(p) for p in out["WinProbTeamB"]]
    out["FairSpreadTeamA"] = -np.asarray(center_margin, dtype=float)
    out["FairTotal"] = np.asarray(center_total, dtype=float)
    if bundle.external_prior_models and bundle.external_prior_feature_columns:
        out["ExtPriorWinProbTeamA"] = ext_win_prob
        out["ExtPriorMargin"] = ext_margin
        out["ExtPriorTotal"] = ext_total

    for alpha in sorted(quantile_alphas):
        label = quantile_label(alpha)
        out[f"PredMargin{label}"] = margin_preds[alpha]
        out[f"PredTotal{label}"] = total_preds[alpha]

    market_cols = [
        "MarketProbTeamA",
        "MarketMoneylineTeamA",
        "MarketMoneylineTeamB",
        "MarketSpreadTeamA",
        "MarketSpreadPriceA",
        "MarketSpreadPriceB",
        "MarketTotal",
        "MarketOverPrice",
        "MarketUnderPrice",
        "MarketBookCount",
        "MoneylineHold",
        "SpreadHold",
        "TotalHold",
    ]
    for col in market_cols:
        if col in matchup_frame.columns:
            out[col] = matchup_frame[col]

    direct_cover_prob = _predict_binary_probabilities(bundle.cover_model, bundle.cover_calibrator, x_pred)
    direct_over_prob = _predict_binary_probabilities(bundle.over_model, bundle.over_calibrator, x_pred)

    if "MarketSpreadTeamA" in out.columns:
        spread_line = pd.to_numeric(out["MarketSpreadTeamA"], errors="coerce")
        dist_cover = np.full(len(out), np.nan, dtype=float)
        has_line = spread_line.notna().to_numpy(dtype=bool)
        if has_line.any():
            dist_cover[has_line] = cover_prob_team_a(
                np.asarray(center_margin, dtype=float)[has_line],
                np.asarray(sigma_margin, dtype=float)[has_line],
                spread_line.loc[has_line].to_numpy(dtype=float),
            )
        out["DistProbCoverTeamA"] = dist_cover
        out["DirectProbCoverTeamA"] = np.where(has_line, direct_cover_prob, np.nan)
        cover_weight = float(np.clip(getattr(bundle.config, "direct_cover_blend_weight", 0.60), 0.0, 1.0))
        blended_cover = np.where(
            np.isfinite(out["DirectProbCoverTeamA"].to_numpy(dtype=float)),
            (1.0 - cover_weight) * out["DistProbCoverTeamA"].to_numpy(dtype=float) + cover_weight * out["DirectProbCoverTeamA"].to_numpy(dtype=float),
            out["DistProbCoverTeamA"].to_numpy(dtype=float),
        )
        out["ModelProbCoverTeamA"] = np.clip(blended_cover, 1e-6, 1.0 - 1e-6)
        out.loc[~has_line, "ModelProbCoverTeamA"] = np.nan
        out["ModelProbCoverTeamB"] = 1.0 - out["ModelProbCoverTeamA"]
        out.loc[~has_line, "ModelProbCoverTeamB"] = np.nan
    else:
        out["DistProbCoverTeamA"] = np.nan
        out["DirectProbCoverTeamA"] = np.nan
        out["ModelProbCoverTeamA"] = np.nan
        out["ModelProbCoverTeamB"] = np.nan

    if "MarketTotal" in out.columns:
        total_line = pd.to_numeric(out["MarketTotal"], errors="coerce")
        dist_over = np.full(len(out), np.nan, dtype=float)
        has_total = total_line.notna().to_numpy(dtype=bool)
        if has_total.any():
            dist_over[has_total] = over_prob(
                np.asarray(center_total, dtype=float)[has_total],
                np.asarray(sigma_total, dtype=float)[has_total],
                total_line.loc[has_total].to_numpy(dtype=float),
            )
        out["DistProbOver"] = dist_over
        out["DirectProbOver"] = np.where(has_total, direct_over_prob, np.nan)
        total_weight = float(np.clip(getattr(bundle.config, "direct_total_blend_weight", 0.60), 0.0, 1.0))
        blended_total = np.where(
            np.isfinite(out["DirectProbOver"].to_numpy(dtype=float)),
            (1.0 - total_weight) * out["DistProbOver"].to_numpy(dtype=float) + total_weight * out["DirectProbOver"].to_numpy(dtype=float),
            out["DistProbOver"].to_numpy(dtype=float),
        )
        out["ModelProbOver"] = np.clip(blended_total, 1e-6, 1.0 - 1e-6)
        out.loc[~has_total, "ModelProbOver"] = np.nan
        out["ModelProbUnder"] = 1.0 - out["ModelProbOver"]
        out.loc[~has_total, "ModelProbUnder"] = np.nan
    else:
        out["DistProbOver"] = np.nan
        out["DirectProbOver"] = np.nan
        out["ModelProbOver"] = np.nan
        out["ModelProbUnder"] = np.nan

    return out



def walk_forward_backtest(
    tourney_training_rows: pd.DataFrame,
    team_features: pd.DataFrame,
    config: TrainConfig,
) -> pd.DataFrame:
    seasons = sorted(
        season
        for season in tourney_training_rows["Season"].unique()
        if season >= config.eval_start_season and season < config.target_season
    )
    rows: list[dict[str, float | int]] = []

    for holdout_season in seasons:
        train_rows = tourney_training_rows.loc[
            (tourney_training_rows["Season"] >= config.min_train_season)
            & (tourney_training_rows["Season"] < holdout_season)
        ].copy()
        test_rows = tourney_training_rows.loc[tourney_training_rows["Season"] == holdout_season].copy()
        if train_rows.empty or test_rows.empty:
            continue

        eval_config = TrainConfig(**config.to_dict())
        eval_config.target_season = int(holdout_season)
        bundle, _ = train_model_bundle(train_rows, team_features, eval_config)
        preds = predict_matchups(matchups=test_rows[[c for c in test_rows.columns if c in set(test_rows.columns)]].copy(), team_features=team_features, bundle=bundle)
        metrics = _score_predictions(
            actual_win=test_rows["TeamAWin"].to_numpy(),
            win_prob=preds["WinProbTeamA"].to_numpy(),
            actual_margin=test_rows["Margin"].to_numpy(),
            pred_margin=preds["PredMargin"].to_numpy(),
            actual_total=test_rows["Total"].to_numpy(),
            pred_total=preds["PredTotal"].to_numpy(),
        )
        metrics["season"] = int(holdout_season)
        rows.append(metrics)

    return pd.DataFrame(rows)



def train_from_raw_data(
    regular_results: pd.DataFrame,
    tourney_results: pd.DataFrame,
    seeds: pd.DataFrame,
    massey: pd.DataFrame | None,
    config: TrainConfig,
    market_history: pd.DataFrame | None = None,
    extra_team_features: pd.DataFrame | None = None,
) -> tuple[ModelBundle, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    team_features = build_team_features(
        regular_results=regular_results,
        seeds=seeds,
        massey=massey,
        cutoff_daynum=config.tourney_daynum_cutoff,
    )
    team_features = merge_external_team_features(team_features, extra_team_features)
    tourney_rows = make_tournament_training_rows(tourney_results)
    tourney_rows = merge_market_history_into_rows(tourney_rows, market_history)
    train_rows = tourney_rows.loc[
        (tourney_rows["Season"] >= config.min_train_season)
        & (tourney_rows["Season"] < config.target_season)
    ].copy()
    bundle, training_frame = train_model_bundle(train_rows, team_features, config)
    ext_models, ext_cols, ext_fill, ext_summary = fit_external_prior_models(
        regular_results=regular_results,
        team_features=team_features,
        config=config,
    )
    bundle.external_prior_models = ext_models
    bundle.external_prior_feature_columns = ext_cols
    bundle.external_prior_fill_values = ext_fill
    bundle.external_prior_summary = ext_summary
    backtest = walk_forward_backtest(tourney_rows, team_features, config)
    return bundle, team_features, training_frame, backtest



def export_feature_importance(bundle: ModelBundle) -> pd.DataFrame:
    def _frame(model: Any, model_name: str) -> pd.DataFrame:
        if model is None:
            return pd.DataFrame(columns=["model", "feature", "gain"])
        gain = model.get_booster().get_score(importance_type="gain")
        df = pd.DataFrame({"feature": list(gain.keys()), "gain": list(gain.values())})
        if df.empty:
            return pd.DataFrame(columns=["model", "feature", "gain"])
        df["model"] = model_name
        return df[["model", "feature", "gain"]].sort_values("gain", ascending=False)

    frames = [_frame(bundle.win_model, "win_model")]
    for key, model in sorted(bundle.margin_quantile_models.items()):
        frames.append(_frame(model, f"margin_{key.lower()}"))
    for key, model in sorted(bundle.total_quantile_models.items()):
        frames.append(_frame(model, f"total_{key.lower()}"))
    if getattr(bundle, "cover_model", None) is not None:
        frames.append(_frame(bundle.cover_model, "cover_model"))
    if getattr(bundle, "over_model", None) is not None:
        frames.append(_frame(bundle.over_model, "over_model"))
    return pd.concat(frames, ignore_index=True)



def save_bundle(bundle: ModelBundle, artifact_dir: str | Path) -> None:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_dir / "model_bundle.joblib")
    (artifact_dir / "feature_columns.json").write_text(json.dumps(bundle.feature_columns, indent=2), encoding="utf-8")
    (artifact_dir / "fill_values.json").write_text(json.dumps(bundle.fill_values, indent=2), encoding="utf-8")
    bundle.config.save(artifact_dir / "train_config.json")



def load_bundle(artifact_dir: str | Path) -> ModelBundle:
    artifact_dir = Path(artifact_dir)
    bundle_path = artifact_dir / "model_bundle.joblib"
    if bundle_path.exists():
        return joblib.load(bundle_path)
    raise FileNotFoundError(f"Could not find bundle at {bundle_path}")
