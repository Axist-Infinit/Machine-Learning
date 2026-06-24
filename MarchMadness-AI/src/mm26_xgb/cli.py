from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import pandas as pd

from .attached_data import (
    attached_team_name_map,
    build_attached_team_features,
    build_attached_tournament_rows,
    load_attached_schedule,
)
from .bracket import simulate_bracket
from .config import TrainConfig
from .data import (
    load_aliases_csv,
    load_kaggle_data,
    resolve_team_identifier,
    team_lookup,
    team_name_map,
)
from .execution import build_consensus_matchups, candidates_to_ledger, price_bookmaker_sides
from .external_features import load_external_team_features, merge_external_team_features
from .features import build_team_features
from .ledger import append_open_bets, load_ledger, save_ledger, settle_ledger
from .markets import (
    fetch_odds_api_odds,
    fetch_odds_api_scores,
    flatten_odds_api_response,
    flatten_scores_api_response,
    load_api_key,
    load_market_csv,
    resolve_market_team_ids,
    write_json,
)
from .modeling import (
    export_feature_importance,
    fit_external_prior_models,
    load_bundle,
    make_tournament_training_rows,
    merge_market_history_into_rows,
    predict_matchups,
    save_bundle,
    train_from_raw_data,
    train_model_bundle,
    walk_forward_backtest,
)



def _build_team_features_from_bundle(
    data_dir: Path,
    bundle_config: TrainConfig,
    artifact_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[int, str]]:
    data = load_kaggle_data(data_dir)
    artifact_path = Path(artifact_dir) if artifact_dir is not None else None
    saved_team_features = artifact_path / "team_features.csv" if artifact_path is not None else None
    if saved_team_features is not None and saved_team_features.exists():
        team_features = pd.read_csv(saved_team_features, low_memory=False)
    else:
        team_features = build_team_features(
            regular_results=data.regular,
            seeds=data.seeds,
            massey=data.massey,
            cutoff_daynum=bundle_config.tourney_daynum_cutoff,
        )
        external_path = artifact_path / "external_team_features.csv" if artifact_path is not None else None
        if external_path is not None and external_path.exists():
            team_features = merge_external_team_features(team_features, pd.read_csv(external_path))
    names = team_name_map(data.teams)
    return team_features, names



def _load_optional_external_team_features(args: argparse.Namespace, data) -> tuple[pd.DataFrame | None, dict[str, object]]:
    win_loss_csv = getattr(args, "win_loss_csv", None)
    ats_csv = getattr(args, "ats_csv", None)
    ou_csv = getattr(args, "ou_csv", None)
    if not any([win_loss_csv, ats_csv, ou_csv]):
        return None, {}
    ext_bundle = load_external_team_features(
        teams_df=data.teams,
        spellings_df=data.spellings,
        target_season=int(getattr(args, "target_season", 2026)),
        win_loss_csv=win_loss_csv,
        ats_csv=ats_csv,
        ou_csv=ou_csv,
        aliases_csv=getattr(args, "team_aliases_csv", None),
        strict=bool(getattr(args, "strict_external_team_mapping", False)),
        default_season=getattr(args, "external_stats_season", None),
    )
    meta = {
        "coverage": ext_bundle.coverage,
        "unresolved": ext_bundle.unresolved,
    }
    return ext_bundle.frame, meta



def _attach_team_names(df: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    out = df.copy()
    for col in [c for c in out.columns if c.endswith("TeamID") or c in {"TeamAID", "TeamBID"}]:
        name_col = col.replace("ID", "Name")
        out[name_col] = out[col].map(names)
    return out



def _load_matchups_csv(path: Path, teams_df: pd.DataFrame, spellings_df: pd.DataFrame | None, aliases_csv: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    alias_lookup = team_lookup(teams_df, spellings_df, load_aliases_csv(aliases_csv, teams_df))

    id_base = ["Season", "TeamAID", "TeamBID"]
    if set(id_base).issubset(df.columns):
        out = df.copy()
        out["TeamAID"] = out["TeamAID"].astype(int)
        out["TeamBID"] = out["TeamBID"].astype(int)
        return out

    if {"Season", "TeamA", "TeamB"}.issubset(df.columns):
        out = df.copy()
        out["TeamAID"] = out["TeamA"].apply(lambda x: resolve_team_identifier(x, alias_lookup))
        out["TeamBID"] = out["TeamB"].apply(lambda x: resolve_team_identifier(x, alias_lookup))
        return out

    raise ValueError(
        "Input matchup CSV must contain either Season,TeamAID,TeamBID or Season,TeamA,TeamB columns."
    )



def _load_market_frame_for_training(
    path: str | None,
    teams_df: pd.DataFrame,
    spellings_df: pd.DataFrame | None,
    aliases_csv: str | None,
) -> pd.DataFrame | None:
    if path is None:
        return None
    df = load_market_csv(path)
    if df.empty:
        return df
    if {"TeamAID", "TeamBID"}.issubset(df.columns):
        return df
    if not {"Season", "TeamA", "TeamB"}.issubset(df.columns):
        raise ValueError(
            "Market history CSV must contain Season plus either TeamAID/TeamBID or TeamA/TeamB columns."
        )
    alias_lookup = team_lookup(teams_df, spellings_df, load_aliases_csv(aliases_csv, teams_df))
    return resolve_market_team_ids(df, alias_lookup)



def _resolve_flat_odds_with_team_ids(
    flat_odds: pd.DataFrame,
    teams_df: pd.DataFrame,
    spellings_df: pd.DataFrame | None,
    season: int,
    aliases_csv: str | None,
) -> pd.DataFrame:
    alias_lookup = team_lookup(teams_df, spellings_df, load_aliases_csv(aliases_csv, teams_df))
    resolved = resolve_market_team_ids(flat_odds, alias_lookup, season=season)
    if "Bookmaker" not in resolved.columns:
        resolved["Bookmaker"] = "consensus"
    return resolved



def cmd_train(args: argparse.Namespace) -> int:
    data = load_kaggle_data(args.data_dir)
    market_history = _load_market_frame_for_training(
        path=args.market_history_csv,
        teams_df=data.teams,
        spellings_df=data.spellings,
        aliases_csv=args.team_aliases_csv,
    )
    external_team_features, external_meta = _load_optional_external_team_features(args, data)
    config = TrainConfig(
        target_season=args.target_season,
        min_train_season=args.min_train_season,
        eval_start_season=args.eval_start_season,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        calibration_method=args.calibration_method,
        calibration_holdout_seasons=args.calibration_holdout_seasons,
    )
    config.n_estimators_cls = args.n_estimators_cls
    config.n_estimators_quantile = args.n_estimators_quantile
    config.learning_rate = args.learning_rate
    config.max_depth = args.max_depth
    config.min_child_weight = args.min_child_weight
    config.subsample = args.subsample
    config.colsample_bytree = args.colsample_bytree
    config.reg_alpha = args.reg_alpha
    config.reg_lambda = args.reg_lambda
    config.gamma = args.gamma
    if args.skip_backtest:
        team_features = build_team_features(
            regular_results=data.regular,
            seeds=data.seeds,
            massey=data.massey,
            cutoff_daynum=config.tourney_daynum_cutoff,
        )
        team_features = merge_external_team_features(team_features, external_team_features)
        tourney_rows = make_tournament_training_rows(data.tourney)
        tourney_rows = merge_market_history_into_rows(tourney_rows, market_history)
        train_rows = tourney_rows.loc[
            (tourney_rows["Season"] >= config.min_train_season)
            & (tourney_rows["Season"] < config.target_season)
        ].copy()
        bundle, training_frame = train_model_bundle(train_rows, team_features, config)
        ext_models, ext_cols, ext_fill, ext_summary = fit_external_prior_models(
            regular_results=data.regular,
            team_features=team_features,
            config=config,
        )
        bundle.external_prior_models = ext_models
        bundle.external_prior_feature_columns = ext_cols
        bundle.external_prior_fill_values = ext_fill
        bundle.external_prior_summary = ext_summary
        backtest = pd.DataFrame()
    else:
        bundle, team_features, training_frame, backtest = train_from_raw_data(
            regular_results=data.regular,
            tourney_results=data.tourney,
            seeds=data.seeds,
            massey=data.massey,
            config=config,
            market_history=market_history,
            extra_team_features=external_team_features,
        )

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_bundle(bundle, artifact_dir)
    team_features.to_csv(artifact_dir / "team_features.csv", index=False)
    if external_team_features is not None and not external_team_features.empty:
        external_team_features.to_csv(artifact_dir / "external_team_features.csv", index=False)
    training_frame.head(1000).to_csv(artifact_dir / "training_frame_head.csv", index=False)
    feature_importance = export_feature_importance(bundle)
    feature_importance.to_csv(artifact_dir / "feature_importance.csv", index=False)
    if not backtest.empty:
        backtest.to_csv(artifact_dir / "backtest_metrics.csv", index=False)
        summary = {
            "mean_brier": float(backtest["brier"].mean()),
            "mean_logloss": float(backtest["logloss"].mean()),
            "mean_accuracy": float(backtest["accuracy"].mean()),
            "mean_margin_mae": float(backtest["margin_mae"].mean()),
            "mean_total_mae": float(backtest["total_mae"].mean()),
            "seasons_tested": int(backtest["season"].nunique()),
            "market_history_rows": int(len(market_history)) if market_history is not None else 0,
            "calibration_method": config.calibration_method,
        }
    else:
        summary = {
            "warning": "Backtest did not run. Check available seasons in your data.",
            "market_history_rows": int(len(market_history)) if market_history is not None else 0,
            "calibration_method": config.calibration_method,
        }
    if external_meta:
        summary["external_team_features"] = external_meta
    if bundle.external_prior_summary:
        summary["external_prior"] = bundle.external_prior_summary
    (artifact_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved model artifacts to: {artifact_dir}")
    return 0



def cmd_predict_games(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.artifact_dir)
    data = load_kaggle_data(args.data_dir)
    matchups = _load_matchups_csv(Path(args.matchups_csv), data.teams, data.spellings, args.team_aliases_csv)
    team_features, names = _build_team_features_from_bundle(
        data_dir=Path(args.data_dir),
        bundle_config=bundle.config,
        artifact_dir=args.artifact_dir,
    )
    preds = predict_matchups(matchups=matchups, team_features=team_features, bundle=bundle)
    preds = _attach_team_names(preds, names)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")
    return 0



def cmd_predict_bracket(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.artifact_dir)
    data = load_kaggle_data(args.data_dir)
    if data.slots is None:
        raise FileNotFoundError("MNCAATourneySlots.csv is required to predict the bracket.")
    team_features, names = _build_team_features_from_bundle(
        data_dir=Path(args.data_dir),
        bundle_config=bundle.config,
        artifact_dir=args.artifact_dir,
    )
    bracket = simulate_bracket(
        season=args.season,
        seeds=data.seeds,
        slots=data.slots,
        team_features=team_features,
        bundle=bundle,
    )
    bracket = _attach_team_names(bracket, names)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bracket.to_csv(out_path, index=False)
    print(f"Saved bracket predictions to: {out_path}")
    return 0



def cmd_fetch_odds(args: argparse.Namespace) -> int:
    api_key = load_api_key(args.api_key)
    if not api_key:
        raise ValueError("A valid API key is required. Pass --api-key directly or as env:YOUR_VAR")

    payload, headers = fetch_odds_api_odds(
        api_key=api_key,
        sport=args.sport,
        regions=args.regions,
        markets=args.markets,
        bookmakers=args.bookmakers,
        historical_date=args.historical_date,
    )
    flat = flatten_odds_api_response(payload)

    if args.data_dir:
        data = load_kaggle_data(args.data_dir)
        flat = _resolve_flat_odds_with_team_ids(
            flat_odds=flat,
            teams_df=data.teams,
            spellings_df=data.spellings,
            season=args.season,
            aliases_csv=args.team_aliases_csv,
        )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat.to_csv(out_path, index=False)
    if args.output_json:
        write_json(args.output_json, payload)
    if args.consensus_output_csv:
        consensus = build_consensus_matchups(flat)
        Path(args.consensus_output_csv).parent.mkdir(parents=True, exist_ok=True)
        consensus.to_csv(args.consensus_output_csv, index=False)
    print(json.dumps({"rows": int(len(flat)), "headers": headers}, indent=2))
    print(f"Saved odds snapshot to: {out_path}")
    return 0



def cmd_scan_odds(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.artifact_dir)
    data = load_kaggle_data(args.data_dir)

    if args.odds_csv:
        flat = load_market_csv(args.odds_csv)
    else:
        api_key = load_api_key(args.api_key)
        if not api_key:
            raise ValueError("A valid API key is required unless --odds-csv is provided.")
        payload, _ = fetch_odds_api_odds(
            api_key=api_key,
            sport=args.sport,
            regions=args.regions,
            markets=args.markets,
            bookmakers=args.bookmakers,
        )
        flat = flatten_odds_api_response(payload)

    flat = _resolve_flat_odds_with_team_ids(
        flat_odds=flat,
        teams_df=data.teams,
        spellings_df=data.spellings,
        season=args.season,
        aliases_csv=args.team_aliases_csv,
    )

    team_features, _ = _build_team_features_from_bundle(
        data_dir=Path(args.data_dir),
        bundle_config=bundle.config,
        artifact_dir=args.artifact_dir,
    )

    consensus = build_consensus_matchups(flat)
    predictions = predict_matchups(matchups=consensus, team_features=team_features, bundle=bundle)
    candidates = price_bookmaker_sides(
        flat_odds_resolved=flat,
        event_predictions=predictions,
        bankroll=args.bankroll,
        fractional_kelly=args.fractional_kelly,
        max_stake_fraction=args.max_stake_fraction,
        min_moneyline_ev=args.min_moneyline_ev,
        min_spread_ev=args.min_spread_ev,
        min_total_ev=args.min_total_ev,
        min_edge_prob=args.min_edge_prob,
        min_market_books=args.min_market_books,
    )

    names = team_name_map(data.teams)
    if not candidates.empty:
        candidates["TeamAName"] = candidates["TeamAID"].map(names).fillna(candidates["TeamAName"])
        candidates["TeamBName"] = candidates["TeamBID"].map(names).fillna(candidates["TeamBName"])

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_path, index=False)

    if args.predictions_output_csv:
        pred_out = _attach_team_names(predictions, names)
        Path(args.predictions_output_csv).parent.mkdir(parents=True, exist_ok=True)
        pred_out.to_csv(args.predictions_output_csv, index=False)

    if args.ledger_csv:
        ledger = load_ledger(args.ledger_csv)
        new_rows = candidates_to_ledger(candidates)
        updated = append_open_bets(ledger, new_rows)
        save_ledger(updated, args.ledger_csv)

    summary = {
        "candidate_rows": int(len(candidates)),
        "recommended_bets": int(candidates["IsRecommended"].sum()) if not candidates.empty else 0,
        "events": int(candidates["EventID"].nunique()) if not candidates.empty else 0,
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved candidate bets to: {out_path}")
    return 0



def cmd_settle_ledger(args: argparse.Namespace) -> int:
    ledger = load_ledger(args.ledger_csv)
    if ledger.empty:
        print(json.dumps({"message": "Ledger is empty."}, indent=2))
        return 0

    if args.scores_csv:
        scores = pd.read_csv(args.scores_csv)
    else:
        api_key = load_api_key(args.api_key)
        if not api_key:
            raise ValueError("A valid API key is required unless --scores-csv is provided.")
        payload, _ = fetch_odds_api_scores(api_key=api_key, sport=args.sport, days_from=args.days_from)
        scores = flatten_scores_api_response(payload)
        if args.output_scores_csv:
            Path(args.output_scores_csv).parent.mkdir(parents=True, exist_ok=True)
            scores.to_csv(args.output_scores_csv, index=False)

    settled, summary = settle_ledger(ledger, scores)
    save_ledger(settled, args.ledger_csv)
    print(json.dumps(summary, indent=2))
    print(f"Updated ledger: {args.ledger_csv}")
    return 0



def _build_attached_config(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig(
        target_season=args.target_season,
        min_train_season=args.min_train_season,
        eval_start_season=args.eval_start_season,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        calibration_method=args.calibration_method,
        calibration_holdout_seasons=args.calibration_holdout_seasons,
    )
    config.n_estimators_cls = args.n_estimators_cls
    config.n_estimators_quantile = args.n_estimators_quantile
    config.learning_rate = args.learning_rate
    config.max_depth = args.max_depth
    config.min_child_weight = args.min_child_weight
    config.subsample = args.subsample
    config.colsample_bytree = args.colsample_bytree
    config.reg_alpha = args.reg_alpha
    config.reg_lambda = args.reg_lambda
    config.gamma = args.gamma
    return config


def cmd_train_attached(args: argparse.Namespace) -> int:
    team_features = build_attached_team_features(args.data_dir)
    tournament_rows = build_attached_tournament_rows(args.data_dir)
    config = _build_attached_config(args)

    train_rows = tournament_rows.loc[
        (tournament_rows["Season"] >= config.min_train_season)
        & (tournament_rows["Season"] < config.target_season)
    ].copy()
    if train_rows.empty:
        raise ValueError(
            "No attached tournament rows are available for the requested training window. "
            "Check --min-train-season and the seasons present in your custom data directory."
        )

    bundle, training_frame = train_model_bundle(train_rows, team_features, config)

    if args.skip_backtest:
        backtest = pd.DataFrame()
    else:
        backtest_config = TrainConfig(**config.to_dict())
        if args.backtest_max_seasons is not None and int(args.backtest_max_seasons) > 0:
            holdout_seasons = sorted(
                int(season)
                for season in tournament_rows["Season"].unique()
                if int(season) >= backtest_config.eval_start_season and int(season) < backtest_config.target_season
            )
            if len(holdout_seasons) > int(args.backtest_max_seasons):
                backtest_config.eval_start_season = int(holdout_seasons[-int(args.backtest_max_seasons)])
        backtest = walk_forward_backtest(tournament_rows, team_features, backtest_config)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_bundle(bundle, artifact_dir)
    team_features.to_csv(artifact_dir / "team_features.csv", index=False)
    training_frame.head(1000).to_csv(artifact_dir / "training_frame_head.csv", index=False)
    feature_importance = export_feature_importance(bundle)
    feature_importance.to_csv(artifact_dir / "feature_importance.csv", index=False)
    if not backtest.empty:
        backtest.to_csv(artifact_dir / "backtest_metrics.csv", index=False)
        summary = {
            "training_mode": "attached_custom",
            "mean_brier": float(backtest["brier"].mean()),
            "mean_logloss": float(backtest["logloss"].mean()),
            "mean_accuracy": float(backtest["accuracy"].mean()),
            "mean_margin_mae": float(backtest["margin_mae"].mean()),
            "mean_total_mae": float(backtest["total_mae"].mean()),
            "seasons_tested": int(backtest["season"].nunique()),
            "train_rows": int(len(train_rows)),
            "team_feature_rows": int(len(team_features)),
            "available_seasons": sorted(int(x) for x in team_features["Season"].dropna().unique().tolist()),
            "calibration_method": config.calibration_method,
        }
    else:
        summary = {
            "training_mode": "attached_custom",
            "warning": "Backtest did not run. Check available seasons in your data.",
            "train_rows": int(len(train_rows)),
            "team_feature_rows": int(len(team_features)),
            "available_seasons": sorted(int(x) for x in team_features["Season"].dropna().unique().tolist()),
            "calibration_method": config.calibration_method,
        }
    (artifact_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved attached-data model artifacts to: {artifact_dir}")
    return 0


def cmd_predict_schedule(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.artifact_dir)
    team_features = build_attached_team_features(args.data_dir)
    schedule = load_attached_schedule(args.schedule_csv, team_features, data_dir=args.data_dir)
    preds = predict_matchups(matchups=schedule, team_features=team_features, bundle=bundle)

    names = attached_team_name_map(team_features)
    preds = _attach_team_names(preds, names)
    output = schedule.merge(preds, on=["Season", "TeamAID", "TeamBID"], how="left")
    if "TeamA" not in output.columns and "TeamAName" in output.columns:
        output["TeamA"] = output["TeamAName"]
    if "TeamB" not in output.columns and "TeamBName" in output.columns:
        output["TeamB"] = output["TeamBName"]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    print(f"Saved schedule predictions to: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="March Madness 2026 XGBoost trading model")
    sub = parser.add_subparsers(dest="command", required=True)

    train_attached = sub.add_parser("train-attached", help="Train on the uploaded multi-source attached tournament dataset.")
    train_attached.add_argument("--data-dir", required=True, help="Folder containing the extracted attached CSV files.")
    train_attached.add_argument("--artifact-dir", required=True, help="Output folder for saved model files.")
    train_attached.add_argument("--target-season", type=int, default=2026)
    train_attached.add_argument("--min-train-season", type=int, default=2008)
    train_attached.add_argument("--eval-start-season", type=int, default=2016)
    train_attached.add_argument("--random-state", type=int, default=42)
    train_attached.add_argument("--n-jobs", type=int, default=4)
    train_attached.add_argument("--calibration-method", default="isotonic", choices=["none", "isotonic", "sigmoid"])
    train_attached.add_argument("--calibration-holdout-seasons", type=int, default=3)
    train_attached.add_argument("--n-estimators-cls", type=int, default=120)
    train_attached.add_argument("--n-estimators-quantile", type=int, default=120)
    train_attached.add_argument("--learning-rate", type=float, default=0.05)
    train_attached.add_argument("--max-depth", type=int, default=3)
    train_attached.add_argument("--min-child-weight", type=float, default=4.0)
    train_attached.add_argument("--subsample", type=float, default=0.8)
    train_attached.add_argument("--colsample-bytree", type=float, default=0.6)
    train_attached.add_argument("--reg-alpha", type=float, default=0.3)
    train_attached.add_argument("--reg-lambda", type=float, default=4.0)
    train_attached.add_argument("--gamma", type=float, default=0.05)
    train_attached.add_argument("--skip-backtest", action="store_true", help="Train the final bundle without running walk-forward backtests.")
    train_attached.add_argument("--backtest-max-seasons", type=int, default=3, help="Maximum number of most-recent holdout seasons to evaluate in the walk-forward backtest.")
    train_attached.set_defaults(func=cmd_train_attached)

    predict_schedule = sub.add_parser("predict-schedule", help="Predict a future round schedule from the attached custom data schema.")
    predict_schedule.add_argument("--data-dir", required=True, help="Folder containing the extracted attached CSV files, ideally with 2026 rows added.")
    predict_schedule.add_argument("--artifact-dir", required=True, help="Folder containing a saved attached-data model bundle.")
    predict_schedule.add_argument("--schedule-csv", required=True, help="CSV with Season plus TeamA/TeamB or TeamAID/TeamBID. CurrentRound is recommended.")
    predict_schedule.add_argument("--output-csv", required=True)
    predict_schedule.set_defaults(func=cmd_predict_schedule)

    train = sub.add_parser("train", help="Train the calibrated market-aware model stack.")
    train.add_argument("--data-dir", required=True, help="Folder containing Kaggle CSV files.")
    train.add_argument("--artifact-dir", required=True, help="Output folder for saved model files.")
    train.add_argument("--target-season", type=int, default=2026)
    train.add_argument("--min-train-season", type=int, default=2003)
    train.add_argument("--eval-start-season", type=int, default=2018)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--n-jobs", type=int, default=4)
    train.add_argument("--market-history-csv", default=None, help="Optional consensus or raw historical odds CSV.")
    train.add_argument("--team-aliases-csv", default=None, help="Optional alias CSV for external team names.")
    train.add_argument("--win-loss-csv", default=None, help="Optional team-level win/loss metrics CSV (e.g. win_loss.csv).")
    train.add_argument("--ats-csv", default=None, help="Optional team-level against-the-spread metrics CSV (e.g. ats.csv).")
    train.add_argument("--ou-csv", default=None, help="Optional team-level over/under metrics CSV (e.g. ou.csv).")
    train.add_argument("--external-stats-season", type=int, default=None, help="Season to assign when external team stats files omit Season.")
    train.add_argument("--strict-external-team-mapping", action="store_true", help="Fail if any external team names cannot be resolved.")
    train.add_argument("--calibration-method", default="isotonic", choices=["none", "isotonic", "sigmoid"])
    train.add_argument("--n-estimators-cls", type=int, default=220)
    train.add_argument("--n-estimators-quantile", type=int, default=180)
    train.add_argument("--learning-rate", type=float, default=0.05)
    train.add_argument("--max-depth", type=int, default=4)
    train.add_argument("--min-child-weight", type=float, default=4.0)
    train.add_argument("--subsample", type=float, default=0.85)
    train.add_argument("--colsample-bytree", type=float, default=0.80)
    train.add_argument("--reg-alpha", type=float, default=0.15)
    train.add_argument("--reg-lambda", type=float, default=2.5)
    train.add_argument("--gamma", type=float, default=0.05)
    train.add_argument("--calibration-holdout-seasons", type=int, default=3)
    train.add_argument("--skip-backtest", action="store_true", help="Skip walk-forward backtesting for faster final-model training.")
    train.set_defaults(func=cmd_train)

    pred_games = sub.add_parser("predict-games", help="Predict any list of matchups.")
    pred_games.add_argument("--data-dir", required=True)
    pred_games.add_argument("--artifact-dir", required=True)
    pred_games.add_argument("--matchups-csv", required=True)
    pred_games.add_argument("--output-csv", required=True)
    pred_games.add_argument("--team-aliases-csv", default=None)
    pred_games.set_defaults(func=cmd_predict_games)

    pred_bracket = sub.add_parser("predict-bracket", help="Predict the actual tournament bracket from seeds and slots.")
    pred_bracket.add_argument("--data-dir", required=True)
    pred_bracket.add_argument("--artifact-dir", required=True)
    pred_bracket.add_argument("--season", required=True, type=int)
    pred_bracket.add_argument("--output-csv", required=True)
    pred_bracket.set_defaults(func=cmd_predict_bracket)

    fetch = sub.add_parser("fetch-odds", help="Fetch live or historical odds snapshots from The Odds API.")
    fetch.add_argument("--api-key", required=True, help="API key or env:VAR_NAME")
    fetch.add_argument("--output-csv", required=True)
    fetch.add_argument("--output-json", default=None)
    fetch.add_argument("--consensus-output-csv", default=None)
    fetch.add_argument("--sport", default="basketball_ncaab")
    fetch.add_argument("--regions", default="us")
    fetch.add_argument("--markets", default="h2h,spreads,totals")
    fetch.add_argument("--bookmakers", default=None)
    fetch.add_argument("--historical-date", default=None, help="ISO timestamp for historical snapshot, e.g. 2025-03-20T18:00:00Z")
    fetch.add_argument("--data-dir", default=None, help="Optional Kaggle data dir to resolve TeamAID/TeamBID.")
    fetch.add_argument("--team-aliases-csv", default=None)
    fetch.add_argument("--season", type=int, default=2026)
    fetch.set_defaults(func=cmd_fetch_odds)

    scan = sub.add_parser("scan-odds", help="Score bookmaker lines and output bet candidates.")
    scan.add_argument("--data-dir", required=True)
    scan.add_argument("--artifact-dir", required=True)
    scan.add_argument("--output-csv", required=True)
    scan.add_argument("--odds-csv", default=None, help="Optional normalized raw odds CSV. If omitted, live odds are fetched.")
    scan.add_argument("--api-key", default=None, help="API key or env:VAR_NAME for live fetch.")
    scan.add_argument("--sport", default="basketball_ncaab")
    scan.add_argument("--regions", default="us")
    scan.add_argument("--markets", default="h2h,spreads,totals")
    scan.add_argument("--bookmakers", default=None)
    scan.add_argument("--season", type=int, default=2026)
    scan.add_argument("--team-aliases-csv", default=None)
    scan.add_argument("--bankroll", type=float, default=1000.0)
    scan.add_argument("--fractional-kelly", type=float, default=0.25)
    scan.add_argument("--max-stake-fraction", type=float, default=0.02)
    scan.add_argument("--min-moneyline-ev", type=float, default=0.015)
    scan.add_argument("--min-spread-ev", type=float, default=0.010)
    scan.add_argument("--min-total-ev", type=float, default=0.010)
    scan.add_argument("--min-edge-prob", type=float, default=0.015)
    scan.add_argument("--min-market-books", type=int, default=1)
    scan.add_argument("--predictions-output-csv", default=None)
    scan.add_argument("--ledger-csv", default=None)
    scan.set_defaults(func=cmd_scan_odds)

    settle = sub.add_parser("settle-ledger", help="Settle open bets from the ledger using scores data.")
    settle.add_argument("--ledger-csv", required=True)
    settle.add_argument("--scores-csv", default=None)
    settle.add_argument("--api-key", default=None, help="API key or env:VAR_NAME if fetching scores live.")
    settle.add_argument("--sport", default="basketball_ncaab")
    settle.add_argument("--days-from", type=int, default=3)
    settle.add_argument("--output-scores-csv", default=None)
    settle.set_defaults(func=cmd_settle_ledger)

    return parser



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))



def train_entry() -> int:
    import sys

    return main(["train", *sys.argv[1:]])



def predict_games_entry() -> int:
    import sys

    return main(["predict-games", *sys.argv[1:]])



def predict_bracket_entry() -> int:
    import sys

    return main(["predict-bracket", *sys.argv[1:]])



def fetch_odds_entry() -> int:
    import sys

    return main(["fetch-odds", *sys.argv[1:]])



def scan_odds_entry() -> int:
    import sys

    return main(["scan-odds", *sys.argv[1:]])



def settle_ledger_entry() -> int:
    import sys

    return main(["settle-ledger", *sys.argv[1:]])


def train_attached_entry() -> int:
    import sys

    return main(["train-attached", *sys.argv[1:]])


def predict_schedule_entry() -> int:
    import sys

    return main(["predict-schedule", *sys.argv[1:]])
