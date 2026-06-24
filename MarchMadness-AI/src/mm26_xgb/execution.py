from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .ledger import LEDGER_COLUMNS
from .markets import break_even_prob, expected_value_per_unit, kelly_fraction
from .simulation import cover_prob_team_a, over_prob


CANDIDATE_COLUMNS = [
    "EventID",
    "CommenceTime",
    "TeamAID",
    "TeamBID",
    "TeamAName",
    "TeamBName",
    "Bookmaker",
    "BetType",
    "BetSide",
    "Line",
    "OddsAmerican",
    "ModelProb",
    "BreakEvenProb",
    "EdgeProb",
    "EVPerUnit",
    "KellyFraction",
    "StakeFraction",
    "StakeAmount",
    "MarketBookCount",
    "IsRecommended",
]



def build_consensus_matchups(flat_odds_resolved: pd.DataFrame) -> pd.DataFrame:
    if flat_odds_resolved.empty:
        return flat_odds_resolved.copy()

    required_cols = ["EventID", "CommenceTime", "Season", "TeamAID", "TeamBID", "TeamA", "TeamB"]
    missing_cols = [col for col in required_cols if col not in flat_odds_resolved.columns]
    if missing_cols:
        raise ValueError(f"flat_odds_resolved is missing required columns: {missing_cols}")

    work = flat_odds_resolved.copy()
    for col in ["Season", "TeamAID", "TeamBID"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    valid_mask = (
        work["EventID"].notna()
        & work["CommenceTime"].notna()
        & work["Season"].notna()
        & work["TeamAID"].notna()
        & work["TeamBID"].notna()
    )
    work = work.loc[valid_mask].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "EventID",
                "CommenceTime",
                "Season",
                "TeamAID",
                "TeamBID",
                "TeamA",
                "TeamB",
                "MarketBookCount",
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
            ]
        )

    group_cols = ["EventID", "CommenceTime", "Season", "TeamAID", "TeamBID", "TeamA", "TeamB"]
    rows = []

    for keys, group in work.groupby(group_cols, dropna=False):
        event_id, commence_time, season, team_aid, team_bid, team_a, team_b = keys
        row = {
            "EventID": event_id,
            "CommenceTime": commence_time,
            "Season": int(season),
            "TeamAID": int(team_aid),
            "TeamBID": int(team_bid),
            "TeamA": team_a,
            "TeamB": team_b,
            "MarketBookCount": int(group["Bookmaker"].nunique()) if "Bookmaker" in group.columns else 0,
        }

        ml_mask = group["MarketMoneylineTeamA"].notna() & group["MarketMoneylineTeamB"].notna()
        if ml_mask.any():
            fair_probs_a = []
            fair_probs_b = []
            for book_row in group.loc[ml_mask, ["MarketMoneylineTeamA", "MarketMoneylineTeamB"]].itertuples(index=False):
                a = float(book_row.MarketMoneylineTeamA)
                b = float(book_row.MarketMoneylineTeamB)
                pa = 100.0 / (a + 100.0) if a > 0 else abs(a) / (abs(a) + 100.0)
                pb = 100.0 / (b + 100.0) if b > 0 else abs(b) / (abs(b) + 100.0)
                total = pa + pb
                fair_probs_a.append(pa / total)
                fair_probs_b.append(pb / total)
            row["MarketProbTeamA"] = float(np.mean(fair_probs_a))
            row["MarketProbTeamB"] = float(np.mean(fair_probs_b))
            row["MarketMoneylineTeamA"] = float(group.loc[ml_mask, "MarketMoneylineTeamA"].mean())
            row["MarketMoneylineTeamB"] = float(group.loc[ml_mask, "MarketMoneylineTeamB"].mean())
        else:
            row["MarketProbTeamA"] = np.nan
            row["MarketProbTeamB"] = np.nan
            row["MarketMoneylineTeamA"] = np.nan
            row["MarketMoneylineTeamB"] = np.nan

        spread_mask = group["MarketSpreadTeamA"].notna()
        row["MarketSpreadTeamA"] = float(group.loc[spread_mask, "MarketSpreadTeamA"].mean()) if spread_mask.any() else np.nan
        row["MarketSpreadPriceA"] = float(group.loc[group["MarketSpreadPriceA"].notna(), "MarketSpreadPriceA"].mean()) if group["MarketSpreadPriceA"].notna().any() else np.nan
        row["MarketSpreadPriceB"] = float(group.loc[group["MarketSpreadPriceB"].notna(), "MarketSpreadPriceB"].mean()) if group["MarketSpreadPriceB"].notna().any() else np.nan

        total_mask = group["MarketTotal"].notna()
        row["MarketTotal"] = float(group.loc[total_mask, "MarketTotal"].mean()) if total_mask.any() else np.nan
        row["MarketOverPrice"] = float(group.loc[group["MarketOverPrice"].notna(), "MarketOverPrice"].mean()) if group["MarketOverPrice"].notna().any() else np.nan
        row["MarketUnderPrice"] = float(group.loc[group["MarketUnderPrice"].notna(), "MarketUnderPrice"].mean()) if group["MarketUnderPrice"].notna().any() else np.nan
        rows.append(row)

    return pd.DataFrame(rows)



def price_bookmaker_sides(
    flat_odds_resolved: pd.DataFrame,
    event_predictions: pd.DataFrame,
    bankroll: float = 1000.0,
    fractional_kelly: float = 0.25,
    max_stake_fraction: float = 0.02,
    min_moneyline_ev: float = 0.015,
    min_spread_ev: float = 0.010,
    min_total_ev: float = 0.010,
    min_edge_prob: float = 0.015,
    min_market_books: int = 1,
) -> pd.DataFrame:
    if flat_odds_resolved.empty or event_predictions.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    pred_cols = [
        "EventID",
        "CommenceTime",
        "TeamAID",
        "TeamBID",
        "WinProbTeamA",
        "WinProbTeamB",
        "PredMarginQ50",
        "PredTotalQ50",
        "MarginSigma",
        "TotalSigma",
        "MarketBookCount",
        "ModelProbCoverTeamA",
        "ModelProbCoverTeamB",
        "ModelProbOver",
        "ModelProbUnder",
    ]
    joined = flat_odds_resolved.merge(
        event_predictions[pred_cols].rename(columns={"MarketBookCount": "ConsensusBookCount"}),
        on=["EventID", "CommenceTime", "TeamAID", "TeamBID"],
        how="left",
    )

    rows: list[dict[str, object]] = []

    for row in joined.to_dict("records"):
        consensus_book_count = int(float(pd.to_numeric(pd.Series([row.get("ConsensusBookCount")]), errors="coerce").iloc[0] or 0.0))
        if consensus_book_count < min_market_books:
            continue

        event_base = {
            "EventID": row.get("EventID"),
            "CommenceTime": row.get("CommenceTime"),
            "TeamAID": int(float(row.get("TeamAID"))) if pd.notna(row.get("TeamAID")) else 0,
            "TeamBID": int(float(row.get("TeamBID"))) if pd.notna(row.get("TeamBID")) else 0,
            "TeamAName": row.get("TeamA"),
            "TeamBName": row.get("TeamB"),
            "Bookmaker": row.get("Bookmaker"),
            "MarketBookCount": consensus_book_count,
        }

        def add_candidate(bet_type: str, bet_side: str, line: float | None, odds: float | None, model_prob: float | None, ev_threshold: float) -> None:
            if odds is None or model_prob is None or not np.isfinite(float(odds)) or not np.isfinite(float(model_prob)):
                return
            odds_val = float(odds)
            prob_val = float(np.clip(model_prob, 1e-6, 1 - 1e-6))
            break_even = break_even_prob(odds_val)
            edge_prob = prob_val - break_even
            ev = expected_value_per_unit(prob_val, odds_val)
            kelly = kelly_fraction(prob_val, odds_val)
            stake_fraction = min(max_stake_fraction, fractional_kelly * kelly) if ev > 0 else 0.0
            stake_amount = bankroll * stake_fraction
            rows.append(
                {
                    **event_base,
                    "BetType": bet_type,
                    "BetSide": bet_side,
                    "Line": line,
                    "OddsAmerican": odds_val,
                    "ModelProb": prob_val,
                    "BreakEvenProb": break_even,
                    "EdgeProb": edge_prob,
                    "EVPerUnit": ev,
                    "KellyFraction": kelly,
                    "StakeFraction": stake_fraction,
                    "StakeAmount": stake_amount,
                    "IsRecommended": bool(ev >= ev_threshold and edge_prob >= min_edge_prob and stake_amount > 0),
                }
            )

        ml_a = row.get("MarketMoneylineTeamA")
        ml_b = row.get("MarketMoneylineTeamB")
        add_candidate("moneyline", "TeamA", np.nan, ml_a if pd.notna(ml_a) else None, row.get("WinProbTeamA"), min_moneyline_ev)
        add_candidate("moneyline", "TeamB", np.nan, ml_b if pd.notna(ml_b) else None, row.get("WinProbTeamB"), min_moneyline_ev)

        spread_line = row.get("MarketSpreadTeamA")
        spread_price_a = row.get("MarketSpreadPriceA")
        spread_price_b = row.get("MarketSpreadPriceB")
        if pd.notna(spread_line):
            model_cover_a = row.get("ModelProbCoverTeamA")
            model_cover_b = row.get("ModelProbCoverTeamB")
            if pd.isna(model_cover_a) and pd.notna(row.get("PredMarginQ50")) and pd.notna(row.get("MarginSigma")):
                model_cover_a = float(cover_prob_team_a(np.asarray([row["PredMarginQ50"]]), np.asarray([row["MarginSigma"]]), np.asarray([spread_line]))[0])
                model_cover_b = 1.0 - model_cover_a
            add_candidate("spread", "TeamA", float(spread_line), spread_price_a if pd.notna(spread_price_a) else None, model_cover_a if pd.notna(model_cover_a) else None, min_spread_ev)
            add_candidate("spread", "TeamB", -float(spread_line), spread_price_b if pd.notna(spread_price_b) else None, model_cover_b if pd.notna(model_cover_b) else None, min_spread_ev)

        total_line = row.get("MarketTotal")
        over_price = row.get("MarketOverPrice")
        under_price = row.get("MarketUnderPrice")
        if pd.notna(total_line):
            model_over = row.get("ModelProbOver")
            model_under = row.get("ModelProbUnder")
            if pd.isna(model_over) and pd.notna(row.get("PredTotalQ50")) and pd.notna(row.get("TotalSigma")):
                model_over = float(over_prob(np.asarray([row["PredTotalQ50"]]), np.asarray([row["TotalSigma"]]), np.asarray([total_line]))[0])
                model_under = 1.0 - model_over
            add_candidate("total", "Over", float(total_line), over_price if pd.notna(over_price) else None, model_over if pd.notna(model_over) else None, min_total_ev)
            add_candidate("total", "Under", float(total_line), under_price if pd.notna(under_price) else None, model_under if pd.notna(model_under) else None, min_total_ev)

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    recommended = candidates.loc[candidates["IsRecommended"]].copy()
    if not recommended.empty:
        keep_idx = (
            recommended.sort_values(["EVPerUnit", "EdgeProb", "StakeAmount"], ascending=[False, False, False])
            .groupby(["EventID", "Bookmaker", "BetType"], as_index=False)
            .head(1)
            .index
        )
        candidates["IsRecommended"] = candidates.index.isin(keep_idx)
    else:
        candidates["IsRecommended"] = False

    return candidates[CANDIDATE_COLUMNS].sort_values(["IsRecommended", "EVPerUnit", "StakeAmount"], ascending=[False, False, False]).reset_index(drop=True)



def candidates_to_ledger(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    open_rows = candidates.loc[candidates["IsRecommended"]].copy()
    if open_rows.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    open_rows["CreatedAtUTC"] = now
    open_rows["Status"] = "open"
    open_rows["FinalScoreTeamA"] = np.nan
    open_rows["FinalScoreTeamB"] = np.nan
    open_rows["Result"] = "pending"
    open_rows["Profit"] = 0.0
    return open_rows.rename(columns={"Line": "Line"})[LEDGER_COLUMNS].copy()
