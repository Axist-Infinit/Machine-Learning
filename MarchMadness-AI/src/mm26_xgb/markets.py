from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json
import math

import numpy as np
import pandas as pd
import requests

from .data import normalize_team_name, resolve_team_identifier


ODDS_API_HOST = "https://api.the-odds-api.com"
FEATURED_MARKETS = "h2h,spreads,totals"

RAW_ODDS_COLUMNS = [
    "EventID",
    "CommenceTime",
    "Bookmaker",
    "BookTitle",
    "BookLastUpdate",
    "TeamA",
    "TeamB",
    "MarketMoneylineTeamA",
    "MarketMoneylineTeamB",
    "MarketSpreadTeamA",
    "MarketSpreadPriceA",
    "MarketSpreadPriceB",
    "MarketTotal",
    "MarketOverPrice",
    "MarketUnderPrice",
]

CONSENSUS_MARKET_COLUMNS = [
    "EventID",
    "CommenceTime",
    "TeamA",
    "TeamB",
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



def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)



def american_to_implied_prob(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)



def fair_prob_from_two_way_moneyline(odds_a: float, odds_b: float) -> tuple[float, float]:
    p_a = american_to_implied_prob(odds_a)
    p_b = american_to_implied_prob(odds_b)
    total = p_a + p_b
    if total <= 0:
        return 0.5, 0.5
    return p_a / total, p_b / total



def fair_prob_to_american(prob: float) -> int:
    p = float(np.clip(prob, 1e-6, 1 - 1e-6))
    if p >= 0.5:
        odds = -(100.0 * p / (1.0 - p))
    else:
        odds = 100.0 * (1.0 - p) / p
    return int(round(odds))



def decimal_to_american(decimal_odds: float) -> int:
    dec = float(decimal_odds)
    if dec <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100.0))
    return int(round(-100.0 / (dec - 1.0)))



def expected_value_per_unit(win_prob: float, american_odds: float) -> float:
    dec = american_to_decimal(american_odds)
    profit_multiple = dec - 1.0
    p = float(np.clip(win_prob, 0.0, 1.0))
    return p * profit_multiple - (1.0 - p)



def break_even_prob(american_odds: float) -> float:
    return american_to_implied_prob(american_odds)



def kelly_fraction(win_prob: float, american_odds: float) -> float:
    dec = american_to_decimal(american_odds)
    b = dec - 1.0
    p = float(np.clip(win_prob, 0.0, 1.0))
    q = 1.0 - p
    raw = (b * p - q) / b
    return max(0.0, raw)



def american_profit(stake: float, american_odds: float) -> float:
    dec = american_to_decimal(american_odds)
    return float(stake) * (dec - 1.0)



def load_api_key(value: str | None) -> str | None:
    if value is None:
        return None
    if value.startswith("env:"):
        import os

        return os.getenv(value.split(":", 1)[1])
    return value



def odds_api_get(path: str, api_key: str, params: dict[str, Any] | None = None, timeout: int = 20) -> tuple[Any, dict[str, str]]:
    request_params = dict(params or {})
    request_params["apiKey"] = api_key
    response = requests.get(f"{ODDS_API_HOST}{path}", params=request_params, timeout=timeout)
    response.raise_for_status()
    return response.json(), dict(response.headers)



def fetch_odds_api_odds(
    api_key: str,
    sport: str = "basketball_ncaab",
    regions: str = "us",
    markets: str = FEATURED_MARKETS,
    bookmakers: str | None = None,
    odds_format: str = "american",
    historical_date: str | None = None,
    timeout: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    params: dict[str, Any] = {
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers

    if historical_date:
        payload, headers = odds_api_get(
            path=f"/v4/historical/sports/{sport}/odds",
            api_key=api_key,
            params={**params, "date": historical_date},
            timeout=timeout,
        )
        data = payload.get("data", []) if isinstance(payload, dict) else []
        return list(data), headers

    payload, headers = odds_api_get(
        path=f"/v4/sports/{sport}/odds",
        api_key=api_key,
        params=params,
        timeout=timeout,
    )
    return list(payload), headers



def fetch_odds_api_scores(
    api_key: str,
    sport: str = "basketball_ncaab",
    days_from: int = 3,
    timeout: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    payload, headers = odds_api_get(
        path=f"/v4/sports/{sport}/scores",
        api_key=api_key,
        params={"daysFrom": days_from, "dateFormat": "iso"},
        timeout=timeout,
    )
    return list(payload), headers



def write_json(path: str | Path, payload: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")



def flatten_odds_api_response(payload: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for event in payload:
        event_id = str(event.get("id", ""))
        commence_time = event.get("commence_time")
        team_a = str(event.get("home_team", "")).strip()
        team_b = str(event.get("away_team", "")).strip()
        bookmakers = event.get("bookmakers", []) or []

        for book in bookmakers:
            row: dict[str, Any] = {
                "EventID": event_id,
                "CommenceTime": commence_time,
                "Bookmaker": book.get("key"),
                "BookTitle": book.get("title"),
                "BookLastUpdate": book.get("last_update"),
                "TeamA": team_a,
                "TeamB": team_b,
                "MarketMoneylineTeamA": np.nan,
                "MarketMoneylineTeamB": np.nan,
                "MarketSpreadTeamA": np.nan,
                "MarketSpreadPriceA": np.nan,
                "MarketSpreadPriceB": np.nan,
                "MarketTotal": np.nan,
                "MarketOverPrice": np.nan,
                "MarketUnderPrice": np.nan,
            }

            for market in book.get("markets", []) or []:
                outcomes = market.get("outcomes", []) or []
                key = market.get("key")

                if key == "h2h":
                    for outcome in outcomes:
                        name = str(outcome.get("name", "")).strip()
                        price = _coerce_float(outcome.get("price"))
                        if name == team_a:
                            row["MarketMoneylineTeamA"] = price
                        elif name == team_b:
                            row["MarketMoneylineTeamB"] = price

                elif key == "spreads":
                    for outcome in outcomes:
                        name = str(outcome.get("name", "")).strip()
                        point = _coerce_float(outcome.get("point"))
                        price = _coerce_float(outcome.get("price"))
                        if name == team_a:
                            row["MarketSpreadTeamA"] = point
                            row["MarketSpreadPriceA"] = price
                        elif name == team_b:
                            row["MarketSpreadPriceB"] = price
                            if point is not None and pd.isna(row["MarketSpreadTeamA"]):
                                row["MarketSpreadTeamA"] = -float(point)

                elif key == "totals":
                    for outcome in outcomes:
                        name = str(outcome.get("name", "")).strip().lower()
                        point = _coerce_float(outcome.get("point"))
                        price = _coerce_float(outcome.get("price"))
                        if point is not None and pd.isna(row["MarketTotal"]):
                            row["MarketTotal"] = point
                        if name == "over":
                            row["MarketOverPrice"] = price
                        elif name == "under":
                            row["MarketUnderPrice"] = price

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=RAW_ODDS_COLUMNS)
    for col in RAW_ODDS_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[RAW_ODDS_COLUMNS].copy()



def flatten_scores_api_response(payload: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in payload:
        team_a = str(event.get("home_team", "")).strip()
        team_b = str(event.get("away_team", "")).strip()
        score_a = np.nan
        score_b = np.nan
        for item in event.get("scores", []) or []:
            name = str(item.get("name", "")).strip()
            score = _coerce_float(item.get("score"))
            if name == team_a:
                score_a = score
            elif name == team_b:
                score_b = score
        rows.append(
            {
                "EventID": str(event.get("id", "")),
                "CommenceTime": event.get("commence_time"),
                "TeamA": team_a,
                "TeamB": team_b,
                "ScoreTeamA": score_a,
                "ScoreTeamB": score_b,
                "Completed": bool(event.get("completed", False)),
                "LastUpdate": event.get("last_update"),
            }
        )
    return pd.DataFrame(rows)



def compute_hold_two_way(price_a: float | None, price_b: float | None) -> float | None:
    if price_a is None or price_b is None:
        return None
    return float(american_to_implied_prob(price_a) + american_to_implied_prob(price_b) - 1.0)



def build_event_consensus(flat_odds: pd.DataFrame) -> pd.DataFrame:
    if flat_odds.empty:
        return pd.DataFrame(columns=CONSENSUS_MARKET_COLUMNS)

    rows: list[dict[str, Any]] = []
    group_cols = ["EventID", "CommenceTime", "TeamA", "TeamB"]

    for keys, group in flat_odds.groupby(group_cols, dropna=False):
        event_id, commence_time, team_a, team_b = keys
        moneyline_probs: list[tuple[float, float]] = []
        moneyline_team_a: list[float] = []
        moneyline_team_b: list[float] = []
        spread_points: list[float] = []
        spread_price_a: list[float] = []
        spread_price_b: list[float] = []
        totals: list[float] = []
        total_over: list[float] = []
        total_under: list[float] = []
        ml_holds: list[float] = []
        spread_holds: list[float] = []
        total_holds: list[float] = []

        for row in group.to_dict("records"):
            ml_a = _coerce_float(row.get("MarketMoneylineTeamA"))
            ml_b = _coerce_float(row.get("MarketMoneylineTeamB"))
            if ml_a is not None and ml_b is not None:
                moneyline_probs.append(fair_prob_from_two_way_moneyline(ml_a, ml_b))
                moneyline_team_a.append(ml_a)
                moneyline_team_b.append(ml_b)
                hold = compute_hold_two_way(ml_a, ml_b)
                if hold is not None:
                    ml_holds.append(hold)

            spread = _coerce_float(row.get("MarketSpreadTeamA"))
            spread_a = _coerce_float(row.get("MarketSpreadPriceA"))
            spread_b = _coerce_float(row.get("MarketSpreadPriceB"))
            if spread is not None:
                spread_points.append(spread)
            if spread_a is not None:
                spread_price_a.append(spread_a)
            if spread_b is not None:
                spread_price_b.append(spread_b)
            hold = compute_hold_two_way(spread_a, spread_b)
            if hold is not None:
                spread_holds.append(hold)

            total = _coerce_float(row.get("MarketTotal"))
            over = _coerce_float(row.get("MarketOverPrice"))
            under = _coerce_float(row.get("MarketUnderPrice"))
            if total is not None:
                totals.append(total)
            if over is not None:
                total_over.append(over)
            if under is not None:
                total_under.append(under)
            hold = compute_hold_two_way(over, under)
            if hold is not None:
                total_holds.append(hold)

        if moneyline_probs:
            avg_prob_a = float(np.mean([x[0] for x in moneyline_probs]))
            avg_prob_b = float(np.mean([x[1] for x in moneyline_probs]))
            avg_ml_a = float(np.mean(moneyline_team_a))
            avg_ml_b = float(np.mean(moneyline_team_b))
        else:
            avg_prob_a = np.nan
            avg_prob_b = np.nan
            avg_ml_a = np.nan
            avg_ml_b = np.nan

        rows.append(
            {
                "EventID": event_id,
                "CommenceTime": commence_time,
                "TeamA": team_a,
                "TeamB": team_b,
                "MarketProbTeamA": avg_prob_a,
                "MarketProbTeamB": avg_prob_b,
                "MarketMoneylineTeamA": avg_ml_a,
                "MarketMoneylineTeamB": avg_ml_b,
                "MarketSpreadTeamA": float(np.mean(spread_points)) if spread_points else np.nan,
                "MarketSpreadPriceA": float(np.mean(spread_price_a)) if spread_price_a else np.nan,
                "MarketSpreadPriceB": float(np.mean(spread_price_b)) if spread_price_b else np.nan,
                "MarketTotal": float(np.mean(totals)) if totals else np.nan,
                "MarketOverPrice": float(np.mean(total_over)) if total_over else np.nan,
                "MarketUnderPrice": float(np.mean(total_under)) if total_under else np.nan,
                "MarketBookCount": int(group["Bookmaker"].nunique()),
                "MoneylineHold": float(np.mean(ml_holds)) if ml_holds else np.nan,
                "SpreadHold": float(np.mean(spread_holds)) if spread_holds else np.nan,
                "TotalHold": float(np.mean(total_holds)) if total_holds else np.nan,
            }
        )

    return pd.DataFrame(rows)[CONSENSUS_MARKET_COLUMNS].copy()



def normalize_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in [
        "MarketMoneylineTeamA",
        "MarketMoneylineTeamB",
        "MarketProbTeamA",
        "MarketProbTeamB",
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
    ]:
        if col not in out.columns:
            out[col] = np.nan

    if "MarketProbTeamA" not in out.columns or out["MarketProbTeamA"].isna().all():
        if {"MarketMoneylineTeamA", "MarketMoneylineTeamB"}.issubset(out.columns):
            mask = out["MarketMoneylineTeamA"].notna() & out["MarketMoneylineTeamB"].notna()
            if mask.any():
                fair = out.loc[mask, ["MarketMoneylineTeamA", "MarketMoneylineTeamB"]].apply(
                    lambda row: fair_prob_from_two_way_moneyline(row.iloc[0], row.iloc[1]), axis=1
                )
                out.loc[mask, "MarketProbTeamA"] = [p[0] for p in fair]
                out.loc[mask, "MarketProbTeamB"] = [p[1] for p in fair]
    if "MarketProbTeamB" not in out.columns:
        out["MarketProbTeamB"] = 1.0 - out["MarketProbTeamA"]
    else:
        mask = out["MarketProbTeamA"].notna() & out["MarketProbTeamB"].isna()
        out.loc[mask, "MarketProbTeamB"] = 1.0 - out.loc[mask, "MarketProbTeamA"]

    if "MoneylineHold" not in out.columns or out["MoneylineHold"].isna().all():
        if {"MarketMoneylineTeamA", "MarketMoneylineTeamB"}.issubset(out.columns):
            out["MoneylineHold"] = out.apply(
                lambda row: compute_hold_two_way(
                    _coerce_float(row.get("MarketMoneylineTeamA")),
                    _coerce_float(row.get("MarketMoneylineTeamB")),
                ),
                axis=1,
            )

    if "SpreadHold" not in out.columns or out["SpreadHold"].isna().all():
        out["SpreadHold"] = out.apply(
            lambda row: compute_hold_two_way(
                _coerce_float(row.get("MarketSpreadPriceA")),
                _coerce_float(row.get("MarketSpreadPriceB")),
            ),
            axis=1,
        )

    if "TotalHold" not in out.columns or out["TotalHold"].isna().all():
        out["TotalHold"] = out.apply(
            lambda row: compute_hold_two_way(
                _coerce_float(row.get("MarketOverPrice")),
                _coerce_float(row.get("MarketUnderPrice")),
            ),
            axis=1,
        )

    return out



def add_reverse_market_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = normalize_market_frame(df).copy()
    reverse = out.copy()
    if {"TeamAID", "TeamBID"}.issubset(reverse.columns):
        reverse[["TeamAID", "TeamBID"]] = reverse[["TeamBID", "TeamAID"]]
    if {"TeamA", "TeamB"}.issubset(reverse.columns):
        reverse[["TeamA", "TeamB"]] = reverse[["TeamB", "TeamA"]]
    if {"MarketProbTeamA", "MarketProbTeamB"}.issubset(reverse.columns):
        reverse[["MarketProbTeamA", "MarketProbTeamB"]] = reverse[["MarketProbTeamB", "MarketProbTeamA"]]
    if {"MarketMoneylineTeamA", "MarketMoneylineTeamB"}.issubset(reverse.columns):
        reverse[["MarketMoneylineTeamA", "MarketMoneylineTeamB"]] = reverse[["MarketMoneylineTeamB", "MarketMoneylineTeamA"]]
    if "MarketSpreadTeamA" in reverse.columns:
        reverse["MarketSpreadTeamA"] = -reverse["MarketSpreadTeamA"]
    if {"MarketSpreadPriceA", "MarketSpreadPriceB"}.issubset(reverse.columns):
        reverse[["MarketSpreadPriceA", "MarketSpreadPriceB"]] = reverse[["MarketSpreadPriceB", "MarketSpreadPriceA"]]

    out["Orientation"] = "forward"
    reverse["Orientation"] = "reverse"
    combined = pd.concat([out, reverse], ignore_index=True)
    return combined.drop_duplicates(subset=[c for c in ["Season", "TeamAID", "TeamBID", "Orientation"] if c in combined.columns])



def resolve_market_team_ids(
    df: pd.DataFrame,
    team_alias_lookup: dict[str, int],
    season: int | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    required = {"TeamA", "TeamB"}
    if not required.issubset(df.columns):
        raise ValueError(f"Market frame must contain columns {sorted(required)} to resolve names.")

    out = df.copy()
    out["TeamAID"] = out["TeamA"].apply(lambda x: resolve_team_identifier(str(x), team_alias_lookup))
    out["TeamBID"] = out["TeamB"].apply(lambda x: resolve_team_identifier(str(x), team_alias_lookup))
    if season is not None and "Season" not in out.columns:
        out["Season"] = int(season)
    return out



def load_market_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_market_frame(df)



def aggregate_market_history_to_consensus(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if {"Bookmaker", "TeamA", "TeamB", "EventID"}.issubset(df.columns):
        consensus = build_event_consensus(df)
        if "Season" in df.columns:
            season_cols = [c for c in ["EventID", "Season"] if c in df.columns]
            consensus = consensus.merge(df[season_cols].drop_duplicates(), on="EventID", how="left")
        if {"TeamAID", "TeamBID"}.issubset(df.columns):
            id_cols = [c for c in ["EventID", "TeamAID", "TeamBID"] if c in df.columns]
            consensus = consensus.merge(df[id_cols].drop_duplicates(), on="EventID", how="left")
        return normalize_market_frame(consensus)
    return normalize_market_frame(df)
