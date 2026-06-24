from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .markets import american_profit


LEDGER_COLUMNS = [
    "CreatedAtUTC",
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
    "Status",
    "FinalScoreTeamA",
    "FinalScoreTeamB",
    "Result",
    "Profit",
]



def load_ledger(path: str | Path) -> pd.DataFrame:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    df = pd.read_csv(ledger_path)
    for col in LEDGER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[LEDGER_COLUMNS].copy()



def save_ledger(df: pd.DataFrame, path: str | Path) -> None:
    out = df.copy()
    for col in LEDGER_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    out[LEDGER_COLUMNS].to_csv(ledger_path, index=False)



def append_open_bets(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if new_rows.empty:
        return existing.copy()
    if existing.empty:
        combined = new_rows[LEDGER_COLUMNS].copy()
    else:
        combined = pd.concat([existing[LEDGER_COLUMNS], new_rows[LEDGER_COLUMNS]], ignore_index=True)
    dedupe_cols = ["EventID", "Bookmaker", "BetType", "BetSide", "Line", "OddsAmerican", "StakeAmount"]
    return combined.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)



def settle_ledger(ledger: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if ledger.empty:
        return ledger.copy(), {"open_bets": 0, "settled_bets": 0, "profit": 0.0}

    score_lookup_event = {}
    if not scores.empty and "EventID" in scores.columns:
        for row in scores.to_dict("records"):
            score_lookup_event[str(row.get("EventID", ""))] = row

    out_rows: list[dict[str, Any]] = []
    settled_count = 0

    for row in ledger.to_dict("records"):
        if str(row.get("Status", "open")) == "settled":
            out_rows.append(row)
            continue

        event = score_lookup_event.get(str(row.get("EventID", "")))
        if not event or not bool(event.get("Completed", False)):
            out_rows.append(row)
            continue

        score_a = float(event.get("ScoreTeamA"))
        score_b = float(event.get("ScoreTeamB"))
        margin = score_a - score_b
        total = score_a + score_b
        bet_type = str(row.get("BetType", ""))
        bet_side = str(row.get("BetSide", ""))
        line = float(row.get("Line", 0.0)) if pd.notna(row.get("Line")) else 0.0
        odds = float(row.get("OddsAmerican", np.nan))
        stake = float(row.get("StakeAmount", 0.0))

        result = "pending"
        profit = 0.0

        if bet_type == "moneyline":
            if (bet_side == "TeamA" and margin > 0) or (bet_side == "TeamB" and margin < 0):
                result = "win"
                profit = american_profit(stake, odds)
            else:
                result = "loss"
                profit = -stake

        elif bet_type == "spread":
            value = margin + line if bet_side == "TeamA" else -margin - line
            if value > 0:
                result = "win"
                profit = american_profit(stake, odds)
            elif value < 0:
                result = "loss"
                profit = -stake
            else:
                result = "push"
                profit = 0.0

        elif bet_type == "total":
            value = total - line
            if bet_side == "Under":
                value = -value
            if value > 0:
                result = "win"
                profit = american_profit(stake, odds)
            elif value < 0:
                result = "loss"
                profit = -stake
            else:
                result = "push"
                profit = 0.0

        row["FinalScoreTeamA"] = score_a
        row["FinalScoreTeamB"] = score_b
        row["Result"] = result
        row["Profit"] = profit
        row["Status"] = "settled"
        settled_count += 1
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    profit = float(pd.to_numeric(out["Profit"], errors="coerce").fillna(0.0).sum())
    summary = {
        "open_bets": int((out["Status"] != "settled").sum()),
        "settled_bets": int((out["Status"] == "settled").sum()),
        "newly_settled": int(settled_count),
        "profit": profit,
    }
    return out, summary
