"""Unit tests for bet settlement and ledger bookkeeping in mm26_xgb.ledger.

settle_ledger is where scores turn into realised profit/loss, so every bet
type (moneyline / spread / total) and every outcome (win / loss / push) is
pinned, plus the guards for already-settled and not-yet-completed games.
"""
import numpy as np
import pandas as pd
import pytest

from mm26_xgb.ledger import (
    LEDGER_COLUMNS,
    append_open_bets,
    load_ledger,
    save_ledger,
    settle_ledger,
)
from mm26_xgb.markets import american_profit


def make_ledger(rows):
    """Build a ledger DataFrame with every LEDGER_COLUMN present."""
    df = pd.DataFrame(rows)
    for col in LEDGER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[LEDGER_COLUMNS].copy()


def make_scores(rows):
    return pd.DataFrame(
        rows,
        columns=["EventID", "ScoreTeamA", "ScoreTeamB", "Completed"],
    )


def settle_one(bet, score):
    """Settle a single bet against a single completed game; return the row."""
    ledger = make_ledger([{"EventID": "E1", "Status": "open", **bet}])
    scores = make_scores([{"EventID": "E1", **score, "Completed": True}])
    out, summary = settle_ledger(ledger, scores)
    assert summary["newly_settled"] == 1
    return out.iloc[0]


# --- moneyline -----------------------------------------------------------

def test_moneyline_teama_win():
    row = settle_one(
        {"BetType": "moneyline", "BetSide": "TeamA", "OddsAmerican": 150, "StakeAmount": 100},
        {"ScoreTeamA": 80, "ScoreTeamB": 70},
    )
    assert row["Result"] == "win"
    assert row["Profit"] == pytest.approx(american_profit(100, 150))  # 150.0
    assert row["Status"] == "settled"


def test_moneyline_teama_loss():
    row = settle_one(
        {"BetType": "moneyline", "BetSide": "TeamA", "OddsAmerican": 150, "StakeAmount": 100},
        {"ScoreTeamA": 60, "ScoreTeamB": 70},
    )
    assert row["Result"] == "loss"
    assert row["Profit"] == pytest.approx(-100.0)


def test_moneyline_teamb_win():
    row = settle_one(
        {"BetType": "moneyline", "BetSide": "TeamB", "OddsAmerican": -120, "StakeAmount": 60},
        {"ScoreTeamA": 65, "ScoreTeamB": 70},
    )
    assert row["Result"] == "win"
    assert row["Profit"] == pytest.approx(american_profit(60, -120))


# --- spread --------------------------------------------------------------

def test_spread_teama_covers():
    # TeamA +2.5, loses by 2 -> covers.
    row = settle_one(
        {"BetType": "spread", "BetSide": "TeamA", "Line": 2.5, "OddsAmerican": -110, "StakeAmount": 100},
        {"ScoreTeamA": 70, "ScoreTeamB": 72},
    )
    assert row["Result"] == "win"
    assert row["Profit"] == pytest.approx(american_profit(100, -110))


def test_spread_teama_loss():
    # TeamA -5.5, wins by only 4 -> fails to cover.
    row = settle_one(
        {"BetType": "spread", "BetSide": "TeamA", "Line": -5.5, "OddsAmerican": -110, "StakeAmount": 100},
        {"ScoreTeamA": 74, "ScoreTeamB": 70},
    )
    assert row["Result"] == "loss"
    assert row["Profit"] == pytest.approx(-100.0)


def test_spread_push_on_integer_line():
    # TeamA -10, wins by exactly 10 -> push, stake returned (profit 0).
    row = settle_one(
        {"BetType": "spread", "BetSide": "TeamA", "Line": -10, "OddsAmerican": -110, "StakeAmount": 100},
        {"ScoreTeamA": 80, "ScoreTeamB": 70},
    )
    assert row["Result"] == "push"
    assert row["Profit"] == pytest.approx(0.0)


def test_spread_teamb_covers():
    # TeamB +3 and wins outright by 10 -> easily covers (value = -margin - line).
    row = settle_one(
        {"BetType": "spread", "BetSide": "TeamB", "Line": 3, "OddsAmerican": -110, "StakeAmount": 100},
        {"ScoreTeamA": 70, "ScoreTeamB": 80},
    )
    assert row["Result"] == "win"


# --- total ---------------------------------------------------------------

def test_total_over_win():
    row = settle_one(
        {"BetType": "total", "BetSide": "Over", "Line": 150, "OddsAmerican": -110, "StakeAmount": 100},
        {"ScoreTeamA": 80, "ScoreTeamB": 78},  # total 158
    )
    assert row["Result"] == "win"


def test_total_under_win():
    row = settle_one(
        {"BetType": "total", "BetSide": "Under", "Line": 150, "OddsAmerican": -110, "StakeAmount": 100},
        {"ScoreTeamA": 70, "ScoreTeamB": 70},  # total 140
    )
    assert row["Result"] == "win"


def test_total_push():
    row = settle_one(
        {"BetType": "total", "BetSide": "Over", "Line": 158, "OddsAmerican": -110, "StakeAmount": 100},
        {"ScoreTeamA": 80, "ScoreTeamB": 78},  # total 158 == line
    )
    assert row["Result"] == "push"
    assert row["Profit"] == pytest.approx(0.0)


# --- guards / bookkeeping ------------------------------------------------

def test_already_settled_row_is_untouched():
    ledger = make_ledger([
        {"EventID": "E1", "Status": "settled", "BetType": "moneyline", "BetSide": "TeamA",
         "OddsAmerican": 150, "StakeAmount": 100, "Result": "win", "Profit": 150.0},
    ])
    scores = make_scores([{"EventID": "E1", "ScoreTeamA": 1, "ScoreTeamB": 99, "Completed": True}])
    out, summary = settle_ledger(ledger, scores)
    # Despite the (losing) score, the pre-settled row keeps its booked profit.
    assert summary["newly_settled"] == 0
    assert out.iloc[0]["Profit"] == pytest.approx(150.0)
    assert out.iloc[0]["Result"] == "win"


def test_incomplete_game_stays_open():
    ledger = make_ledger([
        {"EventID": "E1", "Status": "open", "BetType": "moneyline", "BetSide": "TeamA",
         "OddsAmerican": 150, "StakeAmount": 100},
    ])
    scores = make_scores([{"EventID": "E1", "ScoreTeamA": np.nan, "ScoreTeamB": np.nan, "Completed": False}])
    out, summary = settle_ledger(ledger, scores)
    assert summary["newly_settled"] == 0
    assert summary["open_bets"] == 1
    assert str(out.iloc[0]["Status"]) != "settled"


def test_missing_score_event_stays_open():
    ledger = make_ledger([
        {"EventID": "E1", "Status": "open", "BetType": "moneyline", "BetSide": "TeamA",
         "OddsAmerican": 150, "StakeAmount": 100},
    ])
    out, summary = settle_ledger(ledger, make_scores([]))
    assert summary["newly_settled"] == 0
    assert summary["open_bets"] == 1


def test_empty_ledger():
    out, summary = settle_ledger(make_ledger([]), make_scores([]))
    assert summary == {"open_bets": 0, "settled_bets": 0, "profit": 0.0}
    assert out.empty


def test_summary_aggregates_profit():
    ledger = make_ledger([
        {"EventID": "E1", "Status": "open", "BetType": "moneyline", "BetSide": "TeamA",
         "OddsAmerican": 100, "StakeAmount": 100},
        {"EventID": "E2", "Status": "open", "BetType": "moneyline", "BetSide": "TeamA",
         "OddsAmerican": 100, "StakeAmount": 100},
    ])
    scores = make_scores([
        {"EventID": "E1", "ScoreTeamA": 80, "ScoreTeamB": 70, "Completed": True},  # win +100
        {"EventID": "E2", "ScoreTeamA": 60, "ScoreTeamB": 70, "Completed": True},  # loss -100
    ])
    out, summary = settle_ledger(ledger, scores)
    assert summary["newly_settled"] == 2
    assert summary["settled_bets"] == 2
    assert summary["profit"] == pytest.approx(0.0)  # +100 - 100


# --- append_open_bets / load / save -------------------------------------

def test_append_dedupes_identical_bets():
    bet = {"EventID": "E1", "Bookmaker": "pinnacle", "BetType": "moneyline", "BetSide": "TeamA",
           "Line": np.nan, "OddsAmerican": 150, "StakeAmount": 20.0}
    new = make_ledger([bet, dict(bet)])  # exact duplicate
    out = append_open_bets(make_ledger([]), new)
    assert len(out) == 1


def test_append_keeps_distinct_bets():
    base = {"EventID": "E1", "Bookmaker": "pinnacle", "BetType": "moneyline", "BetSide": "TeamA",
            "Line": np.nan, "OddsAmerican": 150, "StakeAmount": 20.0}
    other = dict(base, BetSide="TeamB", OddsAmerican=-170)
    out = append_open_bets(make_ledger([base]), make_ledger([other]))
    assert len(out) == 2


def test_load_save_round_trip(tmp_path):
    ledger = make_ledger([
        {"EventID": "E1", "Status": "open", "BetType": "moneyline", "BetSide": "TeamA",
         "OddsAmerican": 150, "StakeAmount": 100},
    ])
    path = tmp_path / "ledger.csv"
    save_ledger(ledger, path)
    reloaded = load_ledger(path)
    assert list(reloaded.columns) == LEDGER_COLUMNS
    assert reloaded.iloc[0]["EventID"] == "E1"


def test_load_missing_file_returns_empty_schema(tmp_path):
    out = load_ledger(tmp_path / "nope.csv")
    assert out.empty
    assert list(out.columns) == LEDGER_COLUMNS
