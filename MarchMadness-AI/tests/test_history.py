import numpy as np
import pandas as pd

from mmai.ids.resolver import TeamResolver
from mmai.market.history import build_line_table, choose_lines, compare_to_reference


def _odds_row(event_id, provider, pid, *, spread=None, details=None, total=None, prio=None, close_spread=None,
              close_total=None, open_spread=None, open_total=None):
    return {"event_id": event_id, "provider": provider, "provider_id": pid, "provider_priority": prio,
            "spread_home": spread, "spread_home_from_details": details, "total": total,
            "ml_home": None, "ml_away": None, "close_spread_home": close_spread, "close_total": close_total,
            "close_ml_home": None, "close_ml_away": None, "open_spread_home": open_spread, "open_total": open_total}


ODDS = pd.DataFrame([
    # e1: a structured close beats consensus
    _odds_row("e1", "consensus", "1004", spread=-7.0, details=-7.0, total=140.0),
    _odds_row("e1", "ESPN BET", "58", spread=-7.5, details=-7.5, total=139.5, prio=1, close_spread=-8.0,
              close_total=138.5, open_spread=-6.5, open_total=141.0),
    # e2: consensus beats a plain book line
    _odds_row("e2", "Westgate", "36", spread=3.0, details=3.0, total=150.0, prio=2),
    _odds_row("e2", "consensus", "1004", spread=2.5, details=2.5, total=151.0),
    # e3: ESPN's spread sign disagrees with its own details string -> details win, flagged
    _odds_row("e3", "Caesars", "38", spread=4.0, details=-4.0, total=130.0, prio=1),
])


def test_choose_lines_priority_and_sign_check():
    lines = choose_lines(ODDS).set_index("event_id")
    assert (lines.loc["e1", "line_kind"], lines.loc["e1", "spread_home"], lines.loc["e1", "total"]) == ("close", -8.0, 138.5)
    assert (lines.loc["e1", "open_spread_home"], lines.loc["e1", "n_providers"]) == (-6.5, 2)
    assert (lines.loc["e2", "line_kind"], lines.loc["e2", "spread_home"]) == ("consensus_final", 2.5)
    assert lines.loc["e3", "spread_home"] == -4.0 and bool(lines.loc["e3", "spread_sign_conflict"])


def test_choose_lines_skips_live_and_opening_feeds_and_takes_book_medians():
    odds = pd.DataFrame([
        # e4: in-play and opening feeds never count; three books' finals -> their median
        _odds_row("e4", "DraftKings - Live Odds", "200", spread=-15.0, details=-15.0, total=120.0,
                  close_spread=-15.0, close_total=120.0),
        _odds_row("e4", "Opening", "23", spread=-1.0, details=-1.0, total=150.0),
        _odds_row("e4", "Westgate", "25", spread=-3.0, details=-3.0, total=141.0),
        _odds_row("e4", "Unibet", "36", spread=-4.0, details=-4.0, total=142.0),
        _odds_row("e4", "MGM", "47", spread=-6.0, details=-6.0, total=146.0),
        # e5: structured closes from two books -> their median; finals ignored
        _odds_row("e5", "ESPN BET", "58", spread=2.0, details=2.0, total=150.0, close_spread=2.5, close_total=151.0),
        _odds_row("e5", "DraftKings", "40", spread=2.0, details=2.0, total=150.0, close_spread=1.5, close_total=152.0),
    ])
    odds.loc[odds["provider"].eq("Westgate"), "ml_home"] = -105.0
    odds.loc[odds["provider"].eq("Unibet"), "ml_home"] = 105.0
    odds.loc[odds["provider"].eq("MGM"), "ml_home"] = -110.0
    lines = choose_lines(odds).set_index("event_id")
    e4, e5 = lines.loc["e4"], lines.loc["e5"]
    assert (e4["line_kind"], e4["spread_home"], e4["total"], e4["n_providers"]) == ("book_final", -4.0, 142.0, 3)
    assert e4["provider"] == "median of 3 books"
    assert e4["ml_home"] == -105.0  # medianed as decimal odds: 1.909, 1.952, 2.05 -> 1.952
    assert (e5["line_kind"], e5["spread_home"], e5["total"]) == ("close", 2.0, 151.5)


def _games():
    return pd.DataFrame([
        {"event_id": "e1", "season": 2025, "date_et": "2025-03-20", "start_utc": None, "neutral": True,
         "home_id": "150", "away_id": "2561", "home_name": "Duke", "away_name": "Siena", "home_score": 80,
         "away_score": 70, "completed": True},
        {"event_id": "e2", "season": 2025, "date_et": "2025-03-21", "start_utc": None, "neutral": True,
         "home_id": "193", "away_id": "2390", "home_name": "Miami (OH)", "away_name": "Miami", "home_score": 60,
         "away_score": 66, "completed": True},
    ])


def test_build_line_table_keeps_games_without_lines():
    t = build_line_table(_games(), ODDS[ODDS["event_id"] == "e1"])
    assert len(t) == 2 and t.loc[t["event_id"] == "e2", "spread_home"].isna().all()


def test_compare_to_reference_orients_spreads_by_kaggle_team():
    resolver = TeamResolver(pd.DataFrame([
        {"gender": "M", "source": "espn", "key": k, "kaggle_id": v}
        for k, v in {"150": 1181, "2561": 1373, "193": 1275, "2390": 1274}.items()]))
    lines = build_line_table(_games(), ODDS)
    ref = pd.DataFrame([
        # e1 listed with the ESPN *away* team as TeamA: ESPN home -8.0 -> TeamA +8.0
        {"CommenceTime": "2025-03-20T23:00:00Z", "TeamAID": 1373, "TeamBID": 1181, "MarketSpreadTeamA": 8.5, "MarketTotal": 139.0},
        {"CommenceTime": "2025-03-21T18:00:00Z", "TeamAID": 1275, "TeamBID": 1274, "MarketSpreadTeamA": 2.5, "MarketTotal": 151.0},
        {"CommenceTime": "2025-03-25T18:00:00Z", "TeamAID": 1, "TeamBID": 2, "MarketSpreadTeamA": 1.0, "MarketTotal": 140.0},
    ])
    rep = compare_to_reference(lines, ref, resolver)
    assert rep["matched"] == 2
    assert np.isclose(rep["spread_mae"], 0.25) and np.isclose(rep["total_mae"], 0.25)
