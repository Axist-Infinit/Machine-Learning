"""ESPN parsers against trimmed fixtures that mirror the real responses (checked 2026-11-11)."""

from mmai.sources.espn import parse_box, parse_odds, parse_scoreboard, spread_from_details


def _team(tid, abbr, name, short, conf="2"):
    return {"id": tid, "abbreviation": abbr, "displayName": name, "shortDisplayName": short,
            "location": short, "conferenceId": conf}


SCOREBOARD = {"events": [{
    "id": "401856478",
    "date": "2026-03-20T03:00Z",
    "season": {"year": 2026, "type": 3},
    "competitions": [{
        "date": "2026-03-20T03:00Z",
        "neutralSite": True,
        "conferenceCompetition": False,
        "tournamentId": 22,
        "notes": [{"type": "event", "headline": "NCAA Men's Basketball Championship - East Region - 1st Round"}],
        "venue": {"fullName": "Bon Secours Wellness Arena", "address": {"city": "Greenville", "state": "SC"}},
        "status": {"type": {"completed": True, "name": "STATUS_FINAL"}},
        "competitors": [
            {"homeAway": "home", "score": "71", "winner": True, "linescores": [{}, {}, {}],
             "team": _team("150", "DUKE", "Duke Blue Devils", "Duke")},
            {"homeAway": "away", "score": "65", "winner": False, "linescores": [{}, {}, {}],
             "team": _team("2561", "SIENA", "Siena Saints", "Siena", "27")},
        ],
    }],
}]}


def test_parse_scoreboard_fields():
    (row,) = parse_scoreboard(SCOREBOARD, "M")
    assert row["event_id"] == "401856478"
    assert row["date_et"] == "2026-03-19"  # 11pm EDT tip belongs to the 19th
    assert row["neutral"] and row["completed"] and row["season_type"] == 3
    assert (row["home_id"], row["home_score"], row["away_id"], row["away_score"]) == ("150", 71, "2561", 65)
    assert row["home_abbr"] == "DUKE" and row["away_conf_id"] == "27"
    assert row["periods"] == 3  # one overtime in a men's game
    assert "1st Round" in row["note"]


def test_parse_scoreboard_skips_malformed_events():
    bad = {"events": [{"id": "1", "competitions": [{"competitors": [{"homeAway": "home"}]}]}, {"id": "2"}]}
    assert parse_scoreboard(bad, "M") == []


def _stats(fg, three, ft, oreb, dreb, ast, stl, blk, to, total_to, pf):
    return [
        {"name": "fieldGoalsMade-fieldGoalsAttempted", "displayValue": fg},
        {"name": "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "displayValue": three},
        {"name": "freeThrowsMade-freeThrowsAttempted", "displayValue": ft},
        {"name": "offensiveRebounds", "displayValue": oreb},
        {"name": "defensiveRebounds", "displayValue": dreb},
        {"name": "assists", "displayValue": ast},
        {"name": "steals", "displayValue": stl},
        {"name": "blocks", "displayValue": blk},
        {"name": "turnovers", "displayValue": to},
        {"name": "totalTurnovers", "displayValue": total_to},
        {"name": "fouls", "displayValue": pf},
    ]


SUMMARY = {"boxscore": {"teams": [
    {"homeAway": "away", "team": {"id": "2561"}, "statistics": _stats("25-65", "9-26", "6-6", "11", "20", "15", "5", "2", "6", "7", "16")},
    {"homeAway": "home", "team": {"id": "150"}, "statistics": _stats("24-59", "5-26", "18-21", "16", "27", "11", "3", "3", "8", "8", "12")},
]}}


def test_parse_box_maps_to_kaggle_columns():
    away, home = parse_box(SUMMARY, "M", "401856478")
    assert (away["team_id"], away["FGM"], away["FGA"], away["FGM3"], away["FGA3"]) == ("2561", 25, 65, 9, 26)
    assert (home["FTM"], home["FTA"], home["OR"], home["DR"], home["PF"]) == (18, 21, 16, 27, 12)
    assert away["TO"] == 7  # team turnovers included, like Kaggle


def test_parse_box_returns_nothing_without_shooting_stats():
    summary = {"boxscore": {"teams": [{"team": {"id": "1"}, "statistics": []}, {"team": {"id": "2"}, "statistics": []}]}}
    assert parse_box(summary, "M", "1") == []


ODDS = {"items": [
    {   # 2025+ style: open/close blocks
        "provider": {"id": "100", "name": "DraftKings", "priority": 1},
        "details": "DUKE -28.5", "overUnder": 137.5, "spread": -28.5, "overOdds": -110.0, "underOdds": -110.0,
        "homeTeamOdds": {"moneyLine": -20000, "spreadOdds": -105.0,
                         "open": {"pointSpread": {"american": "-25.5"}, "moneyLine": {"american": "-20000"}},
                         "close": {"pointSpread": {"american": "-28.5"}, "moneyLine": {"american": "-20000"}}},
        "awayTeamOdds": {"moneyLine": 3500, "spreadOdds": -115.0,
                         "open": {"pointSpread": {"american": "+25.5"}, "moneyLine": {"american": "+3500"}},
                         "close": {"pointSpread": {"american": "+28.5"}, "moneyLine": {"american": "+3500"}}},
        "open": {"total": {"american": "144.5"}},
        "close": {"total": {"american": "141.5"}},
    },
    {   # pre-2025 style: final line only, away team favored
        "provider": {"id": "1004", "name": "consensus"},
        "details": "SIENA -2", "overUnder": 140.0, "spread": 2.0,
        "homeTeamOdds": {"moneyLine": 110}, "awayTeamOdds": {"moneyLine": -130},
    },
]}


def test_parse_odds_reads_open_close_and_final_lines():
    dk, cons = parse_odds(ODDS, "M", "401856478", "DUKE", "SIENA")
    assert (dk["spread_home"], dk["open_spread_home"], dk["close_spread_home"]) == (-28.5, -25.5, -28.5)
    assert (dk["total"], dk["open_total"], dk["close_total"]) == (137.5, 144.5, 141.5)
    assert (dk["ml_home"], dk["ml_away"], dk["close_ml_away"]) == (-20000.0, 3500.0, 3500.0)
    assert dk["spread_home_from_details"] == -28.5
    assert cons["close_total"] is None and cons["total"] == 140.0
    assert cons["spread_home"] == 2.0 and cons["spread_home_from_details"] == 2.0


def test_spread_from_details_edge_cases():
    assert spread_from_details("EVEN", "A", "B") == 0.0
    assert spread_from_details("A -3.5", "A", "B") == -3.5
    assert spread_from_details("B -3.5", "A", "B") == 3.5
    assert spread_from_details("C -3.5", "A", "B") is None
    assert spread_from_details(None, "A", "B") is None
