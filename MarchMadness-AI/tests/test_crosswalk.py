"""Crosswalks are learned from games, never from names."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from mmai.ids.crosswalk import (
    kaggle_results_with_dates,
    learn_espn_crosswalk,
    learn_name_crosswalk,
    unmapped_division_one,
)

# Truth: ESPN id -> Kaggle id. Two "Miami"s on purpose: names must not matter.
TRUTH = {"193": 1275, "2390": 1274, "150": 1181, "153": 1314, "2561": 1373, "45": 1203}
NAMES = {"193": "Miami (OH) RedHawks", "2390": "Miami Hurricanes", "150": "Duke Blue Devils",
         "153": "North Carolina Tar Heels", "2561": "Siena Saints", "45": "George Washington Revolutionaries"}
LOC = {"193": "Miami (OH)", "2390": "Miami", "150": "Duke", "153": "North Carolina", "2561": "Siena", "45": "George Washington"}


def _season_games():
    """Round robin with distinct scores, played twice, over a few weeks."""
    ids = list(TRUTH)
    games, d, score = [], date(2026, 1, 3), 60
    for _rep in range(2):
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                score += 1
                games.append((d, a, b, score, score - 7))  # home team a wins by 7
                d += timedelta(days=1)
    return games


def _espn_frame(games, shift_days=None):
    rows = []
    for k, (d, h, a, hs, as_) in enumerate(games):
        if shift_days and k in shift_days:
            d = d + timedelta(days=1)  # ESPN's ET date can differ by a day (late west-coast tips)
        rows.append({"date_et": d.isoformat(), "completed": True, "home_id": h, "away_id": a,
                     "home_score": hs, "away_score": as_, "home_name": NAMES[h], "away_name": NAMES[a]})
    return pd.DataFrame(rows)


def _kaggle_frame(games):
    zero = date(2025, 11, 3)
    rows = [{"Season": 2026, "DayNum": (d - zero).days, "WTeamID": TRUTH[h], "LTeamID": TRUTH[a], "WScore": hs,
             "LScore": as_, "WLoc": "H", "NumOT": 0} for d, h, a, hs, as_ in games]
    seasons = pd.DataFrame({"Season": [2026], "DayZero": ["11/03/2025"]})
    return kaggle_results_with_dates(pd.DataFrame(rows), seasons)


def test_kaggle_dates_come_from_dayzero():
    k = _kaggle_frame([(date(2026, 3, 15), "150", "153", 70, 60)])
    assert k["date"].iloc[0] == date(2026, 3, 15)


def test_learns_exact_mapping_despite_date_shifts_and_similar_names():
    games = _season_games()
    xw, rep = learn_espn_crosswalk(_espn_frame(games, shift_days={0, 5, 9}), _kaggle_frame(games))
    assert dict(zip(xw["espn_id"], xw["kaggle_id"])) == TRUTH
    assert rep.matched == len(games) and rep.conflicts == []


def test_non_d1_opponents_are_unmatched_not_mapped():
    games = _season_games()
    espn = _espn_frame(games)
    extra = pd.DataFrame([{"date_et": "2026-01-20", "completed": True, "home_id": "150", "away_id": "99999",
                           "home_score": 101, "away_score": 50, "home_name": "Duke", "away_name": "D2 College"}])
    xw, rep = learn_espn_crosswalk(pd.concat([espn, extra]), _kaggle_frame(games))
    assert "99999" not in set(xw["espn_id"]) and rep.unmatched == 1


def test_identical_scores_on_the_same_day_are_ambiguous_not_guessed():
    games = _season_games()
    d0 = games[0][0]
    kaggle = _kaggle_frame(games + [(d0, "45", "2561", games[0][3], games[0][4])])
    espn = _espn_frame(games)
    _, rep = learn_espn_crosswalk(espn, kaggle)
    assert rep.ambiguous >= 1


def test_low_vote_teams_are_not_mapped():
    games = _season_games()[:3]  # too few games for most teams
    xw, rep = learn_espn_crosswalk(_espn_frame(games), _kaggle_frame(games), min_votes=3)
    assert len(xw) < len(TRUTH) and rep.low_confidence


def test_odds_api_names_inherit_ids_through_tip_times():
    t0 = datetime(2026, 3, 19, 19, 0, tzinfo=timezone.utc)
    espn = pd.DataFrame([
        {"start_utc": t0, "home_id": "150", "away_id": "2561", "home_name": NAMES["150"], "away_name": NAMES["2561"],
         "home_location": LOC["150"], "away_location": LOC["2561"]},
        {"start_utc": t0, "home_id": "2390", "away_id": "45", "home_name": NAMES["2390"], "away_name": NAMES["45"],
         "home_location": LOC["2390"], "away_location": LOC["45"]},
        {"start_utc": t0 + timedelta(hours=2), "home_id": "193", "away_id": "153", "home_name": NAMES["193"],
         "away_name": NAMES["153"], "home_location": LOC["193"], "away_location": LOC["153"]},
    ])
    events = pd.DataFrame([
        {"commence_utc": t0 + timedelta(minutes=5), "home_team": "Duke Blue Devils", "away_team": "Siena Saints"},
        # Odds API lists this one with home/away swapped and the short GW name.
        {"commence_utc": t0, "home_team": "GW Revolutionaries", "away_team": "Miami Hurricanes"},
        {"commence_utc": t0 + timedelta(hours=2), "home_team": "Miami (OH) RedHawks", "away_team": "North Carolina Tar Heels"},
        {"commence_utc": t0 + timedelta(days=3), "home_team": "Nobody", "away_team": "Nowhere"},  # no ESPN game
    ])
    nx, rep = learn_name_crosswalk(events, espn, {k: v for k, v in TRUTH.items()})
    got = dict(zip(nx["name"], nx["kaggle_id"]))
    assert got["Miami Hurricanes"] == 1274 and got["Miami (OH) RedHawks"] == 1275
    assert got["GW Revolutionaries"] == 1203 and got["Siena Saints"] == 1373
    assert rep["matched"] == 3 and rep["unmatched"] == 1


def test_unmapped_division_one_separates_new_teams_from_crosswalk_failures():
    def game(season, home, away, home_conf="46", away_conf="46"):
        return {"season": season, "home_id": home, "away_id": away, "home_name": f"T{home}", "away_name": f"T{away}",
                "home_conf_id": home_conf, "away_conf_id": away_conf}
    games = pd.DataFrame([
        game(2026, "1", "2"), game(2026, "3", "5"),           # 5: in a D1 conference Kaggle covers, unmapped
        game(2027, "1", "4"), game(2027, "2", "3"),           # 4: first D1 season after Kaggle's last -> new
        game(2027, "1", "9", away_conf=None),                 # 9: non-D1 opponent (no conference)
        game(2027, "7", "8", home_conf="99", away_conf="99"),  # conference with no mapped teams: not D1
    ])
    gaps = unmapped_division_one(games, {"1", "2", "3"}, kaggle_last_season=2026).set_index("espn_id")
    assert set(gaps.index) == {"4", "5"}
    assert gaps.loc["4", "kind"] == "new" and gaps.loc["5", "kind"] == "unmapped"
    assert gaps.loc["5", "first_d1_season"] == 2026
