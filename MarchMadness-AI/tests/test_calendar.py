from datetime import date, datetime, timezone

import pandas as pd
import pytest

from mmai.calendar import (
    SELECTION_SUNDAY_DAYNUM,
    date_of,
    date_range,
    day_zero,
    daynum,
    et_date,
    season_of,
    to_utc,
)

SEASONS = pd.DataFrame({"Season": [2025, 2026], "DayZero": ["11/04/2024", "11/03/2025"]})


def test_season_of_names_season_by_end_year():
    assert season_of(date(2026, 11, 11)) == 2027
    assert season_of(date(2027, 3, 14)) == 2027
    assert season_of(date(2026, 6, 30)) == 2026


def test_day_zero_prefers_kaggle_seasons_file():
    assert day_zero(2026, SEASONS) == date(2025, 11, 3)


def test_2026_selection_sunday_was_daynum_132():
    assert daynum(date(2026, 3, 15), 2026, SEASONS) == SELECTION_SUNDAY_DAYNUM


def test_2027_override_puts_selection_sunday_on_daynum_132():
    assert daynum(date(2027, 3, 14), 2027) == SELECTION_SUNDAY_DAYNUM


def test_unknown_season_raises():
    with pytest.raises(KeyError):
        day_zero(2031)


def test_daynum_round_trip():
    d = date(2027, 1, 5)
    assert date_of(daynum(d, 2027), 2027) == d


def test_late_tip_stays_on_its_eastern_date():
    # 11:00pm EDT on 2026-03-19 is 03:00 UTC on 2026-03-20.
    assert et_date("2026-03-20T03:00Z") == date(2026, 3, 19)
    assert et_date("2026-03-19T19:00Z") == date(2026, 3, 19)
    # Winter (EST, UTC-5): 11:30pm ET on 2026-01-10 is 04:30 UTC on 01-11.
    assert et_date("2026-01-11T04:30Z") == date(2026, 1, 10)


def test_to_utc_accepts_z_offsets_and_naive():
    expected = datetime(2026, 3, 19, 19, 0, tzinfo=timezone.utc)
    assert to_utc("2026-03-19T19:00:00Z") == expected
    assert to_utc("2026-03-19T15:00:00-04:00") == expected
    assert to_utc(datetime(2026, 3, 19, 19, 0)) == expected


def test_date_range_is_inclusive():
    assert date_range(date(2026, 3, 1), date(2026, 3, 3)) == [date(2026, 3, 1), date(2026, 3, 2), date(2026, 3, 3)]
