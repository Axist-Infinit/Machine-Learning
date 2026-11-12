"""Season calendar.

Conventions (shared with Kaggle's March Mania data):

- A season is named by the calendar year it ends in: the 2026-27 season is ``2027``.
- ``DayNum`` counts days from the season's ``DayZero`` (a Monday in early November);
  Selection Sunday is DayNum 132.
- Game dates are US/Eastern calendar dates. Tip times are stored in UTC.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
SELECTION_SUNDAY_DAYNUM = 132

#: DayZero for seasons Kaggle has not published yet. 2027: Selection Sunday is Sun 2027-03-14
#: (NCAA calendar), and Selection Sunday is always DayNum 132, so DayZero = 2026-11-02.
#: Kaggle's own M/WSeasons values take precedence once they exist; ``mmai doctor`` flags any
#: disagreement.
DAY_ZERO_OVERRIDES: dict[int, date] = {2027: date(2026, 11, 2)}


def season_of(d: date) -> int:
    """Season a calendar date belongs to (July onward counts toward the next season)."""
    return d.year + 1 if d.month >= 7 else d.year


def day_zero(season: int, seasons: pd.DataFrame | None = None) -> date:
    if seasons is not None and len(seasons):
        hit = seasons.loc[seasons["Season"] == season, "DayZero"]
        if len(hit):
            return datetime.strptime(str(hit.iloc[0]), "%m/%d/%Y").date()
    if season in DAY_ZERO_OVERRIDES:
        return DAY_ZERO_OVERRIDES[season]
    raise KeyError(f"No DayZero known for season {season}; add it to DAY_ZERO_OVERRIDES")


def daynum(d: date, season: int | None = None, seasons: pd.DataFrame | None = None) -> int:
    season = season_of(d) if season is None else season
    return (d - day_zero(season, seasons)).days


def date_of(day: int, season: int, seasons: pd.DataFrame | None = None) -> date:
    return day_zero(season, seasons) + timedelta(days=int(day))


def to_utc(ts: str | datetime) -> datetime:
    """Parse ISO timestamps such as ESPN's ``2026-03-19T19:00Z`` into aware UTC datetimes."""
    if isinstance(ts, datetime):
        dt = ts
    else:
        dt = datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def et_date(ts: str | datetime) -> date:
    """Eastern calendar date of a tip time (an 11pm ET tip is 04:00 UTC the next day)."""
    return to_utc(ts).astimezone(ET).date()


def date_range(start: date, end: date) -> list[date]:
    """Inclusive list of dates."""
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]
