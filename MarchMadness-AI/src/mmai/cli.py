"""``mmai`` command line. Run ``./mmai --help`` from the project root."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from mmai.calendar import date_range, season_of
from mmai.config import Paths, get_paths
from mmai.ids.crosswalk import kaggle_results_with_dates, learn_espn_crosswalk, learn_name_crosswalk
from mmai.sources.espn import ESPN, collect
from mmai.sources.http import CachedJSONClient
from mmai.sources.kaggle import KaggleData
from mmai.sources.odds_api import CreditBudgetExceeded, MissingAPIKey, OddsAPI
from mmai.sources.odds_logger import log_snapshot
from mmai.store import read_parquets, write_parquet

TABLE_KEYS = {"games": ["event_id"], "box": ["event_id", "team_id"], "odds": ["event_id", "provider_id"]}
KNOWN_PROBLEM_NAMES = ["GW Revolutionaries", "Miami Hurricanes", "Miami (OH) RedHawks",
                       "Queens University Royals", "San José St Spartans"]


def espn_store(paths: Paths, kind: str, gender: str) -> Path:
    return paths.store / "espn" / kind / f"gender={gender}"


def crosswalk_path(paths: Paths) -> Path:
    return paths.store / "ids" / "team_crosswalk.csv"


def _merge_write(new: pd.DataFrame, path: Path, keys: list[str]) -> int:
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    new = new.drop_duplicates(keys, keep="last").reset_index(drop=True)
    write_parquet(new, path)
    return len(new)


def _season_windows(first: int, last: int, window: str) -> list[tuple[date, date]]:
    if window == "full":
        return [(date(s - 1, 11, 1), date(s, 4, 10)) for s in range(first, last + 1)]
    # postseason: the last regular-season week, conference tournaments, NCAA/NIT
    return [(date(s, 2, 25), date(s, 4, 10)) for s in range(first, last + 1) if s != 2020]


# --- commands ----------------------------------------------------------------------------------

def cmd_ingest_espn(args: argparse.Namespace) -> int:
    paths = get_paths()
    espn = ESPN(CachedJSONClient(paths.raw / "espn", min_interval_s=args.min_interval))
    if args.seasons:
        first, last = (int(x) for x in args.seasons.split("-"))
        windows = _season_windows(first, last, args.window)
    else:
        if not (args.start and args.end):
            print("give --start/--end or --seasons", file=sys.stderr)
            return 2
        windows = [(date.fromisoformat(args.start), date.fromisoformat(args.end))]
    for start, end in windows:
        for g in args.gender:
            res = collect(espn, g, date_range(start, end), box=args.box, odds=args.odds, log=print)
            games = res["games"]
            if games.empty:
                print(f"{g} {start}..{end}: no games")
                continue
            games["season"] = [season_of(date.fromisoformat(d)) for d in games["date_et"]]
            season_of_event = dict(zip(games["event_id"], games["season"]))
            for kind, df in res.items():
                if df.empty:
                    continue
                df = df.copy()
                df["season"] = df["event_id"].map(season_of_event)
                for season, part in df.groupby("season"):
                    path = espn_store(paths, kind, g) / f"season={int(season)}.parquet"
                    n = _merge_write(part, path, TABLE_KEYS[kind])
                    print(f"{g} {kind:5s} season {int(season)}: {n:6d} rows  ({start}..{end})")
    print(f"ESPN requests made: {espn.client.requests_made}")
    return 0


def load_odds_api_events(paths: Paths) -> pd.DataFrame:
    """Every (gender, commence time, home, away) seen in cached/logged Odds API data."""
    rows = []
    for f in sorted(paths.legacy_odds_cache.glob("*.json")) if paths.legacy_odds_cache.exists() else []:
        payload = json.loads(f.read_text()).get("payload")
        events = payload.get("data") if isinstance(payload, dict) else payload
        for e in events or []:
            if isinstance(e, dict) and e.get("home_team") and e.get("commence_time"):
                gender = "W" if "wncaab" in str(e.get("sport_key", "")) else "M"
                rows.append((gender, e["commence_time"], e["home_team"], e["away_team"]))
    snaps = read_parquets(paths.store / "odds_snapshots")
    if not snaps.empty:
        rows += list(snaps[["gender", "commence_utc", "home_team", "away_team"]].itertuples(index=False, name=None))
    df = pd.DataFrame(rows, columns=["gender", "commence_utc", "home_team", "away_team"])
    df["commence_utc"] = pd.to_datetime(df["commence_utc"], utc=True)
    return df.drop_duplicates().reset_index(drop=True)


def cmd_crosswalk(args: argparse.Namespace) -> int:
    paths = get_paths()
    kd = KaggleData.latest(paths.kaggle_zips)
    out: list[dict] = []
    espn_to_kaggle: dict[str, dict[str, int]] = {}
    for g in ("M", "W"):
        games = read_parquets(espn_store(paths, "games", g))
        if games.empty:
            print(f"{g}: no ESPN games in the store; run `mmai ingest-espn` first")
            continue
        phases = [p for p in ("RegularSeason", "NCAATourney", "SecondaryTourney")
                  if kd.has(f"{g}{p}CompactResults.csv")]
        kres = pd.concat([kd.compact(g, p) for p in phases], ignore_index=True)
        seasons = sorted(set(games["season"]) & set(kres["Season"]))
        if args.seasons:
            seasons = [s for s in seasons if s in set(args.seasons)]
        kres = kaggle_results_with_dates(kres[kres["Season"].isin(seasons)], kd.seasons(g))
        xw, rep = learn_espn_crosswalk(games[games["season"].isin(seasons)], kres)
        print(f"{g} ESPN->Kaggle from seasons {seasons}: {rep.games} ESPN games, {rep.matched} matched, "
              f"{rep.ambiguous} ambiguous, {rep.unmatched} unmatched (non-D1 opponents) -> {rep.mapped} teams mapped; "
              f"{len(rep.low_confidence)} low-confidence, {len(rep.conflicts)} conflicts")
        for row in rep.conflicts[:10]:
            print(f"   conflict: {row}")
        espn_to_kaggle[g] = dict(zip(xw["espn_id"], xw["kaggle_id"]))
        team_names = dict(zip(kd.teams(g)["TeamID"], kd.teams(g)["TeamName"]))
        for r in xw.itertuples(index=False):
            out.append({"gender": g, "source": "espn", "key": r.espn_id, "kaggle_id": r.kaggle_id,
                        "votes": r.votes, "share": r.share, "label": f"{r.espn_name} = {team_names.get(r.kaggle_id)}"})

    events = load_odds_api_events(paths)
    for g, ev in events.groupby("gender"):
        games = read_parquets(espn_store(paths, "games", g))
        if games.empty or g not in espn_to_kaggle:
            continue
        nx, nrep = learn_name_crosswalk(ev, games, espn_to_kaggle[g])
        mapped = nx.dropna(subset=["kaggle_id"])
        print(f"{g} Odds API names: {nrep['matched']}/{nrep['events']} events matched to ESPN games -> "
              f"{len(mapped)}/{len(nx)} names mapped to Kaggle IDs")
        for r in mapped.itertuples(index=False):
            out.append({"gender": g, "source": "odds_api", "key": r.name, "kaggle_id": int(r.kaggle_id),
                        "votes": r.votes, "share": r.share, "label": r.name})
        names = {r.name: r for r in nx.itertuples(index=False)}
        if g == "M":
            for n in KNOWN_PROBLEM_NAMES:
                r = names.get(n)
                print(f"   {n!r:32s} -> " + (f"Kaggle {r.kaggle_id} (votes {r.votes})" if r is not None else "not seen"))

    df = pd.DataFrame(out, columns=["gender", "source", "key", "kaggle_id", "votes", "share", "label"])
    path = crosswalk_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} rows -> {path.relative_to(paths.root)}")
    return 0


def cmd_log_odds(args: argparse.Namespace) -> int:
    paths = get_paths()
    api = OddsAPI(CachedJSONClient(paths.raw / "odds_api", min_interval_s=1.0),
                  paths.store / "odds_api_ledger.csv", monthly_budget=args.budget)
    try:
        written = log_snapshot(api, paths.store, args.gender, bookmakers=args.books)
    except (MissingAPIKey, CreditBudgetExceeded) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for p in written:
        print(f"{p.relative_to(paths.root)}: {len(pd.read_parquet(p))} rows")
    print(f"credits used this month (ledger): {api.spent_this_month()}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from mmai.doctor import run_doctor

    checks = run_doctor(get_paths())
    for status, name, detail in checks:
        print(f"[{status:4s}] {name:22s} {detail}")
    return 1 if any(s == "FAIL" for s, _, _ in checks) else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mmai", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest-espn", help="pull ESPN scoreboards (+ box scores, + lines) into the store")
    s.add_argument("--gender", nargs="+", default=["M", "W"], choices=["M", "W"])
    s.add_argument("--start", help="first date (YYYY-MM-DD)")
    s.add_argument("--end", help="last date (YYYY-MM-DD)")
    s.add_argument("--seasons", help="season range such as 2017-2026 (instead of --start/--end)")
    s.add_argument("--window", choices=["postseason", "full"], default="postseason")
    s.add_argument("--box", action="store_true", help="also fetch box scores for completed games")
    s.add_argument("--odds", action="store_true", help="also fetch lines for completed games")
    s.add_argument("--min-interval", type=float, default=0.35, help="seconds between requests")
    s.set_defaults(func=cmd_ingest_espn)

    s = sub.add_parser("crosswalk", help="learn ESPN/Odds API -> Kaggle team IDs from games")
    s.add_argument("--seasons", nargs="*", type=int, help="seasons to learn from (default: all overlapping)")
    s.set_defaults(func=cmd_crosswalk)

    s = sub.add_parser("log-odds", help="snapshot current pre-game lines (needs ODDS_API_KEY)")
    s.add_argument("--gender", nargs="+", default=["M", "W"], choices=["M", "W"])
    s.add_argument("--books", nargs="*", help="bookmaker keys (<=10 count as one region)")
    s.add_argument("--budget", type=int, default=20_000, help="hard monthly credit cap")
    s.set_defaults(func=cmd_log_odds)

    s = sub.add_parser("doctor", help="preflight checks: data, calendar, IDs, keys, logger")
    s.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)
