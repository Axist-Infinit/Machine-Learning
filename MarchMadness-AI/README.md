# MarchMadness-AI

NCAA basketball predictions for March 2027, built so they can be relied on:

- as-of team ratings: strictly pre-tip, with no leakage
- a full score distribution for every game
- a 76-team Monte Carlo bracket and a pool optimizer
- Kaggle March Mania submissions (men's + women's)
- a betting card that only stakes money if pre-registered gates pass

**Status: M0 (foundation).** The full plan, milestones and acceptance gates are in
[`docs/PLAN_2027.md`](docs/PLAN_2027.md); the gates are pinned in [`src/mmai/eval/prereg.yaml`](src/mmai/eval/prereg.yaml).

## Quick start

```bash
./mmai doctor            # preflight: Kaggle data, season calendar, team IDs, API key, odds logger
./mmai ingest-espn --gender M W --start 2026-11-02 --end 2026-11-11 --box   # results + box scores
./mmai ingest-espn --gender M --seasons 2017-2026 --window postseason --odds # historical lines
./mmai crosswalk         # learn ESPN / Odds API -> Kaggle TeamIDs from game results
./mmai log-odds          # snapshot current lines (needs ODDS_API_KEY; hard monthly credit cap)
```

`./mmai` runs straight from a checkout (it puts `src/` on `PYTHONPATH`); after
`pip install -e .` the `mmai` command does the same.

**Data.**
- Put Kaggle's `march-machine-learning-mania-YYYY.zip` in `workspace/data/kaggle_zips/`, and never commit it: the competition rules forbid redistribution.
- Everything generated lives in the gitignored `workspace/`:
  - `raw/` holds API responses, kept forever.
  - `store/` holds parquet tables, rebuildable from `raw/`.

**Team IDs.** Team IDs are learned from game results, not names, which is how the legacy pipeline turned Miami (OH) into Miami (FL). An unknown team is a hard error. Manual overrides go in `config/team_aliases.csv`.

## Layout

```
src/mmai/
  calendar.py        season / DayNum / Eastern-date conventions (DayZero 2027 = 2026-11-02)
  sources/           kaggle (reads the zip), espn (results, box scores, lines), odds_api, odds_logger, http (cache)
  ids/               crosswalk (learned from games), resolver (strict)
  market/odds_math.py  odds conversion, EV, Kelly (ported with its tests)
  eval/prereg.yaml   pre-registered gates
  cli.py, doctor.py
config/team_aliases.csv
legacy/              the pre-2027 pipeline, frozen (see legacy/README.md)
docs/                PLAN_2027.md, REVIEW_2026-06.md
```

## Development

```bash
pip install -r requirements.lock && pip install -e ".[dev]"
pytest                    # or, without installing: python3 -m pytest
```

CI (`.github/workflows/marchmadness-ci.yml`) runs the tests on Python 3.10 and 3.12, plus `ruff check src tests`.
Tests use synthetic fixtures only, so no Kaggle data or API keys are needed.

## Odds logger schedule (example crontab)

Cron doesn't read `~/.bashrc`, so load the key from a `chmod 600` file:

```
0 10-23/2 * 11,12,1,2,3,4 * cd ~/machine-learning/MarchMadness-AI && ODDS_API_KEY=$(cat ~/.odds_api_key) ./mmai log-odds >> workspace/logs/odds.log 2>&1
```

A WSL2 machine only runs cron while Windows is awake. ESPN's closing lines backstop any gaps, and `mmai doctor` reports how old the newest snapshot is.
