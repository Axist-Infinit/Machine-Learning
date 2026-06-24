# March Madness market console

This bundle gives you one Python entry point:

```bash
python run_march_madness.py
```

It supports:

- pulling historical Odds API snapshots for a season or tournament window
- backtesting 2025 against actual game results
- scoring the current 2026 March Madness market against the model
- simulating the full 2026 bracket

## Included

- `run_march_madness.py` — main interactive + CLI runner
- `src/mm26_xgb/` — market-aware XGBoost package
- `trained_kaggle_market_model/` — bundled 2026 model artifacts
- `assets/march-machine-learning-mania-2026.zip` — Kaggle data bundle
- `assets/mm26_first_four_round1_prep/` — current 2026 First Four + Round 1 prep bundle
- `assets/win_loss.csv`, `assets/ats.csv`, `assets/ou.csv` — external 2026 team metrics
- `config/defaults.json` — defaults and API wiring

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Fastest path

Launch the menu:

```bash
python run_march_madness.py
```

Or use direct commands.

Bootstrap the workspace:

```bash
python run_march_madness.py bootstrap
```

Pull historical 2025 tournament odds:

```bash
python run_march_madness.py pull-odds --year 2025 --scope tournament --interval-hours 6
```

Pull historical 2025 full-season odds:

```bash
python run_march_madness.py pull-odds --year 2025 --scope season --interval-hours 6
```

Backtest 2025 March Madness, auto-fetching history if missing:

```bash
python run_march_madness.py backtest --year 2025 --scope tournament --auto-fetch
```

Backtest the 2025 full season if you already pulled full-season history:

```bash
python run_march_madness.py backtest --year 2025 --scope season
```

Score the current 2026 tournament market using live Odds API rows where available and the bundled prep lines as fallback:

```bash
python run_march_madness.py predict-current --market-source hybrid
```

Score only the bundled 2026 prep lines without hitting the network:

```bash
python run_march_madness.py predict-current --market-source prep
```

Simulate the full 2026 bracket:

```bash
python run_march_madness.py predict-bracket --season 2026
```

Run a local offline smoke test:

```bash
python run_march_madness.py smoke-test
```

## Odds API constraints you should know

- The current NCAAB featured-market endpoint uses sport key `basketball_ncaab` and markets `h2h,spreads,totals`.
- The historical odds pull used for 2025 backtests depends on The Odds API historical endpoint. If your subscription does not include historical odds, `pull-odds` / `backtest --auto-fetch` will fail until you upgrade or provide your own raw snapshots CSV.
- Historical pulls get expensive fast because each call costs `10 x markets x regions`, so the interval and date window matter.

## Output locations

Everything is written under `workspace/outputs/`.

Typical folders:

- `workspace/outputs/odds_history/<year>_<scope>/`
- `workspace/outputs/backtests/<year>_<scope>/`
- `workspace/outputs/current_market/2026/`
- `workspace/outputs/brackets/2026/`

## What the backtest writes

- `matched_market_consensus.csv`
- `matched_market_book_rows.csv`
- `backtest_predictions.csv`
- `backtest_game_report.csv`
- `backtest_candidate_sides.csv`
- `backtest_recommended_bets_settled.csv`
- `backtest_summary.json`
- `backtest_report.md`

## What the 2026 current-market run writes

- `raw_market_rows.csv`
- `market_consensus.csv`
- `market_predictions.csv`
- `market_candidate_sides.csv`
- `market_summary.json`
- `market_report.md`

## Notes that matter

- The historical Odds API pull is the expensive part. Wider windows and shorter intervals burn credits quickly.
- The bundled 2026 current-market fallback uses the attached prep bundle, so it still works offline.
- The bundled trained model is reused for 2026 by default. The 2025 backtest path trains a fresh holdout model for 2025 when needed.
- `ODDS_API_KEY` in your shell overrides the bundled default automatically.


## Direct ATS / O-U residual models and lineup-tempo features

This bundle now supports:
- direct ATS cover models trained on `CoverTarget = sign(actual_margin + closing_spread_team_a)`
- direct totals models trained on `OverTarget = sign(actual_total - closing_total)`
- blended spread and total probabilities that combine the score-distribution model with the direct market classifiers
- optional external lineup and tempo feature files at `assets/lineup_features.csv` and `assets/tempo_features.csv`
- snapshot training that reuses locally cached market history from `workspace/outputs/odds_history/*/raw_snapshots.csv` and `workspace/cache/season_state/*` before any new Odds API pull

If you do not populate the lineup/tempo CSVs, the script still runs and falls back to the existing feature set.


## Multi-season ATS / O-U market training

The core winner and score models remain **same-season snapshot** models.

The direct ATS and totals heads now work differently:
- they reuse **cached market-labeled history from multiple seasons**
- by default the lookback is `4` prior seasons, with a floor of `2021`
- for a 2025 round snapshot, that means the direct ATS / O-U heads try to use cached rows from `2021-2024` plus the 2025 rows available before that round
- no missing historical seasons are auto-fetched during backtests; the script only uses what is already cached locally unless you explicitly run `pull-odds`

Recommended cache-building sequence if you want the direct ATS / O-U heads to have enough data:

```bash
python run_march_madness.py pull-odds --year 2021 --scope season
python run_march_madness.py pull-odds --year 2022 --scope season
python run_march_madness.py pull-odds --year 2023 --scope season
python run_march_madness.py pull-odds --year 2024 --scope season
python run_march_madness.py pull-odds --year 2025 --scope season
python run_march_madness.py pull-odds --year 2025 --scope tournament
```

Then rerun the 2025 tournament backtest. The script will prefer the local cache and raw snapshot files over new API calls.
