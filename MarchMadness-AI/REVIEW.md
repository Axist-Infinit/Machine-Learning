# MarchMadness-AI — Code & Model Review

_Last updated: 2026-06-24_

A deep-dive review across four tracks — application architecture, ML rigor,
betting/markets correctness, and engineering hygiene. Findings are grouped by
impact. File references use `path:line` against the tree at review time.

> **Note on line numbers.** The dead-code removal in this same change shifted
> line numbers in `run_march_madness.py` (4743 → 4432 lines). References to that
> file below describe the *pre-cleanup* locations and the logic, not exact
> current lines.

---

## Headline: the shipped model is a degenerate training run, and nothing validates it

`trained_kaggle_market_model/training_summary.json` is the smoking gun:

```json
"warning": "Backtest did not run. Check available seasons in your data.",
"market_history_rows": 0,
"external_team_features": { "coverage": { "seasons": [2026] } },
"external_prior": { "win_accuracy": 0.745, "win_logloss": 0.524 }   // in-sample, 722 rows
```

Consequences:

- **~150 of 230 features are dead weight in training.** External betting features
  exist only for 2026 and market history is empty, so for every training row
  (2003–2025) those columns are NaN → filled with the training median (≈0 after
  z-scoring). The model learns them as constants, then at prediction time receives
  real, nonzero values it never saw — a train/serve distribution mismatch
  (`modeling.py:980, 208-220`; `features.py` lacks the high-NaN drop guard that
  `attached_data.py:225` already has).
- **The market-residual engine is inert.** Base margins, cover/over heads, and
  `WinBaseMargin` are keyed off the empty market history, so
  `trained_with_market_* = False` and the machinery silently no-ops
  (`modeling.py:339-343, 736-738`).
- **There is no out-of-sample number for the deployed model.** The walk-forward
  backtest (a) didn't run, and (b) even when it runs, builds a *simpler* model than
  ships — no external prior, no market features, no direct heads
  (`modeling.py:925-961`). The only reported metrics (74.5% / 0.524) are **in-sample**
  on the external prior's own training set (`modeling.py:673-686`).

**Before any tuning, the model needs a training run with real multi-season market
history plus a backtest that exercises the production pipeline. Tuning an
unvalidated artifact is premature.**

---

## Tier 1 — Correctness that determines whether it makes money

1. **The live CLI produces predictions but never sizes bets, and the betting flags
   are silent no-ops.** `predict-current` / `backtest` advertise `--bankroll`,
   `--min-spread-ev`, `--fractional-kelly`, etc., but the active command bodies
   never call `price_bookmaker_sides` or read those args — the only caller was dead
   legacy code (now removed). A user passing `--bankroll 5000 --min-spread-ev 0.03`
   gets no effect. **Fix:** re-wire bet pricing into the active commands, or remove
   the misleading arguments.

2. **The bracket "simulation" is deterministic chalk, not Monte Carlo.**
   `simulate_bracket` runs one pass and advances `argmax(win_prob)` each round — no
   sampling, no `num_sims`, no upset variance — and discards the win probabilities
   it computes (`bracket.py:33-104, 69, 81-82`). There are no Final-Four /
   championship probabilities, which is the main point of a bracket forecast.
   **Fix:** true Monte Carlo — draw `U~Uniform(0,1)` per game, advance over
   N=10k–50k sims, report advancement probabilities with standard errors.

3. **Target leakage in external features.** `ExtATS_CoverPct`, `ExtOU_OverPct`,
   `ExtWL_MOV`, etc. are full-season records with no date cutoff
   (`external_features.py:279-366`). If those CSVs are end-of-season, a historical
   tournament row's features encode the outcome of the very games being predicted.
   **Fix:** require pre-tournament-only external records, or feed them solely
   through the external-prior path, not the main feature matrix.

4. **Calibration is fit to a different model than ships.** Isotonic is fit on a
   *truncated* pre-model's probabilities while the final `win_model` is retrained on
   all seasons (`modeling.py:412-437`); it is also fit on a mirrored sample that
   counts each game twice (`modeling.py:102-126`), inflating the effective N.
   **Fix:** calibrate via season-grouped out-of-fold predictions from the final
   model; de-duplicate mirrored rows before counting.

5. **External prior is trained and reported in-sample, then blended into outputs**
   (weights 0.20/0.25/0.25) without out-of-sample validation
   (`modeling.py:619-688, 771-805`). **Fix:** report cross-validated metrics; choose
   blend weights from held-out performance.

---

## Tier 2 — Financial / risk correctness

The core odds math is **correct** — `american_to_decimal`,
`expected_value_per_unit`, `kelly_fraction`, `american_profit`,
`fair_prob_from_two_way_moneyline` all handle signs and the zero-odds guard
(`markets.py:72-145`). _(Now covered by unit tests; see `tests/`.)_ The problems
are structural:

1. **Bankroll never compounds.** Kelly always sizes off a frozen `$1000`; settled
   P&L never feeds back (`execution.py:139`; no write-back in `cli.py:423-427`).
   **Fix:** persist a running balance and feed it into sizing.

2. **No correlation cap.** Moneyline + spread + total on the same game are sized
   independently against the same bankroll; a blowout wins all three. The dedupe
   only works *within* a bet type (`execution.py:201`, `ledger.py:256-262`).
   **Fix:** cap aggregate staked fraction per event/slate; shrink correlated legs.

3. **Bets settle against synthetic lines.** Consensus *averages* spreads across
   books to e.g. `-3.17`, a line no book offered, then settles a "real" bet at it
   (`execution.py:122`, `ledger.py:119-129`). **Fix:** stake against an actual
   posted book line/price.

4. **Positional home/away settlement.** `settle_ledger` assumes ledger
   `TeamA == home` and joins by position, not team ID, so any reversed/reoriented
   row settles against the wrong score (`ledger.py:100-120` vs `markets.py:514`).
   **Fix:** settle by joining on `TeamAID`/`TeamBID`.

5. **Proportional de-vig only**, equal-weighting books (`markets.py:90-96`); biases
   favorites high / longshots low. **Fix:** offer Shin or power-method de-vig.

6. **Settlement `profit` conflates lifetime vs this-run P&L** (`ledger.py:154-159`).
   **Fix:** report `realized_profit_this_run` and `lifetime_profit` separately.

---

## Tier 3 — Architecture & maintainability

1. **Split-brain design.** A 4.7k-line monolith (`run_march_madness.py`) *and* a
   parallel `src/mm26_xgb/cli.py`, each with its own divergent argparse CLI and its
   **own** Odds-API HTTP client + key loader (`run_march_madness.py:427` vs
   `markets.py:160`). Bug fixes must be made twice. **Fix:** make `src/mm26_xgb/`
   the single source of truth; shrink the monolith to a thin shim.

2. **~260 lines of dead legacy code** (`_legacy_command_*` + orphaned helpers).
   **DONE** in this change — removed via AST, 292 lines deleted.

3. **Config fragmented across three layers** — `TrainConfig` defaults,
   `config/defaults.json`, and inline `settings.get(k, literal)` with `snapshot_*`
   keys that aren't even in the JSON. **Fix:** one validated settings schema.

4. **No packaging; imports rely on a `sys.path.insert` hack**
   (`run_march_madness.py:29-32`). **PARTLY DONE** — added `pyproject.toml`
   (src-layout, `mm26` console script, dev extras) so `pip install -e .` works; the
   bootstrap hack is retained so `python run_march_madness.py` still runs without an
   install.

5. **Hand-built `argparse.Namespace` objects for inter-command calls** are fragile
   (`run_march_madness.py:2134+`, `interactive_menu`). **Fix:** pass a typed options
   dataclass.

---

## Tier 4 — Engineering hygiene

1. **No automated tests** on a money-handling system. **PARTLY DONE** — added a
   `pytest` suite (79 tests) for the pure money/probability functions in
   `markets.py`, `ledger.py`, `simulation.py`. The modeling/feature pipeline and the
   monolith remain untested.

2. **No CI, no lockfile.** Deps are `>=`-only while shipping a version-sensitive
   `joblib` model — loading under newer xgboost/sklearn can silently mis-predict.
   **Fix:** add a GitHub Actions workflow (pytest + ruff + mypy); commit a
   `requirements.lock`; stamp library versions into `train_config.json`.

3. **`joblib.load` with no integrity check** (`modeling.py:1040`) — pickle executes
   arbitrary code on load. **Fix:** verify a SHA-256 of the bundle; prefer XGBoost's
   native `save_model` for weights.

4. **15 broad `except Exception` blocks** swallow errors into empty frames (all in
   the monolith); `print()` everywhere, no `logging`. **Fix:** narrow exception
   types, log before swallowing, adopt `logging`.

5. **Secrets handling is a strength** — the Odds API key is env/CLI-sourced and
   never printed (`markets.py:149-156`, `run_march_madness.py:174`). Minor: add
   `.env` to `.gitignore` and warn against committing a populated `defaults.json`.

---

## Suggested roadmap

| Priority | Item | Effort |
|---|---|---|
| P0 | Restore a real training + multi-season market-history run; make the backtest build the production model | Days |
| P0 | Convert the bracket to true Monte Carlo with advancement probabilities | ~1 day |
| P1 | Re-wire (or remove) the dead bet-pricing/EV CLI args | Hours |
| P1 | Fix external-feature leakage + train/serve NaN mismatch | ~1 day |
| P1 | Add CI (pytest + ruff + mypy) + `requirements.lock` | Hours |
| P2 | Compound bankroll + per-event correlation/exposure cap | ~1 day |
| P2 | Settle on actual book lines, by team ID | Hours |
| P2 | Collapse the two CLIs; de-duplicate the Odds-API client | Days |

## Addressed in the change that introduced this document

- Removed ~292 lines of confirmed-dead legacy code from `run_march_madness.py`.
- Added `pyproject.toml` (src-layout package, `mm26` console script, `dev` extras).
- Added a 79-test `pytest` suite covering the money/probability functions.
