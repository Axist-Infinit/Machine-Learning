# Legacy pipeline (frozen)

This is the pre-2027 `run_march_madness.py` monolith plus the `mm26_xgb` package. It's kept
for reference only: it is unmaintained, excluded from CI, and scheduled for deletion after
milestone M5 of `docs/PLAN_2027.md`.

**Do not use it for predictions.** Known defects (details in `docs/PLAN_2027.md` and
`docs/REVIEW_2026-06.md`):

- **Market-line bug.** `mm26_xgb/modeling.py:340` sets `MarginBase = MarketSpreadTeamA`, which has the wrong sign. The line is then added back onto models trained almost entirely without lines (`:575-577`, `:736-738`).
  - In the 2025 backtest, predicted totals ran about 20 points high (First Four games came out 105–105) and margins collapsed to about 0.
- **Leakage.** Snapshot training rows contain their own outcomes (end-of-window aggregates; `CurrTourney*`). Lines are merged by team pair only, and "closing" lines can come from after tip-off.
- **Name matching.** The fuzzy matcher (Miami OH → Miami FL) silently drops names it can't resolve.
- **Bracket and betting.** The bracket is a single chalk pass, and the betting flags are no-ops.

The legacy unit tests still pass when run by hand: `python3 -m pytest legacy/tests`.
The monolith expects its old directory layout, so it's not guaranteed to run from here.
