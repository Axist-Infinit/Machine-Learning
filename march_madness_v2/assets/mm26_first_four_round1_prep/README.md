# 2026 Men's March Madness — First Four + First Round prep bundle

This bundle is the cleaned schedule/odds companion for the **2026 men's NCAA tournament First Four and Round of 64**.

It was built from:
- the official 2026 NCAA bracket + tip-time slate
- your uploaded `odds_api_ncaab_companion_out.zip`
- your uploaded Kaggle `march-machine-learning-mania-2026` data for canonical team names and TeamIDs

## What was fixed

The uploaded odds bundle had one bad team resolution:
- raw event: `SMU Mustangs` vs `Miami (OH) RedHawks`
- flattened row incorrectly resolved `Miami (OH)` to `Miami FL`

This bundle corrects that row to **Miami OH**.

## Files

### Direct model-input CSVs
These use the exact columns your model expects:

- `mm26_first_four_2026_model_input.csv`
  - 4 First Four games
  - all 4 rows already have market lines

- `mm26_first_round_locked_28_2026_model_input.csv`
  - the 28 Round-of-64 games that already have fixed opponents
  - all 28 rows already have market lines

- `mm26_first_four_plus_locked_round1_32_2026_model_input.csv`
  - 4 First Four + 28 locked first-round games
  - best file to run right now if you want all games currently known with lines

- `mm26_first_round_playin_scenarios_8_2026_model_input.csv`
  - the 8 possible winner-dependent first-round matchups:
    - Michigan vs UMBC / Howard
    - BYU vs Texas / NC State
    - Tennessee vs Miami OH / SMU
    - Florida vs Prairie View / Lehigh
  - market-line columns are blank because those exact opponent-specific lines were not in the uploaded free-tier odds pull

- `mm26_first_four_and_round1_all_scenarios_40_2026_model_input.csv`
  - 4 First Four + 28 locked first-round games + 8 play-in scenarios
  - useful if you want to score every possible path immediately

### Reference / audit CSVs
- `mm26_first_round_after_first_four_patch_template_4.csv`
  - fill the actual play-in winners here after the First Four ends
  - once lines are posted, add them here and append these 4 rows to the locked-28 file

- `mm26_first_round_official_reference_32_2026.csv`
  - official first-round reference sheet with 32 rows
  - includes the 4 pending-TBD games

- `mm26_first_four_and_round1_annotated_schedule.csv`
  - full audit sheet with round, date, time, region, slot, canonical names, official display names, TeamIDs, odds-match status, and market fields

- `mm26_cleaned_tournament_odds_subset_2026.csv`
  - cleaned tournament-only odds subset from your uploaded odds bundle
  - includes the corrected Miami OH row

- `summary.json`
  - row counts and correction summary

## How to run with your model package

Assuming:
- Kaggle data is under `./kaggle/data`
- the trained model artifact is `./trained_kaggle_market_model`
- your installed CLI command is `mm26`

### 1) Predict the First Four
```bash
mm26 predict-games   --data-dir ./kaggle/data   --artifact-dir ./trained_kaggle_market_model   --matchups-csv ./mm26_first_four_2026_model_input.csv   --output-csv ./predictions/first_four_preds.csv
```

### 2) Predict the 28 locked first-round games
```bash
mm26 predict-games   --data-dir ./kaggle/data   --artifact-dir ./trained_kaggle_market_model   --matchups-csv ./mm26_first_round_locked_28_2026_model_input.csv   --output-csv ./predictions/first_round_locked_28_preds.csv
```

### 3) Predict all currently known games in one shot
```bash
mm26 predict-games   --data-dir ./kaggle/data   --artifact-dir ./trained_kaggle_market_model   --matchups-csv ./mm26_first_four_plus_locked_round1_32_2026_model_input.csv   --output-csv ./predictions/first_four_plus_locked_round1_preds.csv
```

### 4) Predict the 8 winner-dependent scenarios right now
```bash
mm26 predict-games   --data-dir ./kaggle/data   --artifact-dir ./trained_kaggle_market_model   --matchups-csv ./mm26_first_round_playin_scenarios_8_2026_model_input.csv   --output-csv ./predictions/first_round_playin_scenarios_preds.csv
```

### 5) After the First Four ends
- fill the winners into `mm26_first_round_after_first_four_patch_template_4.csv`
- add market spread / total / moneylines once you have them
- append those 4 rows to the locked-28 file
- run `mm26 predict-games` on the completed 32-game first-round file

## Important behavior
- Rows with `MarketSpreadTeamA` and `MarketTotal` let the model compute cover / over probabilities.
- Rows with blank market columns still let the model output:
  - win probabilities
  - predicted winner
  - fair spread
  - fair total
  - predicted scores
- The 8 play-in-scenario rows currently have blank market columns by design.

## Column convention
For every model-input CSV:
- `TeamA` is the official left-side / higher-seed team in bracket order when applicable
- `MarketSpreadTeamA` is from **TeamA's perspective**
  - negative => TeamA favored
  - positive => TeamA underdog
