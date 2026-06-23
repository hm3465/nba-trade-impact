# Phase 1: Fix & Complete NBA Trade Impact Data Pipeline

Fix the curated trade registry, clean up scripts, and run the full pipeline to produce a trustworthy dataset.

## User Review Required

> [!IMPORTANT]
> **Trade Registry Overhaul**: The current registry has ~15 entries that are incorrect, offseason trades, or never happened. I will replace them with verified data (details below). This changes the core dataset your model trains on.

> [!IMPORTANT]
> **New Trades Added**: I'll add several major 2024-25 trades that were missing (Luka/AD, De'Andre Hunter, Dennis Schröder BKN→GSW→DET chain). This should increase coverage significantly.

## Proposed Changes

### 1. Trade Registry Corrections — [collect_trades.py](file:///c:/Users/harsh/nba-trade-impact/src/data/collect_trades.py)

#### Incorrect Destinations (verified via web research):

| Player | Current (Wrong) | Correct | Source |
|--------|-----------------|---------|--------|
| Jimmy Butler | MIA → PHX | MIA → **GSW** | Feb 6, 2025 |
| Brandon Ingram | NOP → MIN | NOP → **TOR** | Feb 7, 2025 |
| Zach LaVine | CHI → DEN | CHI → **SAC** | Feb 6, 2025 |
| Nikola Vučević | CHI → GSW | **NOT TRADED in 2024-25** (traded Feb 2026 to BOS) | Remove |

#### Offseason Trades to Remove (not mid-season):

| Player | Listed Season | Actual Trade Date | Issue |
|--------|---------------|-------------------|-------|
| Alex Caruso | 2023-24 | June 20, 2024 | Offseason |
| Josh Giddey | 2023-24 | June 20, 2024 | Offseason |

#### Trades That Never Happened (remove):

| Player | Listed Entry | Reality |
|--------|-------------|---------|
| Collin Sexton | UTA → BKN (Feb 2025) | Not traded in 2024-25 (traded June 2025 to CHA) |
| Cameron Johnson | BKN → HOU (Feb 2025) | Not traded in 2024-25 (traded June 2025 to DEN) |

#### Wrong Season Entries to Remove:

| Player | Listed Season | Actual Season |
|--------|---------------|---------------|
| Jeff Teague | 2015-16 (ATL → IND) | Offseason trade (July 2016) |
| Brandon Knight | 2015-16 (PHX → MIL) | Feb 2015 = **2014-15 season**, direction wrong (MIL → PHX) |
| Reggie Jackson | 2015-16 (OKC → DET) | Feb 2015 = **2014-15 season** |
| Dillon Brooks | 2022-23 (MEM → HOU) | Offseason sign-and-trade (July 2023) |

#### Correct Midseason Trades to Fix/Keep:

| Player | Correction |
|--------|-----------|
| Dorian Finney-Smith 2024-25 | BKN → LAL on **2024-12-29** (not Feb 2025) |
| Dennis Schröder 2024-25 | BKN → GSW on **2024-12-15** (not Feb 2025) |

#### New Major Trades to Add (2024-25):

| Player | Season | Date | From | To |
|--------|--------|------|------|-----|
| Luka Dončić | 2024-25 | 2025-02-02 | DAL | LAL |
| Anthony Davis | 2024-25 | 2025-02-02 | LAL | DAL |
| De'Andre Hunter | 2024-25 | 2025-02-06 | ATL | CLE |
| Dennis Schröder (2nd trade) | 2024-25 | 2025-02-06 | GSW | DET |

---

### 2. Replace fetch.py — [fetch.py](file:///c:/Users/harsh/nba-trade-impact/src/data/fetch.py)

#### [DELETE] [fetch.py](file:///c:/Users/harsh/nba-trade-impact/src/data/fetch.py)

Replace the LeBron demo script with a proper pipeline runner that executes all three collection scripts in order: `collect_trades → collect_team_stats → collect_game_logs`.

#### [NEW] [run_pipeline.py](file:///c:/Users/harsh/nba-trade-impact/src/data/run_pipeline.py)

A convenience script that runs all Phase 1 data collection steps in sequence with timing and summary output.

---

### 3. Add tqdm Progress Bars

#### [MODIFY] [collect_game_logs.py](file:///c:/Users/harsh/nba-trade-impact/src/data/collect_game_logs.py)
- Add `tqdm` progress bar to the main trade iteration loop
- Add progress bar to team stats collection in `collect_team_stats.py`

#### [MODIFY] [collect_team_stats.py](file:///c:/Users/harsh/nba-trade-impact/src/data/collect_team_stats.py)
- Add `tqdm` progress bar to season iteration loop

---

### 4. Improve USG% Computation

#### [MODIFY] [collect_game_logs.py](file:///c:/Users/harsh/nba-trade-impact/src/data/collect_game_logs.py)
- Load `team_stats.csv` to get actual team PACE values per season
- Use PACE to better estimate team possessions for USG% calculation
- Fall back to the current approximation when team data is unavailable

---

### 5. Fix Player Name Resolution

Some player names in the registry use non-standard characters or suffixes that fail `nba_api` lookup:
- `"Jimmy Butler III"` → try `"Jimmy Butler"` as fallback
- `"Kristaps Porziņģis"` → try ASCII normalization
- `"Enes Freedom"` → handle name changes (was Enes Kanter)
- `"Reggie Bullock Jr."` → try without suffix

Add Unicode normalization and suffix-stripping fallback logic to `resolve_player_id()`.

---

## Verification Plan

### Automated Tests
After implementing all changes, I will:

1. **Run the full pipeline**: `python -m src.data.collect_trades`, then `python -m src.data.collect_team_stats`, then `python -m src.data.collect_game_logs`
2. **Validate output counts**: Check that `trades.csv` has the expected ~90-95 entries (after removing ~12 bad entries and adding ~4 new ones)
3. **Spot-check key trades**: Verify Jimmy Butler → GSW, Brandon Ingram → TOR, Zach LaVine → SAC, Luka → LAL appear correctly
4. **Check split coverage**: Verify that `player_trade_splits.csv` has more entries than the current 59 (corrected trade data should improve matching)
5. **Validate no offseason trades remain**: Confirm all trade dates fall within their season's regular season window

### Manual Verification
- Review the generated CSV files for data quality
- Cross-reference a sample of game log entries with basketball-reference
Here's a clear roadmap broken into phases:

Phase 1 — Data Collection
Goal: Build a dataset of players who were traded mid-season with stats from before and after the trade.

Use nba_api to pull per-game stats for every player, season by season
Use the transaction/trade logs (Basketball Reference has these as CSVs) to identify trade dates
For each traded player, split their season stats into pre-trade and post-trade windows
Save everything to data/raw/ as CSVs

Key columns you want: player_id, season, trade_date, team_before, team_after, pts, reb, ast, ts%, usage_rate, plus_minus, etc.

Phase 2 — Feature Engineering
Goal: Represent the "context shift" a player experiences from a trade.

Compute delta features: post-trade stat − pre-trade stat for each metric
Encode team context: e.g. team offensive rating, pace, coach system before vs. after
Add player features: age, position, games played, career trajectory
This is where the OOD framing lives — the trade is the distribution shift, and your features capture how different the new environment is from the old one


Phase 3 — Modeling
Goal: Predict post-trade performance from pre-trade stats + context shift features.

Start simple: linear regression / ridge regression as a baseline
Then try a gradient boosted model (XGBoost) which tends to work well on tabular sports data
Frame it as: given pre-trade stats and the team context delta, predict post-trade pts, ts%, and usage_rate
Evaluate with MAE and R² on a held-out test set of recent trades


Phase 4 — OOD Analysis
Goal: This is what makes the project research-worthy, not just a Kaggle notebook.

Split your test set by "shift magnitude" — trades where the team context changed a lot vs. a little
Show how your model performs under large vs. small distribution shifts
Optionally: try importance weighting (IWERM) or domain adaptation techniques to improve OOD robustness, directly echoing your flood forecasting work


Phase 5 — FastAPI Backend
Goal: Serve predictions via an API endpoint.

Build a /predict endpoint that takes a player name + two team names and returns forecasted stat changes
Load your trained model from a saved file (joblib or torch.save)
Return a JSON response with predicted post-trade stats and a confidence interval


Phase 6 — (Optional) Frontend
Goal: Make it presentable for a portfolio or demo.

Simple HTML/JS frontend that calls your FastAPI backend
Input: player name, old team, new team
Output: bar chart comparing predicted pre vs. post trade stats


Suggested Order of Attack
WeekFocus1Phase 1 — get the data pipeline working2Phase 2 — feature engineering3Phase 3 — baseline model + evaluation4Phase 4 — OOD analysis5Phase 5 — FastAPI6Phase 6 — polish + README

Want to start on Phase 1 now? Since fetch.py is almost set up, the next step would be writing the logic to pull traded players and split their stats around the trade date.
 don't make any changes but can you elaborate and make a plan on how i need to go about implementing phases 2-6 and make it as detailed and elaboarate as possibel


Show more

12:00 PM






Absolutely. Since Phase 1 is the data foundation, Phases 2-6 should be built like a pipeline: raw trade/game data -> modeling table -> trained models -> OOD evaluation -> API -> demo UI.
Below is a detailed implementation plan for Phases 2-6, assuming Phase 1 eventually gives you reliable CSVs like:
data/raw/trades.csv
data/raw/player_game_logs.csv
data/raw/player_trade_splits.csv
data/raw/team_stats.csv
Phase 2 — Feature Engineering
Goal: turn your raw pre/post trade data into a clean modeling dataset where every row is one traded player instance.
You want a file like:
data/processed/modeling_dataset.csv
Each row should represent:
one player + one season + one trade
Example:
James Harden, 2021-22, BKN -> PHI
The row should include:
pre-trade player stats
post-trade target stats
team context before trade
team context after trade
context deltas
player metadata
shift magnitude features
2.1 Create a Feature Engineering Script
Create something like:
src/features/build_features.py
This script should read:
data/raw/player_trade_splits.csv
data/raw/team_stats.csv
data/raw/trades.csv
and write:
data/processed/modeling_dataset.csv
Suggested structure:
src/
  features/
    __init__.py
    build_features.py
The script should do four main things:
Load pre/post split data.
Join team context stats for team_before and team_after.
Compute player stat deltas and team-context deltas.
Save a final model-ready table.
2.2 Decide Prediction Targets
You need to be very clear about what the model is predicting.
A good first version should predict:
post_pts
post_ts_pct
post_usg_rate
Those are strong because they capture:
volume: points per game
efficiency: true shooting percentage
role: usage rate
You can later add:
post_ast
post_reb
post_min
post_plus_minus
But for the first complete project, keep the target set small.
Recommended targets:
target_pts = post_pts
target_ts_pct = post_ts_pct
target_usg_rate = post_usg_rate
You can also model deltas instead:
delta_pts = post_pts - pre_pts
delta_ts_pct = post_ts_pct - pre_ts_pct
delta_usg_rate = post_usg_rate - pre_usg_rate
My recommendation: predict deltas first, then reconstruct post-trade stats.
Why?
Because your problem is really:
How does a player change after the trade?
So the model target should be:
change in performance
For example:
predicted_post_pts = pre_pts + predicted_delta_pts
This makes the project feel much more aligned with “trade impact forecasting.”
2.3 Create Player-Level Features
From player_trade_splits.csv, use pre-trade values as model inputs:
pre_games
pre_min
pre_pts
pre_reb
pre_ast
pre_stl
pre_blk
pre_tov
pre_fg_pct
pre_fg3_pct
pre_ft_pct
pre_ts_pct
pre_usg_rate
pre_plus_minus
Also include rate-style features:
pre_pts_per_min = pre_pts / pre_min
pre_ast_per_min = pre_ast / pre_min
pre_reb_per_min = pre_reb / pre_min
pre_tov_per_min = pre_tov / pre_min
These help separate role from efficiency.
For example, two players might both average 15 PPG, but one does it in 22 minutes and another in 34 minutes. That distinction matters.
2.4 Add Team Context Features
From team_stats.csv, join old-team context and new-team context.
For each trade row:
team_before = team_from
team_after = team_to
season = season
Join old team stats:
old_off_rating
old_def_rating
old_net_rating
old_pace
old_ts_pct
old_efg_pct
old_ast_pct
old_tov_pct
old_reb_pct
Join new team stats:
new_off_rating
new_def_rating
new_net_rating
new_pace
new_ts_pct
new_efg_pct
new_ast_pct
new_tov_pct
new_reb_pct
Then compute deltas:
delta_team_off_rating = new_off_rating - old_off_rating
delta_team_def_rating = new_def_rating - old_def_rating
delta_team_net_rating = new_net_rating - old_net_rating
delta_team_pace = new_pace - old_pace
delta_team_ts_pct = new_ts_pct - old_ts_pct
delta_team_ast_pct = new_ast_pct - old_ast_pct
delta_team_tov_pct = new_tov_pct - old_tov_pct
These are central to the OOD framing.
The “distribution shift” is:
old team environment -> new team environment
The features should quantify how different that shift is.
2.5 Add Shift Magnitude Features
Create a single score that estimates how large the context shift is.
Simple version:
shift_magnitude =
  abs(delta_team_off_rating)
+ abs(delta_team_def_rating)
+ abs(delta_team_pace)
+ abs(delta_team_ast_pct)
+ abs(delta_team_ts_pct)
Better version:
standardize each context delta first:
z_delta_off_rating
z_delta_def_rating
z_delta_pace
z_delta_ast_pct
z_delta_ts_pct
Then compute Euclidean distance:
shift_magnitude = sqrt(
    z_delta_off_rating^2
  + z_delta_def_rating^2
  + z_delta_pace^2
  + z_delta_ast_pct^2
  + z_delta_ts_pct^2
)
This gives you a clean OOD analysis variable later.
You can then bucket trades:
small_shift
medium_shift
large_shift
For example:
small_shift: bottom 33%
medium_shift: middle 33%
large_shift: top 33%
2.6 Add Player Metadata
Useful player features:
age_at_trade
position
height
weight
years_experience
career_games_before_trade
career_minutes_before_trade
Some of these are harder to collect cleanly, so don’t block the project on all of them.
Minimum viable version:
age
position
season
You can get these from nba_api player profile endpoints or a static player metadata table.
Important: avoid leaking post-trade information into the feature set.
Bad feature:
post_min
as an input to predict post_pts.
Why? Because after the trade, you wouldn’t know post-trade minutes yet.
Better:
pre_min
old_team_context
new_team_context
player_age
position
2.7 Handle Missing Data
You’ll need a consistent missing-data strategy.
For numeric columns:
fill missing with median
For categorical columns:
fill missing with "Unknown"
But in the modeling pipeline, do this with sklearn transformers, not manually everywhere.
Recommended approach:
SimpleImputer(strategy="median") for numeric features
SimpleImputer(strategy="most_frequent") for categorical features
OneHotEncoder(handle_unknown="ignore") for categorical features
StandardScaler() for linear models
2.8 Output Files for Phase 2
By the end of Phase 2, you want:
data/processed/modeling_dataset.csv
data/processed/feature_columns.json
data/processed/target_columns.json
The modeling dataset should be easy to inspect manually.
Example columns:
player_id
player_name
season
trade_date
team_from
team_to

pre_pts
pre_reb
pre_ast
pre_ts_pct
pre_usg_rate
pre_min
pre_plus_minus

old_off_rating
old_def_rating
old_pace
old_net_rating

new_off_rating
new_def_rating
new_pace
new_net_rating

delta_team_off_rating
delta_team_def_rating
delta_team_pace
delta_team_net_rating

shift_magnitude
shift_bucket

target_delta_pts
target_delta_ts_pct
target_delta_usg_rate
Phase 3 — Modeling
Goal: predict post-trade performance or performance deltas from pre-trade stats and context-shift features.
You want scripts like:
src/models/train_baseline.py
src/models/train_xgboost.py
src/models/evaluate.py
And outputs like:
models/ridge_delta_pts.joblib
models/ridge_delta_ts_pct.joblib
models/ridge_delta_usg_rate.joblib

reports/metrics/baseline_metrics.json
reports/figures/predicted_vs_actual_pts.png
3.1 Decide Modeling Framing
Best first framing:
Inputs:
  pre-trade player stats
  old team context
  new team context
  team context deltas
  player metadata

Targets:
  delta_pts
  delta_ts_pct
  delta_usg_rate
Then reconstruct:
predicted_post_pts = pre_pts + predicted_delta_pts
predicted_post_ts_pct = pre_ts_pct + predicted_delta_ts_pct
predicted_post_usg_rate = pre_usg_rate + predicted_delta_usg_rate
This gives interpretable forecasts.
Example:
Player averaged 18.2 PPG before trade.
Model predicts +2.1 PPG after trade.
Forecast: 20.3 PPG.
3.2 Train/Test Split
Do not randomly split at first.
Because this is a time-based sports forecasting problem, use a chronological split.
Example:
Train:
2015-16 through 2021-22

Validation:
2022-23

Test:
2023-24 through 2024-25
If your dataset is small, use:
Train:
2015-16 through 2022-23

Test:
2023-24 through 2024-25
This is more realistic because you’re asking:
Can past trades predict future trades?
You can later add cross-validation, but chronological split is the right first evaluation.
3.3 Start With Baselines
Before ML, create dumb baselines.
Baseline 1:
No-change baseline
predicted_post_pts = pre_pts
predicted_post_ts_pct = pre_ts_pct
predicted_post_usg_rate = pre_usg_rate
Equivalent:
predicted_delta = 0
This is a crucial baseline. Your model must beat it.
Baseline 2:
Average historical delta
Example:
predicted_delta_pts = mean training delta_pts
Baseline 3:
Position-level average delta
Example:
guards historically gain +0.8 AST
bigs historically lose -1.2 USG
Only then train real models.
3.4 Train Ridge Regression
First real model:
Ridge regression
Why Ridge?
Because:
small dataset
many correlated stats
good baseline
interpretable
harder to overfit than plain linear regression
Use separate models per target:
ridge_delta_pts
ridge_delta_ts_pct
ridge_delta_usg_rate
Or use multi-output regression:
MultiOutputRegressor(Ridge())
I’d start with separate models because it makes debugging easier.
Pipeline:
ColumnTransformer:
  numeric columns -> imputer -> scaler
  categorical columns -> imputer -> one-hot encoder

Model:
  Ridge(alpha=...)
Tune alpha lightly:
alpha values: 0.1, 1.0, 10.0, 100.0
3.5 Train Tree-Based Models
Next model:
XGBoost
If you don’t want to add XGBoost immediately, use sklearn’s:
HistGradientBoostingRegressor
RandomForestRegressor
GradientBoostingRegressor
XGBoost is good for tabular sports data, but it adds dependency friction. A nice path:
Ridge
Random Forest
XGBoost
For XGBoost:
max_depth: 2-4
learning_rate: 0.03-0.1
n_estimators: 100-500
subsample: 0.7-1.0
colsample_bytree: 0.7-1.0
Keep it conservative because the dataset is likely small.
3.6 Metrics
For each target, compute:
MAE
RMSE
R²
Use MAE as your primary metric.
Why?
Because it’s intuitive:
Model is off by 2.4 points per game on average.
Model is off by 0.037 TS% on average.
Model is off by 3.1 usage points on average.
Example metrics table:
target              baseline_mae   ridge_mae   xgb_mae
delta_pts           3.10           2.72        2.51
delta_ts_pct        0.048          0.043       0.041
delta_usg_rate      4.20           3.75        3.60
Also compute directional accuracy:
Did the model correctly predict whether the stat would go up or down?
Example:
delta_pts_direction_accuracy = 64%
This is very portfolio-friendly because it translates well.
3.7 Model Diagnostics
Generate plots:
predicted vs actual delta_pts
residuals by season
residuals by shift_magnitude
feature importance
top overpredictions
top underpredictions
Useful files:
reports/figures/pred_vs_actual_delta_pts.png
reports/figures/residuals_by_shift_bucket.png
reports/figures/feature_importance_xgb.png
reports/tables/biggest_errors.csv
The “biggest errors” table is valuable because it lets you reason about edge cases.
Columns:
player_name
season
team_from
team_to
actual_delta_pts
predicted_delta_pts
error
shift_magnitude
3.8 Save Models
Use joblib.
Suggested output:
models/
  ridge_delta_pts.joblib
  ridge_delta_ts_pct.joblib
  ridge_delta_usg_rate.joblib
  xgb_delta_pts.joblib
  xgb_delta_ts_pct.joblib
  xgb_delta_usg_rate.joblib
  preprocessing_pipeline.joblib
  model_metadata.json
model_metadata.json should include:
{
  "trained_at": "2026-06-23",
  "train_seasons": ["2015-16", "..."],
  "test_seasons": ["2023-24", "2024-25"],
  "targets": ["delta_pts", "delta_ts_pct", "delta_usg_rate"],
  "features": ["pre_pts", "pre_ts_pct", "..."],
  "model_type": "ridge"
}
Phase 4 — OOD Analysis
Goal: show how models behave when the trade creates a large context shift.
This is the most interesting part of the project.
The core research question:
Do trade-impact models perform worse when players move into very different team environments?
4.1 Define Distribution Shift
In your project, a “distribution shift” is not abstract. It is concrete:
The player moves from one team context to another.
A small shift might be:
similar pace
similar offensive rating
similar usage ecosystem
similar spacing
A large shift might be:
slow team -> fast team
bad offense -> elite offense
low-assist isolation offense -> high-ball-movement offense
lottery team -> contender
You quantify this with shift_magnitude.
4.2 Create Shift Buckets
In Phase 2, you created:
shift_magnitude
Now bucket the test set:
small_shift
medium_shift
large_shift
Use quantiles:
small: 0-33rd percentile
medium: 33rd-66th percentile
large: 66th-100th percentile
Then evaluate separately:
MAE for small shifts
MAE for medium shifts
MAE for large shifts
Example table:
bucket          n     delta_pts_mae   delta_ts_pct_mae   delta_usg_mae
small_shift     18    1.9             0.031              2.4
medium_shift    18    2.6             0.044              3.2
large_shift     18    4.1             0.061              5.0
This is the money table.
If performance degrades on large shifts, you can say:
Model accuracy declines as team-context distribution shift increases.
That is your OOD story.
4.3 Compare Model Types Under Shift
Evaluate each model by bucket:
No-change baseline
Ridge
XGBoost
You may find:
Ridge is more stable
XGBoost performs better overall but worse under large shift
That would be an interesting finding.
Example:
model       small_shift_mae   large_shift_mae
baseline    2.3               4.8
ridge       2.0               4.0
xgboost     1.8               4.5
Interpretation:
XGBoost wins in-distribution but Ridge generalizes better under large shifts.
That is exactly the kind of thing that makes the project feel research-oriented.
4.4 Try Importance Weighting
Optional, but powerful.
The idea:
Give more training weight to examples that resemble large-shift test cases.
Simple implementation:
sample_weight = 1 + shift_magnitude
or:
large_shift rows get weight 2.0
medium_shift rows get weight 1.5
small_shift rows get weight 1.0
Train weighted Ridge or XGBoost.
Then compare:
unweighted model vs weighted model
Metrics:
overall MAE
large-shift MAE
small-shift MAE
You might find weighted training improves large-shift performance but slightly hurts small-shift performance. That is a nice tradeoff to discuss.
4.5 OOD Report
Create:
reports/ood_analysis.md
Include:
1. Definition of distribution shift
2. How shift_magnitude is computed
3. Performance by shift bucket
4. Error analysis
5. Whether importance weighting helped
6. Limitations
Suggested figures:
reports/figures/mae_by_shift_bucket.png
reports/figures/error_vs_shift_magnitude.png
reports/figures/shift_magnitude_distribution.png
4.6 Key OOD Claims You Can Make
Good claims:
The model performs best when players move between similar team environments.
Prediction error increases for large context shifts.
Team pace, offensive rating, and assist rate deltas are meaningful predictors of post-trade role changes.
Importance weighting modestly improves robustness on large-shift examples.
Avoid overclaiming:
This proves causality.
This predicts all future trades reliably.
This fully captures team system fit.
Better wording:
This provides an empirical framework for estimating trade impact under measurable team-context shifts.
Phase 5 — FastAPI Backend
Goal: expose the model through an API.
The backend should load trained models and return predictions for a hypothetical trade.
Suggested structure:
app/
  main.py
  schemas.py
  model_service.py
  feature_service.py
Or:
src/api/
  main.py
  schemas.py
  predictor.py
5.1 Define the API Input
Basic request:
{
  "player_name": "Zach LaVine",
  "season": "2024-25",
  "team_from": "CHI",
  "team_to": "SAC"
}
But the model needs pre-trade stats.
You have two options.
Option A: API accepts only names/teams, then looks up the latest data internally.
{
  "player_name": "Zach LaVine",
  "season": "2024-25",
  "team_from": "CHI",
  "team_to": "SAC"
}
Option B: API accepts full feature inputs.
{
  "player_name": "Zach LaVine",
  "season": "2024-25",
  "team_from": "CHI",
  "team_to": "SAC",
  "pre_pts": 24.0,
  "pre_reb": 4.8,
  "pre_ast": 4.5,
  "pre_ts_pct": 0.578,
  "pre_usg_rate": 28.1,
  "pre_min": 34.2
}
Recommendation: start with Option A for demo friendliness.
Internally, the API can look up the player’s pre-trade row from your processed dataset.
5.2 Define Endpoints
Minimum endpoints:
GET /
GET /health
POST /predict
Example:
GET /health
Returns:
{
  "status": "ok",
  "model_loaded": true
}
Prediction endpoint:
POST /predict
Input:
{
  "player_name": "James Harden",
  "season": "2021-22",
  "team_from": "BKN",
  "team_to": "PHI"
}
Output:
{
  "player_name": "James Harden",
  "season": "2021-22",
  "team_from": "BKN",
  "team_to": "PHI",
  "predictions": {
    "pts": {
      "pre": 22.5,
      "predicted_delta": 1.8,
      "predicted_post": 24.3
    },
    "ts_pct": {
      "pre": 0.576,
      "predicted_delta": 0.012,
      "predicted_post": 0.588
    },
    "usage_rate": {
      "pre": 27.4,
      "predicted_delta": -1.9,
      "predicted_post": 25.5
    }
  },
  "context_shift": {
    "shift_magnitude": 1.42,
    "shift_bucket": "large_shift",
    "delta_team_pace": 2.1,
    "delta_team_off_rating": 4.5
  }
}
5.3 Confidence Intervals
A simple first approach:
prediction interval = predicted value ± validation MAE
Example:
predicted_delta_pts = +2.0
validation_mae_pts = 2.7
interval = [-0.7, +4.7]
Not statistically perfect, but understandable.
Better later:
bootstrap models
quantile regression
conformal prediction
For a portfolio project, validation-MAE intervals are okay if you clearly label them as approximate.
5.4 Model Loading
In model_service.py, load models once at app startup:
models/ridge_delta_pts.joblib
models/ridge_delta_ts_pct.joblib
models/ridge_delta_usg_rate.joblib
Also load:
data/processed/modeling_dataset.csv
models/model_metadata.json
The service should expose:
predict_trade_impact(player_name, season, team_from, team_to)
5.5 Feature Construction for API
This is important.
The API prediction row must go through the same feature construction logic as training.
Avoid duplicating logic manually.
Best structure:
src/features/build_features.py
contains reusable functions like:
build_modeling_dataset()
build_single_prediction_row()
Then API uses:
build_single_prediction_row(player_name, season, team_from, team_to)
This prevents train/serve skew.
Train/serve skew means:
training features are computed one way
API features are computed a slightly different way
That silently breaks models.
5.6 Backend Validation
Use Pydantic schemas.
Input validation:
player_name: str
season: str
team_from: str
team_to: str
Validate:
season format looks like "2023-24"
team abbreviations are valid
player exists in dataset
model is loaded
Return helpful errors:
{
  "detail": "Player not found in processed dataset for selected season."
}
5.7 Backend File Layout
Suggested:
src/
  api/
    __init__.py
    main.py
    schemas.py
    model_service.py
    feature_service.py

models/
  ridge_delta_pts.joblib
  ridge_delta_ts_pct.joblib
  ridge_delta_usg_rate.joblib
  model_metadata.json
Run with:
uvicorn src.api.main:app --reload
5.8 Backend Tests
Add simple tests:
tests/test_api.py
Test:
GET /health returns 200
POST /predict with known player returns 200
POST /predict with unknown player returns 404 or 422
prediction contains pts, ts_pct, usage_rate
Phase 6 — Optional Frontend
Goal: make the project easy to demo.
Keep this simple. The backend and model are the real project.
6.1 Frontend Scope
The frontend should have:
player input
season dropdown
old team dropdown
new team dropdown
predict button
result cards/table
bar chart comparing pre vs predicted post
context shift summary
Do not overbuild it.
6.2 Frontend Options
Simplest:
static HTML + CSS + JavaScript
More polished:
React + Vite
For a portfolio demo, React/Vite is fine, but static HTML is enough if the backend is the focus.
Suggested if using React:
frontend/
  package.json
  src/
    App.jsx
    api.js
    components/
      PredictionForm.jsx
      PredictionChart.jsx
      ContextShiftPanel.jsx
6.3 Frontend Layout
One clean app screen:
Header:
NBA Trade Impact Forecaster

Left panel:
Form inputs

Right panel:
Prediction result

Bottom:
Chart + context shift details
The user flow:
Select player.
Select old team.
Select new team.
Click Predict.
See forecasted stat changes.
6.4 Visual Output
Show a bar chart:
PTS: pre vs predicted post
TS%: pre vs predicted post
USG%: pre vs predicted post
Use:
Chart.js
Recharts
Plotly
For a React frontend, Recharts is very easy.
Example chart:
Metric       Pre      Predicted Post
PTS          22.5     24.3
TS%          57.6     58.8
USG%         27.4     25.5
Also show deltas:
PTS: +1.8
TS%: +1.2 percentage points
USG%: -1.9
6.5 Show Context Shift
This is where you make the OOD angle visible.
Display:
Shift magnitude: Large
Pace change: +2.1
Offensive rating change: +4.5
Assist percentage change: +3.2
Maybe add a small badge:
Large Context Shift
This helps users understand that the model is not just predicting from player averages. It is considering team environment.
6.6 Frontend API Call
Frontend calls:
POST http://localhost:8000/predict
with:
{
  "player_name": "James Harden",
  "season": "2021-22",
  "team_from": "BKN",
  "team_to": "PHI"
}
Then renders the JSON response.
6.7 Frontend Edge Cases
Handle:
loading state
API error
player not found
missing model
empty prediction
Example message:
No matching trade found for that player/team/season combination.
6.8 Optional Demo Presets
Add a few preset buttons:
James Harden: BKN -> PHI
Kyrie Irving: BKN -> DAL
Pascal Siakam: TOR -> IND
OG Anunoby: TOR -> NYK
These make the demo smoother.
Suggested Repository Structure
A solid final structure would be:
nba-trade-impact/
  data/
    raw/
      trades.csv
      player_game_logs.csv
      player_trade_splits.csv
      team_stats.csv
    processed/
      modeling_dataset.csv
      feature_columns.json
      target_columns.json

  models/
    ridge_delta_pts.joblib
    ridge_delta_ts_pct.joblib
    ridge_delta_usg_rate.joblib
    model_metadata.json

  reports/
    metrics/
      baseline_metrics.json
      model_metrics.json
      ood_metrics.json
    figures/
      pred_vs_actual_delta_pts.png
      mae_by_shift_bucket.png
      error_vs_shift_magnitude.png
    ood_analysis.md

  src/
    data/
      collect_trades.py
      collect_game_logs.py
      collect_team_stats.py

    features/
      build_features.py

    models/
      train_baseline.py
      train_model.py
      evaluate.py
      ood_analysis.py

    api/
      main.py
      schemas.py
      model_service.py
      feature_service.py

  frontend/
    package.json
    src/
      App.jsx
      api.js
      components/

  tests/
    test_features.py
    test_models.py
    test_api.py

  README.md
  requirements.txt
Implementation Order
I would do the rest in this order:
Clean Phase 1 first
Before Phase 2, fix the trade registry. The current model will only be as good as the trade data. Incorrect trades will poison everything downstream.

Build modeling_dataset.csv
This is the central artifact. Do not train anything until this file is clean and inspectable.

Train no-change baseline
This gives you the minimum bar.

Train Ridge
Keep it simple and interpretable.

Evaluate chronologically
Use recent trades as test data.

Add OOD shift buckets
Evaluate model performance by small_shift, medium_shift, large_shift.

Try XGBoost
Compare against Ridge and baseline.

Write OOD report
This is what turns the project from “sports prediction notebook” into “distribution shift analysis.”

Build FastAPI
Serve predictions from the best saved model.

Add frontend only after API works

   The frontend should visualize a working model, not compensate for an unfinished backend.
The Most Important Design Choice
The key choice is whether you predict:
post-trade stats directly
or:
change from pre-trade to post-trade
I strongly recommend predicting deltas:
delta_pts
delta_ts_pct
delta_usg_rate
because the project is about trade impact.
That lets you say:
Given this player’s pre-trade profile and the context shift from Team A to Team B, the model pred