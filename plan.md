# NBA Trade Impact Forecaster — Project Roadmap

A six-phase build: raw trade/game data → modeling table → trained models → OOD evaluation → API → demo UI.

**Core design choice:** predict *deltas* (post-trade minus pre-trade stats), not raw post-trade stats. This keeps the project framed around trade impact rather than generic stat prediction, and lets you reconstruct post-trade values as `pre_stat + predicted_delta`.

---

## Phase 1 — Data Collection
**Status: in progress (registry fixes + pipeline cleanup underway)**
**Goal:** a dataset of mid-season-traded players with stats before and after the trade.

- Pull per-game stats season by season via `nba_api`
- Use trade/transaction logs (Basketball Reference) to identify trade dates
- For each traded player, split season stats into pre-trade and post-trade windows
- Save outputs to `data/raw/`

**Key columns:** `player_id, season, trade_date, team_before, team_after, pts, reb, ast, ts_pct, usage_rate, plus_minus, ...`

**Outputs:**
```
data/raw/trades.csv
data/raw/player_game_logs.csv
data/raw/player_trade_splits.csv
data/raw/team_stats.csv
```

---

## Phase 2 — Feature Engineering
**Goal:** turn raw pre/post data into one clean modeling table, where each row = one player + one trade.

**Script:** `src/features/build_features.py`
Reads `trades.csv`, `player_trade_splits.csv`, `team_stats.csv` → writes `data/processed/modeling_dataset.csv`

| Step | What it does |
|---|---|
| 1 | Load pre/post split data |
| 2 | Join team context stats for `team_before` and `team_after` |
| 3 | Compute player stat deltas and team-context deltas |
| 4 | Save the final model-ready table |

### 2.1 Prediction targets
Predict **deltas**, then reconstruct post-trade stats:
```
delta_pts      = post_pts - pre_pts
delta_ts_pct   = post_ts_pct - pre_ts_pct
delta_usg_rate = post_usg_rate - pre_usg_rate
...
predicted_post_pts = pre_pts + predicted_delta_pts
```
Keep the target set small for v1 (pts, ts%, usage). Add ast/reb/min/plus_minus later.

### 2.2 Player-level input features (pre-trade only)
```
pre_games, pre_min, pre_pts, pre_reb, pre_ast, pre_stl, pre_blk, pre_tov,
pre_fg_pct, pre_fg3_pct, pre_ft_pct, pre_ts_pct, pre_usg_rate, pre_plus_minus
```
Plus rate-based features to separate role from efficiency:
```
pre_pts_per_min, pre_ast_per_min, pre_reb_per_min, pre_tov_per_min
```

> ⚠️ **No leakage:** never use post-trade values (e.g. `post_min`) as inputs. Only things known *before* the trade resolves are valid features.

### 2.3 Team context features (joined from `team_stats.csv`)
Old team and new team, each with:
```
off_rating, def_rating, net_rating, pace, ts_pct, efg_pct, ast_pct, tov_pct, reb_pct
```
Then compute deltas:
```
delta_team_off_rating, delta_team_def_rating, delta_team_net_rating,
delta_team_pace, delta_team_ts_pct, delta_team_ast_pct, delta_team_tov_pct
```
These deltas **are the OOD signal** — they quantify the size of the environment shift.

### 2.4 Shift magnitude
Standardize each context delta (z-score), then combine into one score:
```
shift_magnitude = sqrt(
    z_delta_off_rating² + z_delta_def_rating² + z_delta_pace²
  + z_delta_ast_pct² + z_delta_ts_pct²
)
```
Bucket into `small_shift` / `medium_shift` / `large_shift` by tercile. This bucketing is what Phase 4 evaluates against.

### 2.5 Player metadata (minimum viable)
```
age, position, season
```
(Height, weight, years of experience, career totals are nice-to-have — don't block on them.)

### 2.6 Missing data strategy
Use `sklearn` transformers, not manual fills:
- Numeric → `SimpleImputer(strategy="median")`
- Categorical → `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`
- Linear models → add `StandardScaler()`

### 2.7 Outputs
```
data/processed/modeling_dataset.csv
data/processed/feature_columns.json
data/processed/target_columns.json
```

---

## Phase 3 — Modeling
**Goal:** predict performance deltas from pre-trade stats + context-shift features.

### 3.1 Train/test split — chronological, not random
This is a forecasting problem; test on the future, not a random shuffle.
```
Train:      2015-16 → 2021-22 (or → 2022-23 if dataset is small)
Validation: 2022-23
Test:       2023-24 → 2024-25
```

### 3.2 Baselines (build before any ML)
| Baseline | Definition |
|---|---|
| No-change | `predicted_delta = 0` (your model must beat this) |
| Historical average delta | mean training delta per target |
| Position-level average delta | e.g. guards historically gain assists, bigs lose usage |

### 3.3 Models, in order of complexity
1. **Ridge regression** — interpretable, robust on a small, correlated-feature dataset. Train one model per target (simpler to debug than multi-output). Tune `alpha ∈ {0.1, 1, 10, 100}`.
2. **Random Forest / HistGradientBoostingRegressor** — sklearn-native tree baseline, no extra dependency.
3. **XGBoost** — typically strongest on tabular sports data. Keep it conservative given likely small data: `max_depth 2-4`, `learning_rate 0.03-0.1`, `n_estimators 100-500`, `subsample/colsample 0.7-1.0`.

### 3.4 Metrics
Primary: **MAE** (intuitive — "off by 2.4 PPG on average"). Also report RMSE, R², and **directional accuracy** (did the model get the sign of the change right?).

Example reporting table:
| target | baseline_mae | ridge_mae | xgb_mae |
|---|---|---|---|
| delta_pts | 3.10 | 2.72 | 2.51 |
| delta_ts_pct | 0.048 | 0.043 | 0.041 |
| delta_usg_rate | 4.20 | 3.75 | 3.60 |

### 3.5 Diagnostics to generate
- Predicted vs. actual scatter (per target)
- Residuals by season, residuals by shift bucket
- Feature importance (tree models)
- "Biggest errors" table: `player_name, season, team_from, team_to, actual_delta, predicted_delta, error, shift_magnitude`

### 3.6 Outputs
```
models/ridge_delta_pts.joblib
models/ridge_delta_ts_pct.joblib
models/ridge_delta_usg_rate.joblib
models/xgb_delta_pts.joblib  (+ ts_pct, usg_rate)
models/preprocessing_pipeline.joblib
models/model_metadata.json
reports/metrics/baseline_metrics.json
reports/figures/*.png
```

---

## Phase 4 — OOD Analysis
**Goal:** the research-worthy part — show how model accuracy degrades as the team-context shift grows.

**Core question:** *Do trade-impact predictions get worse when a player moves into a very different team environment?*

### 4.1 What "distribution shift" means here
Concretely: the player's team context (pace, offensive rating, usage ecosystem) changes. Small shift = similar system to similar system. Large shift = e.g. slow team → fast team, isolation offense → ball-movement offense, lottery team → contender.

### 4.2 Evaluate by shift bucket
Using the `shift_magnitude` buckets from Phase 2, compute MAE per bucket per model:

| bucket | n | delta_pts_mae | delta_ts_pct_mae | delta_usg_mae |
|---|---|---|---|---|
| small_shift | 18 | 1.9 | 0.031 | 2.4 |
| medium_shift | 18 | 2.6 | 0.044 | 3.2 |
| large_shift | 18 | 4.1 | 0.061 | 5.0 |

This is the central table of the project.

### 4.3 Compare models under shift
Check whether simpler (Ridge) or more complex (XGBoost) models hold up better as shift increases — e.g. XGBoost might win in-distribution but lose stability under large shifts. That contrast is itself a finding worth writing up.

### 4.4 (Optional) Importance weighting
Upweight training rows that resemble large-shift cases:
```
sample_weight = 1 + shift_magnitude
```
Compare weighted vs. unweighted MAE, especially in the large-shift bucket. Expect a tradeoff: better large-shift performance, slightly worse small-shift performance.

### 4.5 Report
**File:** `reports/ood_analysis.md` — covering: shift definition, how `shift_magnitude` is computed, performance by bucket, error analysis, whether weighting helped, and limitations.

### 4.6 Claims to make vs. avoid
✅ "Prediction error increases for large context shifts."
✅ "Team pace, offensive rating, and assist-rate deltas are meaningful predictors of post-trade role change."
❌ "This proves causality." / "This reliably predicts all future trades."

---

## Phase 5 — FastAPI Backend
**Goal:** serve predictions from the trained model via an API.

### 5.1 API input (Option A — demo-friendly, recommended for v1)
```json
{
  "player_name": "Zach LaVine",
  "season": "2024-25",
  "team_from": "CHI",
  "team_to": "SAC"
}
```
The API looks up the player's pre-trade stats internally from the processed dataset — caller doesn't need to supply them.

### 5.2 Endpoints
```
GET  /            
GET  /health        → { "status": "ok", "model_loaded": true }
POST /predict        → see example below
```

**Example `/predict` response:**
```json
{
  "player_name": "James Harden",
  "predictions": {
    "pts":        { "pre": 22.5, "predicted_delta": 1.8,  "predicted_post": 24.3 },
    "ts_pct":      { "pre": 0.576, "predicted_delta": 0.012, "predicted_post": 0.588 },
    "usage_rate":  { "pre": 27.4, "predicted_delta": -1.9, "predicted_post": 25.5 }
  },
  "context_shift": {
    "shift_magnitude": 1.42,
    "shift_bucket": "large_shift",
    "delta_team_pace": 2.1,
    "delta_team_off_rating": 4.5
  }
}
```

### 5.3 Confidence intervals (v1: simple, label as approximate)
```
interval = predicted_value ± validation_MAE
```
Upgrade later to bootstrap, quantile regression, or conformal prediction.

### 5.4 Critical: avoid train/serve skew
The same feature-construction code must run at training time and at request time. Put a shared function in `build_features.py`:
```
build_modeling_dataset()           # used for training
build_single_prediction_row(...)   # used by the API, same logic
```

### 5.5 File layout
```
src/api/
  main.py
  schemas.py        # Pydantic request/response models
  model_service.py  # loads joblib models once at startup
  feature_service.py
```
Run with: `uvicorn src.api.main:app --reload`

### 5.6 Validation & tests
- Validate `season` format, team abbreviations, player existence in dataset
- Return clear errors, e.g. `{"detail": "Player not found in processed dataset for selected season."}`
- `tests/test_api.py`: health check returns 200; known-player predict returns 200; unknown-player predict returns 404/422; response contains pts/ts_pct/usage_rate

---

## Phase 6 — Frontend (Optional)
**Goal:** make the project demoable. Keep it secondary to the backend.

### 6.1 Scope
Player input → season/old-team/new-team dropdowns → Predict button → result cards + bar chart (pre vs. predicted post) + context-shift summary.

### 6.2 Stack choice
Static HTML/JS is enough if the backend is the focus. React + Vite if you want something more polished — Recharts is the easiest charting library to pair with it.

```
frontend/
  src/
    App.jsx
    api.js
    components/
      PredictionForm.jsx
      PredictionChart.jsx
      ContextShiftPanel.jsx
```

### 6.3 What to surface
- Bar chart: PTS / TS% / USG% — pre vs. predicted post, plus deltas (e.g. "+1.8 PPG")
- Context shift panel: shift bucket badge ("Large Context Shift"), pace change, off-rating change, assist% change — this is what makes the OOD framing visible to a viewer, not just baked into the model
- Edge cases: loading state, API error, player-not-found, empty prediction

### 6.4 Demo presets (for smoother live demos)
```
James Harden:  BKN → PHI
Kyrie Irving:  BKN → DAL
Pascal Siakam: TOR → IND
OG Anunoby:    TOR → NYK
```

---

## Repository Structure (target end state)
```
nba-trade-impact/
  data/
    raw/            trades.csv, player_game_logs.csv, player_trade_splits.csv, team_stats.csv
    processed/      modeling_dataset.csv, feature_columns.json, target_columns.json
  models/           *.joblib, model_metadata.json
  reports/
    metrics/        baseline_metrics.json, model_metrics.json, ood_metrics.json
    figures/        *.png
    ood_analysis.md
  src/
    data/           collect_trades.py, collect_game_logs.py, collect_team_stats.py
    features/       build_features.py
    models/         train_baseline.py, train_model.py, evaluate.py, ood_analysis.py
    api/            main.py, schemas.py, model_service.py, feature_service.py
  frontend/
    src/            App.jsx, api.js, components/
  tests/            test_features.py, test_models.py, test_api.py
  README.md
  requirements.txt
```

---

## Suggested Order of Attack

| Step | Task |
|---|---|
| 1 | Finish Phase 1 — fix the trade registry first. Bad trade data poisons everything downstream. |
| 2 | Build `modeling_dataset.csv`. Don't train anything until this is clean and manually inspectable. |
| 3 | Train the no-change baseline — sets your minimum bar. |
| 4 | Train Ridge — simple, interpretable. |
| 5 | Evaluate chronologically — test on the most recent trades. |
| 6 | Add OOD shift buckets — evaluate small/medium/large shift performance. |
| 7 | Try XGBoost — compare against Ridge and baseline. |
| 8 | Write the OOD report — this is what elevates it from "prediction notebook" to "distribution shift analysis." |
| 9 | Build the FastAPI backend — serve from the best saved model. |
| 10 | Add the frontend last — it should visualize a *working* model, not compensate for an unfinished backend. |

| Week | Focus |
|---|---|
| 1 | Phase 1 — data pipeline |
| 2 | Phase 2 — feature engineering |
| 3 | Phase 3 — baseline model + evaluation |
| 4 | Phase 4 — OOD analysis |
| 5 | Phase 5 — FastAPI |
| 6 | Phase 6 — polish + README |