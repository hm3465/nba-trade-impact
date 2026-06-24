"""
build_features.py
-----------------
Phase 2 — Feature Engineering.

Reads the Phase 1 raw data files:
    data/raw/trades.csv
    data/raw/player_trade_splits.csv
    data/raw/team_stats.csv

Produces a single model-ready table:
    data/processed/modeling_dataset.csv
    data/processed/feature_columns.json
    data/processed/target_columns.json

Each row = one player + one trade event, with:
    - Player pre-trade stats (inputs)
    - Per-minute rate features
    - Team context features (old team & new team)
    - Team context deltas
    - Shift magnitude (z-scored Euclidean norm of context deltas)
    - Shift bucket (small / medium / large by tercile)
    - Prediction targets (delta_pts, delta_ts_pct, delta_usg_rate)

Usage:
    python -m src.features.build_features
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS_PATH = RAW_DIR / "player_trade_splits.csv"
TEAM_STATS_PATH = RAW_DIR / "team_stats.csv"
TRADES_PATH = RAW_DIR / "trades.csv"

OUTPUT_DATASET = PROCESSED_DIR / "modeling_dataset.csv"
OUTPUT_FEATURES = PROCESSED_DIR / "feature_columns.json"
OUTPUT_TARGETS = PROCESSED_DIR / "target_columns.json"

# ---------------------------------------------------------------------------
# Team context columns we pull from team_stats.csv
# ---------------------------------------------------------------------------
TEAM_CONTEXT_COLS = [
    "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE",
    "TS_PCT", "EFG_PCT", "AST_PCT", "TM_TOV_PCT", "REB_PCT",
]

# Columns used in the shift magnitude calculation (the OOD signal)
SHIFT_MAGNITUDE_COLS = [
    "delta_team_off_rating",
    "delta_team_def_rating",
    "delta_team_pace",
    "delta_team_ast_pct",
    "delta_team_ts_pct",
]

# ---------------------------------------------------------------------------
# Prediction targets (v1: keep it small)
# ---------------------------------------------------------------------------
TARGET_COLS = ["delta_pts", "delta_ts_pct", "delta_usg_rate"]

# ---------------------------------------------------------------------------
# Player pre-trade stat columns (inputs only — no post-trade leakage)
# ---------------------------------------------------------------------------
PLAYER_PRE_STAT_COLS = [
    "pre_games", "pre_min", "pre_pts", "pre_reb", "pre_ast",
    "pre_stl", "pre_blk", "pre_tov",
    "pre_fg_pct", "pre_fg3_pct", "pre_ft_pct",
    "pre_ts_pct", "pre_usg_rate", "pre_plus_minus",
]

# Per-minute rate features (computed below)
RATE_FEATURES = [
    "pre_pts_per_min", "pre_ast_per_min",
    "pre_reb_per_min", "pre_tov_per_min",
]


# ====================================================================
# Step 1: Load raw data
# ====================================================================

def load_splits() -> pd.DataFrame:
    """Load the player trade splits from Phase 1."""
    if not SPLITS_PATH.exists():
        print(f"[ERROR] {SPLITS_PATH} not found. Run Phase 1 first.")
        sys.exit(1)
    df = pd.read_csv(SPLITS_PATH)
    print(f"  Loaded {len(df)} trade splits from {SPLITS_PATH}")
    return df


def load_team_stats() -> pd.DataFrame:
    """Load team-level advanced stats from Phase 1."""
    if not TEAM_STATS_PATH.exists():
        print(f"[ERROR] {TEAM_STATS_PATH} not found. Run Phase 1 first.")
        sys.exit(1)
    df = pd.read_csv(TEAM_STATS_PATH)
    print(f"  Loaded {len(df)} team-season rows from {TEAM_STATS_PATH}")
    return df


# ====================================================================
# Step 2: Join team context stats
# ====================================================================

def join_team_context(splits: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    For each trade, join the old-team and new-team context stats.

    Produces columns like:
        team_from_off_rating, team_from_def_rating, ...
        team_to_off_rating,   team_to_def_rating,   ...
    """
    # Prepare a slim team lookup: (TEAM_ABBREVIATION, season) → context cols
    team_lookup = team_stats[["TEAM_ABBREVIATION", "season"] + TEAM_CONTEXT_COLS].copy()

    # Rename for the "from" join
    from_cols = {col: f"team_from_{col.lower()}" for col in TEAM_CONTEXT_COLS}
    from_cols["TEAM_ABBREVIATION"] = "team_from"
    from_cols["season"] = "season"
    team_from_df = team_lookup.rename(columns=from_cols)

    # Rename for the "to" join
    to_cols = {col: f"team_to_{col.lower()}" for col in TEAM_CONTEXT_COLS}
    to_cols["TEAM_ABBREVIATION"] = "team_to"
    to_cols["season"] = "season"
    team_to_df = team_lookup.rename(columns=to_cols)

    # Merge
    df = splits.merge(team_from_df, on=["team_from", "season"], how="left")
    df = df.merge(team_to_df, on=["team_to", "season"], how="left")

    # Report join quality
    n_total = len(df)
    n_missing_from = df[[f"team_from_{c.lower()}" for c in TEAM_CONTEXT_COLS]].isnull().any(axis=1).sum()
    n_missing_to = df[[f"team_to_{c.lower()}" for c in TEAM_CONTEXT_COLS]].isnull().any(axis=1).sum()
    print(f"  Team context join: {n_missing_from}/{n_total} missing 'from', "
          f"{n_missing_to}/{n_total} missing 'to'")

    return df


# ====================================================================
# Step 3: Compute team-context deltas
# ====================================================================

def compute_team_context_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delta_team_* = team_to_* − team_from_* for each context stat.

    These deltas quantify how different the new team environment is
    from the old one. They are the core OOD signal.
    """
    for col in TEAM_CONTEXT_COLS:
        col_lower = col.lower()
        from_col = f"team_from_{col_lower}"
        to_col = f"team_to_{col_lower}"
        delta_col = f"delta_team_{col_lower}"
        df[delta_col] = df[to_col] - df[from_col]

    print(f"  Computed {len(TEAM_CONTEXT_COLS)} team context delta columns")
    return df


# ====================================================================
# Step 4: Compute shift magnitude
# ====================================================================

def compute_shift_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute shift_magnitude = Euclidean norm of z-scored context deltas.

    Steps:
        1. Z-score each delta_team_* column
        2. Compute sqrt(sum of squares) across the selected shift columns
    """
    z_cols = []
    for col in SHIFT_MAGNITUDE_COLS:
        z_col = f"z_{col}"
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or pd.isna(std):
            df[z_col] = 0.0
        else:
            df[z_col] = (df[col] - mean) / std
        z_cols.append(z_col)

    # Euclidean norm of z-scores
    df["shift_magnitude"] = np.sqrt((df[z_cols] ** 2).sum(axis=1))

    print(f"  shift_magnitude: min={df['shift_magnitude'].min():.2f}, "
          f"max={df['shift_magnitude'].max():.2f}, "
          f"mean={df['shift_magnitude'].mean():.2f}")

    return df


# ====================================================================
# Step 5: Bucket by tercile
# ====================================================================

def compute_shift_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each trade to small_shift / medium_shift / large_shift
    based on terciles of shift_magnitude.
    """
    terciles = df["shift_magnitude"].quantile([1 / 3, 2 / 3]).values
    t1, t2 = terciles[0], terciles[1]

    def bucket(val):
        if val <= t1:
            return "small_shift"
        elif val <= t2:
            return "medium_shift"
        else:
            return "large_shift"

    df["shift_bucket"] = df["shift_magnitude"].apply(bucket)

    counts = df["shift_bucket"].value_counts()
    print(f"  Shift buckets: {dict(counts)}")
    print(f"  Tercile thresholds: t1={t1:.3f}, t2={t2:.3f}")

    return df


# ====================================================================
# Step 6: Add per-minute rate features
# ====================================================================

def compute_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-minute rate features to separate role from efficiency.
    Guard against division by zero when pre_min == 0.
    """
    safe_min = df["pre_min"].replace(0, np.nan)

    df["pre_pts_per_min"] = (df["pre_pts"] / safe_min).fillna(0).round(4)
    df["pre_ast_per_min"] = (df["pre_ast"] / safe_min).fillna(0).round(4)
    df["pre_reb_per_min"] = (df["pre_reb"] / safe_min).fillna(0).round(4)
    df["pre_tov_per_min"] = (df["pre_tov"] / safe_min).fillna(0).round(4)

    print(f"  Added {len(RATE_FEATURES)} per-minute rate features")
    return df


# ====================================================================
# Step 7: Assemble final dataset & save
# ====================================================================

def get_feature_columns() -> list[str]:
    """
    Return the ordered list of input feature columns for modeling.
    These are all known *before* the trade resolves — no leakage.
    """
    # Player pre-trade stats
    features = list(PLAYER_PRE_STAT_COLS)

    # Per-minute rates
    features += list(RATE_FEATURES)

    # Team context (from/to absolute values)
    for col in TEAM_CONTEXT_COLS:
        features.append(f"team_from_{col.lower()}")
        features.append(f"team_to_{col.lower()}")

    # Team context deltas
    for col in TEAM_CONTEXT_COLS:
        features.append(f"delta_team_{col.lower()}")

    # Shift magnitude (scalar summary of context shift)
    features.append("shift_magnitude")

    return features


def save_outputs(df: pd.DataFrame) -> None:
    """Save the modeling dataset and column manifests."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Feature and target column lists
    feature_cols = get_feature_columns()
    target_cols = list(TARGET_COLS)

    # Metadata columns to keep alongside for debugging/analysis
    meta_cols = [
        "player_id", "player_name", "season", "trade_date",
        "team_from", "team_to",
        "pre_games", "post_games",
        "shift_bucket",
    ]

    # Also keep post-trade actuals for evaluation (not used as inputs)
    actual_cols = ["post_pts", "post_ts_pct", "post_usg_rate"]

    # Build the output column order
    keep_cols = meta_cols + feature_cols + target_cols + actual_cols
    # Only include columns that actually exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for c in keep_cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)

    output_df = df[ordered].copy()
    output_df.to_csv(OUTPUT_DATASET, index=False)
    print(f"\n  [OK] Wrote {len(output_df)} rows x {len(ordered)} cols to {OUTPUT_DATASET}")

    # Save column manifests
    with open(OUTPUT_FEATURES, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  [OK] Feature columns ({len(feature_cols)}) -> {OUTPUT_FEATURES}")

    with open(OUTPUT_TARGETS, "w") as f:
        json.dump(target_cols, f, indent=2)
    print(f"  [OK] Target columns ({len(target_cols)}) -> {OUTPUT_TARGETS}")


# ====================================================================
# Shared function for serving (Phase 5 train/serve skew prevention)
# ====================================================================

def build_single_prediction_row(
    player_pre_stats: dict,
    team_from_stats: dict,
    team_to_stats: dict,
    shift_z_params: dict | None = None,
) -> pd.DataFrame:
    """
    Build a single feature row for prediction (API serving).

    Uses the same logic as the batch pipeline to avoid train/serve skew.
    The caller provides pre-trade player stats and team context dicts.

    Parameters
    ----------
    player_pre_stats : dict
        Pre-trade player averages (keys match PLAYER_PRE_STAT_COLS).
    team_from_stats : dict
        Old team context stats (keys match TEAM_CONTEXT_COLS, lowercased).
    team_to_stats : dict
        New team context stats (keys match TEAM_CONTEXT_COLS, lowercased).
    shift_z_params : dict or None
        Dict of {col: {"mean": float, "std": float}} for z-scoring.
        If None, shift_magnitude is set to NaN.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all feature columns.
    """
    row = {}

    # Player pre-trade stats
    for col in PLAYER_PRE_STAT_COLS:
        row[col] = player_pre_stats.get(col, 0.0)

    # Per-minute rates
    safe_min = row.get("pre_min", 0.0)
    if safe_min > 0:
        row["pre_pts_per_min"] = round(row.get("pre_pts", 0) / safe_min, 4)
        row["pre_ast_per_min"] = round(row.get("pre_ast", 0) / safe_min, 4)
        row["pre_reb_per_min"] = round(row.get("pre_reb", 0) / safe_min, 4)
        row["pre_tov_per_min"] = round(row.get("pre_tov", 0) / safe_min, 4)
    else:
        for rate_col in RATE_FEATURES:
            row[rate_col] = 0.0

    # Team context (from / to)
    for col in TEAM_CONTEXT_COLS:
        col_lower = col.lower()
        row[f"team_from_{col_lower}"] = team_from_stats.get(col_lower, 0.0)
        row[f"team_to_{col_lower}"] = team_to_stats.get(col_lower, 0.0)
        row[f"delta_team_{col_lower}"] = (
            row[f"team_to_{col_lower}"] - row[f"team_from_{col_lower}"]
        )

    # Shift magnitude
    if shift_z_params is not None:
        z_sq_sum = 0.0
        for mag_col in SHIFT_MAGNITUDE_COLS:
            params = shift_z_params.get(mag_col, {"mean": 0.0, "std": 1.0})
            std = params["std"] if params["std"] != 0 else 1.0
            z = (row.get(mag_col, 0.0) - params["mean"]) / std
            z_sq_sum += z ** 2
        row["shift_magnitude"] = np.sqrt(z_sq_sum)
    else:
        row["shift_magnitude"] = np.nan

    return pd.DataFrame([row])[get_feature_columns()]


# ====================================================================
# Main pipeline
# ====================================================================

def main() -> None:
    print("=" * 70)
    print("  Phase 2 — Feature Engineering")
    print("=" * 70)

    # Step 1: Load raw data
    print("\n[Step 1] Loading raw data...")
    splits = load_splits()
    team_stats = load_team_stats()

    # Step 2: Join team context
    print("\n[Step 2] Joining team context stats...")
    df = join_team_context(splits, team_stats)

    # Step 3: Compute team context deltas
    print("\n[Step 3] Computing team context deltas...")
    df = compute_team_context_deltas(df)

    # Step 4: Compute shift magnitude
    print("\n[Step 4] Computing shift magnitude...")
    df = compute_shift_magnitude(df)

    # Step 5: Bucket by tercile
    print("\n[Step 5] Assigning shift buckets...")
    df = compute_shift_buckets(df)

    # Step 6: Add rate features
    print("\n[Step 6] Computing per-minute rate features...")
    df = compute_rate_features(df)

    # Step 7: Save outputs
    print("\n[Step 7] Saving outputs...")
    save_outputs(df)

    # Summary
    print("\n" + "=" * 70)
    print("  Phase 2 complete!")
    print(f"  Dataset: {len(df)} rows")
    print(f"  Features: {len(get_feature_columns())} columns")
    print(f"  Targets: {TARGET_COLS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
