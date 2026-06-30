"""
train_baseline.py
------------------
Train baseline models for NBA trade impact prediction.

Implements simple baselines:
1. No-change baseline: predicted_delta = 0
2. Historical average delta: mean training delta per target

Usage:
    python -m src.models.train_baseline
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "metrics"

DATASET_PATH = DATA_DIR / "modeling_dataset.csv"
FEATURE_COLS_PATH = DATA_DIR / "feature_columns.json"
TARGET_COLS_PATH = DATA_DIR / "target_columns.json"
OUTPUT_PATH = REPORTS_DIR / "baseline_metrics.json"

# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------
def get_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically (not random).
    Train: 2015-16 → 2021-22
    Validation: 2022-23
    Test: 2023-24 → 2024-25
    """
    train = df[df["season"].isin(["2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22"])]
    val = df[df["season"] == "2022-23"]
    test = df[df["season"].isin(["2023-24", "2024-25"])]
    
    return train, val, test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy: did we get the sign right?
    direction_correct = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "directional_accuracy": float(direction_correct),
    }


def no_change_baseline(y_true: np.ndarray) -> dict:
    """Baseline: predict delta = 0 for all samples."""
    y_pred = np.zeros_like(y_true)
    return compute_metrics(y_true, y_pred)


def historical_average_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> dict:
    """Baseline: predict the mean training delta."""
    mean_delta = train_df[target].mean()
    y_true = test_df[target].values
    y_pred = np.full_like(y_true, mean_delta)
    
    metrics = compute_metrics(y_true, y_pred)
    metrics["mean_delta"] = float(mean_delta)
    return metrics


def main() -> None:
    print("=" * 70)
    print("  Training Baseline Models")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} samples from {DATASET_PATH}")
    
    # Load target columns
    with open(TARGET_COLS_PATH) as f:
        target_cols = json.load(f)
    
    # Split chronologically
    train_df, val_df, test_df = get_train_test_split(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Combine train+val for baseline training (use all historical data)
    train_val_df = pd.concat([train_df, val_df])
    
    # Compute baselines
    results = {}
    
    for target in target_cols:
        print(f"\n--- Target: {target} ---")
        
        # No-change baseline
        no_change_test = no_change_baseline(test_df[target].values)
        no_change_val = no_change_baseline(val_df[target].values)
        
        # Historical average baseline
        hist_avg_test = historical_average_baseline(train_val_df, test_df, target)
        hist_avg_val = historical_average_baseline(train_df, val_df, target)
        
        results[target] = {
            "no_change": {
                "test": no_change_test,
                "validation": no_change_val,
            },
            "historical_average": {
                "test": hist_avg_test,
                "validation": hist_avg_val,
            },
        }
        
        print(f"  No-change test MAE: {no_change_test['mae']:.4f}")
        print(f"  Historical avg test MAE: {hist_avg_test['mae']:.4f}")
        print(f"  Historical avg mean delta: {hist_avg_test['mean_delta']:.4f}")
    
    # Save results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Saved baseline metrics to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
