"""
train_model.py
--------------
Train ML models for NBA trade impact prediction.

Implements:
1. Ridge regression
2. HistGradientBoostingRegressor (sklearn tree baseline)
3. XGBoost

Usage:
    python -m src.models.train_model
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
OUTPUT_PATH = REPORTS_DIR / "model_metrics.json"
PREPROCESSING_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------
def get_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def build_preprocessing_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """
    Build preprocessing pipeline for features.
    Numeric: median imputation + standardization
    Categorical: most frequent imputation + one-hot encoding
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    
    return preprocessor


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


def train_ridge(X_train: np.ndarray, y_train: np.ndarray, alphas: list[float] = [0.1, 1, 10, 100]) -> tuple[Ridge, dict]:
    """Train Ridge regression with hyperparameter tuning."""
    best_model = None
    best_mae = float("inf")
    best_alpha = None
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        mae = mean_absolute_error(y_train, y_pred)
        
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_alpha = alpha
    
    return best_model, {"best_alpha": best_alpha, "train_mae": best_mae}


def train_histgb(X_train: np.ndarray, y_train: np.ndarray) -> tuple[HistGradientBoostingRegressor, dict]:
    """Train HistGradientBoostingRegressor."""
    model = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.1,
        max_iter=200,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    
    return model, {"train_mae": mae}


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """Train XGBoost model."""
    try:
        import xgboost as xgb
    except ImportError:
        print("[WARN] XGBoost not installed, skipping XGBoost training")
        return None, {}
    
    model = xgb.XGBRegressor(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    
    return model, {"train_mae": mae}


def main() -> None:
    print("=" * 70)
    print("  Training ML Models")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} samples from {DATASET_PATH}")
    
    # Load feature and target columns
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)
    with open(TARGET_COLS_PATH) as f:
        target_cols = json.load(f)
    
    # All features are numeric (shift_bucket not in feature columns)
    categorical_features = []
    numeric_features = feature_cols
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Split chronologically
    train_df, val_df, test_df = get_train_test_split(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
    
    # Fit preprocessing on training data
    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]
    preprocessor.fit(X_train)
    
    # Transform all splits
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(val_df[feature_cols])
    X_test_processed = preprocessor.transform(test_df[feature_cols])
    
    # Save preprocessing pipeline
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSING_PATH)
    print(f"[OK] Saved preprocessing pipeline to {PREPROCESSING_PATH}")
    
    # Train models for each target
    results = {}
    model_metadata = {
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "models": {},
    }
    
    for i, target in enumerate(target_cols):
        print(f"\n{'=' * 70}")
        print(f"  Target: {target}")
        print(f"{'=' * 70}")
        
        y_train_target = y_train[target].values
        y_val_target = val_df[target].values
        y_test_target = test_df[target].values
        
        results[target] = {}
        model_metadata["models"][target] = {}
        
        # Ridge
        print("\n--- Ridge Regression ---")
        ridge_model, ridge_info = train_ridge(X_train_processed, y_train_target)
        ridge_val_pred = ridge_model.predict(X_val_processed)
        ridge_test_pred = ridge_model.predict(X_test_processed)
        
        ridge_val_metrics = compute_metrics(y_val_target, ridge_val_pred)
        ridge_test_metrics = compute_metrics(y_test_target, ridge_test_pred)
        
        results[target]["ridge"] = {
            "validation": ridge_val_metrics,
            "test": ridge_test_metrics,
            "hyperparameters": ridge_info,
        }
        model_metadata["models"][target]["ridge"] = ridge_info
        
        # Save Ridge model
        ridge_path = MODELS_DIR / f"ridge_{target}.joblib"
        joblib.dump(ridge_model, ridge_path)
        print(f"  Saved to {ridge_path}")
        print(f"  Val MAE: {ridge_val_metrics['mae']:.4f}")
        print(f"  Test MAE: {ridge_test_metrics['mae']:.4f}")
        
        # HistGradientBoosting
        print("\n--- HistGradientBoosting ---")
        histgb_model, histgb_info = train_histgb(X_train_processed, y_train_target)
        histgb_val_pred = histgb_model.predict(X_val_processed)
        histgb_test_pred = histgb_model.predict(X_test_processed)
        
        histgb_val_metrics = compute_metrics(y_val_target, histgb_val_pred)
        histgb_test_metrics = compute_metrics(y_test_target, histgb_test_pred)
        
        results[target]["histgb"] = {
            "validation": histgb_val_metrics,
            "test": histgb_test_metrics,
            "hyperparameters": histgb_info,
        }
        model_metadata["models"][target]["histgb"] = histgb_info
        
        # Save HistGB model
        histgb_path = MODELS_DIR / f"histgb_{target}.joblib"
        joblib.dump(histgb_model, histgb_path)
        print(f"  Saved to {histgb_path}")
        print(f"  Val MAE: {histgb_val_metrics['mae']:.4f}")
        print(f"  Test MAE: {histgb_test_metrics['mae']:.4f}")
        
        # XGBoost
        print("\n--- XGBoost ---")
        xgb_model, xgb_info = train_xgboost(X_train_processed, y_train_target)
        
        if xgb_model is not None:
            xgb_val_pred = xgb_model.predict(X_val_processed)
            xgb_test_pred = xgb_model.predict(X_test_processed)
            
            xgb_val_metrics = compute_metrics(y_val_target, xgb_val_pred)
            xgb_test_metrics = compute_metrics(y_test_target, xgb_test_pred)
            
            results[target]["xgboost"] = {
                "validation": xgb_val_metrics,
                "test": xgb_test_metrics,
                "hyperparameters": xgb_info,
            }
            model_metadata["models"][target]["xgboost"] = xgb_info
            
            # Save XGBoost model
            xgb_path = MODELS_DIR / f"xgboost_{target}.joblib"
            joblib.dump(xgb_model, xgb_path)
            print(f"  Saved to {xgb_path}")
            print(f"  Val MAE: {xgb_val_metrics['mae']:.4f}")
            print(f"  Test MAE: {xgb_test_metrics['mae']:.4f}")
    
    # Save results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved model metrics to {OUTPUT_PATH}")
    
    # Save metadata
    with open(METADATA_PATH, "w") as f:
        json.dump(model_metadata, f, indent=2)
    print(f"[OK] Saved model metadata to {METADATA_PATH}")
    
    # Print summary table
    print(f"\n{'=' * 70}")
    print("  Summary: Test MAE by Target and Model")
    print(f"{'=' * 70}")
    print(f"{'Target':<20} {'Ridge':<12} {'HistGB':<12} {'XGBoost':<12}")
    print("-" * 70)
    
    for target in target_cols:
        ridge_mae = results[target]["ridge"]["test"]["mae"]
        histgb_mae = results[target]["histgb"]["test"]["mae"]
        xgb_mae = results[target].get("xgboost", {}).get("test", {}).get("mae", "N/A")
        
        print(f"{target:<20} {ridge_mae:<12.4f} {histgb_mae:<12.4f} {str(xgb_mae):<12}")


if __name__ == "__main__":
    main()
