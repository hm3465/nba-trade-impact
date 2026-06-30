"""
evaluate.py
-----------
Generate evaluation diagnostics and visualizations for trained models.

Creates:
- Predicted vs actual scatter plots
- Residuals by season
- Residuals by shift bucket
- Feature importance (for tree models)
- Biggest errors table

Usage:
    python -m src.models.evaluate
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET_PATH = DATA_DIR / "modeling_dataset.csv"
FEATURE_COLS_PATH = DATA_DIR / "feature_columns.json"
TARGET_COLS_PATH = DATA_DIR / "target_columns.json"
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


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, target: str, model_name: str) -> Path:
    """Create predicted vs actual scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Delta', fontsize=12)
    ax.set_ylabel('Predicted Delta', fontsize=12)
    ax.set_title(f'{model_name} - {target}: Predicted vs Actual', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add MAE to plot
    mae = mean_absolute_error(y_true, y_pred)
    ax.text(0.05, 0.95, f'MAE: {mae:.4f}', transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    output_path = FIGURES_DIR / f"{model_name}_{target}_pred_vs_actual.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_residuals_by_season(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, 
                             target: str, model_name: str) -> Path:
    """Plot residuals by season."""
    df_copy = df.copy()
    df_copy['residual'] = y_true - y_pred
    df_copy['abs_residual'] = np.abs(df_copy['residual'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    seasons = sorted(df_copy['season'].unique())
    mae_by_season = df_copy.groupby('season')['abs_residual'].mean()
    
    ax.bar(range(len(seasons)), mae_by_season.values, alpha=0.7, edgecolor='k')
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(seasons, rotation=45, ha='right')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title(f'{model_name} - {target}: MAE by Season', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    output_path = FIGURES_DIR / f"{model_name}_{target}_residuals_by_season.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_residuals_by_shift_bucket(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                                    target: str, model_name: str) -> Path:
    """Plot residuals by shift bucket."""
    df_copy = df.copy()
    df_copy['residual'] = y_true - y_pred
    df_copy['abs_residual'] = np.abs(df_copy['residual'])
    
    # Skip if shift_bucket column doesn't exist
    if 'shift_bucket' not in df_copy.columns:
        print(f"[SKIP] shift_bucket column not found for {model_name} - {target}")
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    buckets = ['small_shift', 'medium_shift', 'large_shift']
    mae_by_bucket = []
    for bucket in buckets:
        bucket_data = df_copy[df_copy['shift_bucket'] == bucket]
        if len(bucket_data) > 0:
            mae_by_bucket.append(bucket_data['abs_residual'].mean())
        else:
            mae_by_bucket.append(0)
    
    ax.bar(buckets, mae_by_bucket, alpha=0.7, edgecolor='k', color=['green', 'orange', 'red'])
    ax.set_xlabel('Shift Bucket', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title(f'{model_name} - {target}: MAE by Shift Bucket', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    output_path = FIGURES_DIR / f"{model_name}_{target}_residuals_by_shift.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_feature_importance(model, feature_names: list, target: str, model_name: str) -> Path:
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models, use absolute coefficient values
        importances = np.abs(model.coef_)
    else:
        print(f"[WARN] {model_name} does not have feature_importances_ attribute")
        return None
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    top_n = 15
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(range(top_n), importances[indices[:top_n]], align='center')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[:top_n]])
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'{model_name} - {target}: Top {top_n} Feature Importance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    output_path = FIGURES_DIR / f"{model_name}_{target}_feature_importance.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_biggest_errors_table(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                                   target: str, model_name: str, n: int = 10) -> pd.DataFrame:
    """Generate table of biggest prediction errors."""
    df_copy = df.copy()
    df_copy['actual_delta'] = y_true
    df_copy['predicted_delta'] = y_pred
    df_copy['error'] = np.abs(y_true - y_pred)
    
    # Sort by error and take top n
    worst = df_copy.nlargest(n, 'error')
    
    # Select relevant columns
    cols = ['player_name', 'season', 'team_from', 'team_to', 'actual_delta', 
            'predicted_delta', 'error', 'shift_magnitude']
    result = worst[cols].copy()
    result['model'] = model_name
    result['target'] = target
    
    return result


def main() -> None:
    print("=" * 70)
    print("  Generating Evaluation Diagnostics")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} samples from {DATASET_PATH}")
    
    # Load feature and target columns
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)
    with open(TARGET_COLS_PATH) as f:
        target_cols = json.load(f)
    
    # Load preprocessing pipeline
    preprocessor = joblib.load(PREPROCESSING_PATH)
    
    # Load metadata
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    
    # Split chronologically
    train_df, val_df, test_df = get_train_test_split(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Use test set for evaluation
    X_test = preprocessor.transform(test_df[feature_cols])
    
    # Create figures directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all biggest errors
    all_biggest_errors = []
    
    # Evaluate each model and target
    for target in target_cols:
        print(f"\n{'=' * 70}")
        print(f"  Target: {target}")
        print(f"{'=' * 70}")
        
        y_test = test_df[target].values
        
        for model_name in ["ridge", "histgb", "xgboost"]:
            model_path = MODELS_DIR / f"{model_name}_{target}.joblib"
            
            if not model_path.exists():
                print(f"[SKIP] {model_name} model not found for {target}")
                continue
            
            print(f"\n--- {model_name.upper()} ---")
            
            # Load model
            model = joblib.load(model_path)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Generate plots
            plot_predicted_vs_actual(y_test, y_pred, target, model_name)
            print(f"  Generated predicted vs actual plot")
            
            plot_residuals_by_season(test_df, y_test, y_pred, target, model_name)
            print(f"  Generated residuals by season plot")
            
            shift_result = plot_residuals_by_shift_bucket(test_df, y_test, y_pred, target, model_name)
            if shift_result:
                print(f"  Generated residuals by shift bucket plot")
            
            # Feature importance (onlyfor models that support it)
            if model_name == "xgboost" and hasattr(model, 'feature_importances_'):
                # Get feature names after preprocessing
                feature_names = []
                numeric_features = metadata["numeric_features"]
                categorical_features = metadata["categorical_features"]
                
                # Numeric features keep their names
                feature_names.extend(numeric_features)
                
                # Categorical features are one-hot encoded (only if they exist)
                if categorical_features and hasattr(preprocessor, 'named_transformers_'):
                    cat_transformer = preprocessor.named_transformers_.get('cat')
                    if cat_transformer and hasattr(cat_transformer, 'named_steps'):
                        onehot = cat_transformer.named_steps['onehot']
                        if hasattr(onehot, 'get_feature_names_out') and hasattr(onehot, 'categories_'):
                            cat_names = onehot.get_feature_names_out(categorical_features)
                            feature_names.extend(cat_names)
                
                if len(feature_names) == len(model.feature_importances_):
                    plot_feature_importance(model, feature_names, target, model_name)
                    print(f"  Generated feature importance plot")
                else:
                    print(f"  [SKIP] Feature name mismatch: {len(feature_names)} vs {len(model.feature_importances_)}")
            
            # Biggest errors table
            worst_errors = generate_biggest_errors_table(test_df, y_test, y_pred, target, model_name)
            all_biggest_errors.append(worst_errors)
            print(f"  Generated biggest errors table")
    
    # Combine and save biggest errors
    if all_biggest_errors:
        combined_errors = pd.concat(all_biggest_errors, ignore_index=True)
        errors_path = FIGURES_DIR / "biggest_errors.csv"
        combined_errors.to_csv(errors_path, index=False)
        print(f"\n[OK] Saved biggest errors to {errors_path}")
    
    print(f"\n[OK] All evaluation diagnostics saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
