import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set path to allow imports from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils.seed import set_seed
from src.utils.config import CONFIG

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train Baseline Pose Regressors")
    parser.add_argument("--dataset_source", type=str, default="opencv", choices=["opencv", "unity"], help="Dataset source (opencv or unity)")
    args = parser.parse_args()

    # Set seed
    set_seed(CONFIG["dataset"]["seed"])

    if args.dataset_source == "unity":
        data_dir = "final_project/unity_dataset"
    else:
        data_dir = "final_project/data"
    models_dir = "final_project/models"
    metrics_dir = "final_project/results/metrics"
    figures_dir = "final_project/results/figures"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load splits
    train_path = os.path.join(data_dir, "splits", "train.csv")
    val_path = os.path.join(data_dir, "splits", "val.csv")
    test_path = os.path.join(data_dir, "splits", "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        print("Error: Dataset splits not found. Please run generate_dataset.py first!")
        return

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # 2. Extract fixed-length multi-marker features
    feature_cols = [
        "m10_visible", "m10_cx", "m10_cy", "m10_area",
        "m11_visible", "m11_cx", "m11_cy", "m11_area",
        "m12_visible", "m12_cx", "m12_cy", "m12_area",
        "m13_visible", "m13_cx", "m13_cy", "m13_area"
    ]
    target_cols = ["x", "y", "z", "yaw"]

    # Filter to keep only samples where at least one marker is visible
    # (otherwise features are all 0, which cannot regress pose)
    def filter_visible(df):
        vis_sum = df["m10_visible"] + df["m11_visible"] + df["m12_visible"] + df["m13_visible"]
        return df[vis_sum > 0].copy()

    train_df_filtered = filter_visible(train_df)
    val_df_filtered = filter_visible(val_df)
    test_df_filtered = filter_visible(test_df)

    print(f"Filtered samples with visible markers:")
    print(f"  Train: {len(train_df_filtered)} / {len(train_df)}")
    print(f"  Val: {len(val_df_filtered)} / {len(val_df)}")
    print(f"  Test: {len(test_df_filtered)} / {len(test_df)}")

    if len(train_df_filtered) < 10:
        print("Error: Too few samples with visible markers to train baseline regressor.")
        return

    X_train = train_df_filtered[feature_cols].values
    y_train = train_df_filtered[target_cols].values

    X_val = val_df_filtered[feature_cols].values
    y_val = val_df_filtered[target_cols].values

    X_test = test_df_filtered[feature_cols].values
    y_test = test_df_filtered[target_cols].values

    # 3. Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler_path = os.path.join(models_dir, "baseline_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved baseline scaler to: {scaler_path}")

    # 4. Define models to train and evaluate
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42)),
        "AdaBoost": MultiOutputRegressor(AdaBoostRegressor(n_estimators=100, random_state=42)),
        "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42, early_stopping=True)
    }

    # Evaluate on Validation Set
    val_results = []
    best_avg_r2 = -float("inf")
    best_model_name = ""
    best_model = None

    print("\nTraining and evaluating models on Validation Set...")
    for name, model in models.items():
        print(f"  Fitting {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Predict on validation set
        y_val_pred = model.predict(X_val_scaled)
        
        # Calculate dimension-wise metrics
        r2_dims = []
        mae_dims = []
        rmse_dims = []
        
        for i, dim in enumerate(target_cols):
            mae = mean_absolute_error(y_val[:, i], y_val_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_val[:, i], y_val_pred[:, i]))
            r2 = r2_score(y_val[:, i], y_val_pred[:, i])
            r2_dims.append(r2)
            mae_dims.append(mae)
            rmse_dims.append(rmse)

        avg_r2 = np.mean(r2_dims)
        avg_mae = np.mean(mae_dims)
        avg_rmse = np.mean(rmse_dims)

        print(f"    {name} -> Avg MAE: {avg_mae:.4f} | Avg R2: {avg_r2:.4f}")
        
        val_results.append({
            "Model": name,
            "Avg_MAE": avg_mae,
            "Avg_RMSE": avg_rmse,
            "Avg_R2": avg_r2,
            "R2_x": r2_dims[0], "R2_y": r2_dims[1], "R2_z": r2_dims[2], "R2_yaw": r2_dims[3],
            "MAE_x": mae_dims[0], "MAE_y": mae_dims[1], "MAE_z": mae_dims[2], "MAE_yaw": mae_dims[3],
            "RMSE_x": rmse_dims[0], "RMSE_y": rmse_dims[1], "RMSE_z": rmse_dims[2], "RMSE_yaw": rmse_dims[3]
        })

        if avg_r2 > best_avg_r2:
            best_avg_r2 = avg_r2
            best_model_name = name
            best_model = model

    print(f"\nBest Baseline Model Selected: {best_model_name} (Val R^2: {best_avg_r2:.4f})")

    # Save validation results
    val_results_df = pd.DataFrame(val_results)
    metrics_path = os.path.join(metrics_dir, "baseline_metrics.csv")
    val_results_df.to_csv(metrics_path, index=False)
    print(f"Saved baseline validation metrics table to: {metrics_path}")

    # Save best model
    best_model_path = os.path.join(models_dir, "best_baseline_regressor.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"Saved best baseline model to: {best_model_path}")

    # 5. Evaluate Best Model on Test Set
    y_test_pred = best_model.predict(X_test_scaled)
    print(f"\nFinal Test Set Evaluation for {best_model_name}:")
    
    test_summary = {}
    for i, dim in enumerate(target_cols):
        mae = mean_absolute_error(y_test[:, i], y_test_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_test_pred[:, i]))
        r2 = r2_score(y_test[:, i], y_test_pred[:, i])
        print(f"  {dim} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R^2: {r2:.4f}")
        test_summary[f"test_MAE_{dim}"] = float(mae)
        test_summary[f"test_RMSE_{dim}"] = float(rmse)
        test_summary[f"test_R2_{dim}"] = float(r2)

    # Append test results to metrics folder as a small json summary
    with open(os.path.join(metrics_dir, "baseline_test_summary.json"), "w") as f:
        json.dump(test_summary, f, indent=4)

if __name__ == "__main__":
    main()
