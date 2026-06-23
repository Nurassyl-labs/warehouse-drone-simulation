import os
import sys
import json
import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set path to allow imports from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils.seed import set_seed
from src.utils.config import CONFIG
from src.vision.aruco_detector import detect_aruco_markers
from src.models.hybrid_estimator import HybridPoseEstimator

def compute_metrics(y_true, y_pred):
    """Computes MAE, RMSE, R2 for x, y, z, yaw and returns dict along with averages"""
    dims = ["x", "y", "z", "yaw"]
    results = {}
    mae_list, rmse_list, r2_list = [], [], []
    
    for i, dim in enumerate(dims):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        
        results[f"MAE_{dim}"] = float(mae)
        results[f"RMSE_{dim}"] = float(rmse)
        results[f"R2_{dim}"] = float(r2)
        
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        
    results["avg_MAE"] = float(np.mean(mae_list))
    results["avg_RMSE"] = float(np.mean(rmse_list))
    results["avg_R2"] = float(np.mean(r2_list))
    return results

import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pose Estimators")
    parser.add_argument("--dataset_source", type=str, default="opencv", choices=["opencv", "unity"], help="Dataset source (opencv or unity)")
    args = parser.parse_args()

    # Set seed
    set_seed(CONFIG["dataset"]["seed"])

    if args.dataset_source == "unity":
        data_dir = "final_project/unity_dataset"
    else:
        data_dir = "final_project/data"
        
    metrics_dir = "final_project/results/metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    # Load test set
    test_path = os.path.join(data_dir, "splits", "test.csv")
    if not os.path.exists(test_path):
        print("Error: test.csv split not found. Please run generate_dataset.py first!")
        return

    test_df = pd.read_csv(test_path)
    print(f"Loaded {len(test_df)} samples for final model evaluation.")

    # Auto-detect device
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Initialize Hybrid Estimator (which loads both baseline and CNN internally)
    estimator = HybridPoseEstimator(device=device)

    # Storage lists
    y_true_all = test_df[["x", "y", "z", "yaw"]].values
    
    cnn_preds = []
    baseline_preds = []
    baseline_true = [] # Baseline can only evaluate on samples with markers
    hybrid_preds = []

    # Iterate over test set
    for idx, row in test_df.iterrows():
        img_rel_path = row["image_path"]
        img_full_path = os.path.join(data_dir, img_rel_path)
        img = cv2.imread(img_full_path)
        
        true_pose = row[["x", "y", "z", "yaw"]].values.astype(np.float32)

        # Detect markers
        markers = detect_aruco_markers(img)

        # 1. CNN only prediction
        if estimator.cnn_model is not None:
            pred_cnn, _ = estimator.predict(img, []) # Empty markers forces CNN
            cnn_preds.append(pred_cnn)
        else:
            cnn_preds.append(np.zeros(4))

        # 2. Baseline prediction (if markers are visible)
        vis_sum = row["m10_visible"] + row["m11_visible"] + row["m12_visible"] + row["m13_visible"]
        if vis_sum > 0 and estimator.baseline_model is not None:
            pred_base, _ = estimator.predict(img, markers)
            baseline_preds.append(pred_base)
            baseline_true.append(true_pose)

        # 3. Hybrid prediction
        pred_hybrid, mode = estimator.predict(img, markers)
        hybrid_preds.append(pred_hybrid)

    # Convert to numpy arrays
    cnn_preds = np.array(cnn_preds)
    hybrid_preds = np.array(hybrid_preds)

    # Compute metrics
    print("\n==============================================")
    print("Evaluating CNN-Only Model (all test samples):")
    print("==============================================")
    cnn_metrics = compute_metrics(y_true_all, cnn_preds)
    print(f"  Avg MAE: {cnn_metrics['avg_MAE']:.4f} | Avg R2: {cnn_metrics['avg_R2']:.4f}")

    print("\n==============================================")
    print("Evaluating Baseline-Only Model (samples with markers):")
    print("==============================================")
    if len(baseline_preds) > 0:
        baseline_preds = np.array(baseline_preds)
        baseline_true = np.array(baseline_true)
        baseline_metrics = compute_metrics(baseline_true, baseline_preds)
        print(f"  Avg MAE: {baseline_metrics['avg_MAE']:.4f} | Avg R2: {baseline_metrics['avg_R2']:.4f}")
    else:
        print("  No test samples contained visible markers or baseline not loaded.")
        baseline_metrics = {}

    print("\n==============================================")
    print("Evaluating Hybrid Pose Estimator (all test samples):")
    print("==============================================")
    hybrid_metrics = compute_metrics(y_true_all, hybrid_preds)
    print(f"  Avg MAE: {hybrid_metrics['avg_MAE']:.4f} | Avg R2: {hybrid_metrics['avg_R2']:.4f}")

    # Output full summary to hybrid_metrics.json
    output_metrics = {
        "cnn_only_metrics": cnn_metrics,
        "baseline_only_metrics": baseline_metrics,
        "hybrid_estimator_metrics": hybrid_metrics
    }

    hybrid_metrics_path = os.path.join(metrics_dir, "hybrid_metrics.json")
    with open(hybrid_metrics_path, "w") as f:
        json.dump(output_metrics, f, indent=4)
        
    print(f"\nSaved combined comparison metrics table to: {hybrid_metrics_path}")

if __name__ == "__main__":
    main()
