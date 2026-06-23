import os
import sys
import json
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Set path to allow imports from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils.seed import set_seed
from src.utils.config import CONFIG
from src.vision.inventory_counter import detect_and_count_boxes

import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate Inventory Counting")
    parser.add_argument("--dataset_source", type=str, default="opencv", choices=["opencv", "unity"], help="Dataset source (opencv or unity)")
    args = parser.parse_args()

    set_seed(CONFIG["dataset"]["seed"])

    if args.dataset_source == "unity":
        data_dir = "final_project/unity_dataset"
    else:
        data_dir = "final_project/data"
        
    metrics_dir = "final_project/results/metrics"
    figures_dir = "final_project/results/figures"

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Load test split
    test_path = os.path.join(data_dir, "splits", "test.csv")
    if not os.path.exists(test_path):
        print("Error: test.csv split not found. Please run generate_dataset.py first!")
        return

    test_df = pd.read_csv(test_path)
    print(f"Loaded {len(test_df)} samples for inventory counting evaluation.")

    # Storage arrays
    y_true_counts = test_df["box_count"].values
    y_pred_counts = []
    
    success_images = []
    failure_images = []

    # Run inventory counter on test images
    for idx, row in test_df.iterrows():
        img_rel_path = row["image_path"]
        img_full_path = os.path.join(data_dir, img_rel_path)
        img = cv2.imread(img_full_path)
        
        true_cnt = int(row["box_count"])

        # Detect boxes
        pred_cnt, pred_occ, annotated = detect_and_count_boxes(img)
        y_pred_counts.append(pred_cnt)

        # Store visual examples (up to 3 for success/failure)
        if pred_cnt == true_cnt:
            if len(success_images) < 3:
                # Add text overlay of ground truth vs predicted
                res_img = annotated.copy()
                cv2.putText(res_img, f"GT: {true_cnt} | Pred: {pred_cnt} (MATCH)", 
                            (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                success_images.append(res_img)
        else:
            if len(failure_images) < 3:
                res_img = annotated.copy()
                cv2.putText(res_img, f"GT: {true_cnt} | Pred: {pred_cnt} (MISMATCH)", 
                            (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                failure_images.append(res_img)

    y_pred_counts = np.array(y_pred_counts)

    # Calculate metrics
    exact_matches = np.sum(y_true_counts == y_pred_counts)
    accuracy = exact_matches / len(y_true_counts)
    mae = np.mean(np.abs(y_true_counts - y_pred_counts))
    rmse = np.sqrt(np.mean((y_true_counts - y_pred_counts) ** 2))

    summary = {
        "total_images_evaluated": len(y_true_counts),
        "exact_count_matches": int(exact_matches),
        "accuracy": float(accuracy),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "average_ground_truth_count": float(np.mean(y_true_counts)),
        "average_predicted_count": float(np.mean(y_pred_counts))
    }

    print("\n==============================================")
    print("OpenCV HSV Inventory Counting Metrics:")
    print("==============================================")
    print(f"  Exact Match Accuracy: {accuracy*100:.2f}%")
    print(f"  MAE Count Error:      {mae:.4f} boxes")
    print(f"  RMSE Count Error:     {rmse:.4f} boxes")
    print(f"  Average GT Count:     {summary['average_ground_truth_count']:.2f}")
    print(f"  Average Pred Count:   {summary['average_predicted_count']:.2f}")

    # Save metrics JSON
    metrics_path = os.path.join(metrics_dir, "inventory_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved inventory metrics to: {metrics_path}")

    # Save visual examples grids
    def save_example_grid(images, path, title):
        if not images:
            return
        fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
        if len(images) == 1:
            axes = [axes]
        for i, img in enumerate(images):
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].axis("off")
            axes[i].set_title(f"Example {i+1}")
        plt.suptitle(title, fontsize=14, color="darkblue", weight="bold")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    success_path = os.path.join(figures_dir, "inventory_success_examples.png")
    save_example_grid(success_images, success_path, "Inventory Detection Success Examples")
    print(f"Saved success examples plot to: {success_path}")

    failure_path = os.path.join(figures_dir, "inventory_failure_examples.png")
    save_example_grid(failure_images, failure_path, "Inventory Detection Failure Examples")
    print(f"Saved failure examples plot to: {failure_path}")

if __name__ == "__main__":
    main()
