import os
import sys
import argparse
import csv
import json
import random
import numpy as np
import cv2

# Set path to allow imports from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils.seed import set_seed
from src.utils.config import CONFIG
from src.simulation.perspective_renderer import render_warehouse_view
from src.vision.aruco_detector import detect_aruco_markers

def main():
    parser = argparse.ArgumentParser(description="Warehouse Drone Dataset Generator")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="final_project/data", help="Output directory for dataset")
    parser.add_argument("--seed", type=int, default=CONFIG["dataset"]["seed"], help="Random seed for generation")
    parser.add_argument("--quick", action="store_true", help="Generate small quick-test dataset (500 samples)")
    args = parser.parse_args()

    # Determine number of samples
    if args.quick:
        num_samples = CONFIG["dataset"]["quick_samples"]
        print(f"Quick Mode: generating {num_samples} samples.")
    elif args.num_samples is not None:
        num_samples = args.num_samples
    else:
        num_samples = CONFIG["dataset"]["num_samples"]

    # Set seed
    set_seed(args.seed)

    # Setup directories
    raw_dir = os.path.join(args.output_dir, "raw")
    splits_dir = os.path.join(args.output_dir, "splits")
    metrics_dir = "final_project/results/metrics"
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    labels_csv_path = os.path.join(args.output_dir, "labels.csv")

    print(f"Generating synthetic dataset of {num_samples} samples at {raw_dir}...")

    # Data lists
    rows = []
    
    # Track statistics
    total_boxes = 0
    average_occupancy_sum = 0.0
    marker_visible_count = 0
    marker_visibility_counts = {10: 0, 11: 0, 12: 0, 13: 0}

    # Header columns
    header = [
        "image_path", "x", "y", "z", "yaw", 
        "visible_marker_ids", "marker_center_x", "marker_center_y", "marker_area",
        "box_count", "shelf_occupancy",
        "m10_visible", "m10_cx", "m10_cy", "m10_area",
        "m11_visible", "m11_cx", "m11_cy", "m11_area",
        "m12_visible", "m12_cx", "m12_cy", "m12_area",
        "m13_visible", "m13_cx", "m13_cy", "m13_area"
    ]

    for i in range(num_samples):
        # Sample random camera pose
        x_cam = random.uniform(-1.5, 1.5)
        y_cam = random.uniform(-0.6, 0.6)
        z_cam = random.uniform(1.2, 2.6)
        yaw_cam = random.uniform(-20.0, 20.0)

        # Randomize box placement
        boxes_active = {}
        box_count = 0
        for l_idx in range(3):
            for c_idx in range(3):
                active = random.random() < 0.6
                boxes_active[(l_idx, c_idx)] = active
                if active:
                    box_count += 1
        shelf_occupancy = float(box_count) / 9.0
        
        # Accumulate box stats
        total_boxes += box_count
        average_occupancy_sum += shelf_occupancy

        # Render image
        img = render_warehouse_view(x_cam, y_cam, z_cam, yaw_cam, boxes_active)

        # Detect markers
        visible = detect_aruco_markers(img)

        # Extract features for largest visible marker
        visible_ids_str = ""
        p_cx, p_cy, p_area = 0.0, 0.0, 0.0
        if visible:
            marker_visible_count += 1
            visible.sort(key=lambda x: x["area"], reverse=True)
            largest = visible[0]
            p_cx = largest["cx"]
            p_cy = largest["cy"]
            p_area = largest["area"]
            visible_ids_str = ",".join(str(m["id"]) for m in visible)

        # Initialize fixed-length marker features
        marker_features = {
            10: {"visible": 0.0, "cx": 0.0, "cy": 0.0, "area": 0.0},
            11: {"visible": 0.0, "cx": 0.0, "cy": 0.0, "area": 0.0},
            12: {"visible": 0.0, "cx": 0.0, "cy": 0.0, "area": 0.0},
            13: {"visible": 0.0, "cx": 0.0, "cy": 0.0, "area": 0.0}
        }

        # Populate detected marker information
        for m in visible:
            m_id = m["id"]
            if m_id in marker_features:
                marker_features[m_id] = {
                    "visible": 1.0,
                    "cx": m["cx"],
                    "cy": m["cy"],
                    "area": m["area"]
                }
                marker_visibility_counts[m_id] += 1

        # Save image
        img_filename = f"img_{i:05d}.png"
        img_path_rel = os.path.join("raw", img_filename)
        img_path_full = os.path.join(raw_dir, img_filename)
        cv2.imwrite(img_path_full, img)

        # Build row
        row = [
            img_path_rel, x_cam, y_cam, z_cam, yaw_cam,
            visible_ids_str, p_cx, p_cy, p_area,
            box_count, shelf_occupancy,
            marker_features[10]["visible"], marker_features[10]["cx"], marker_features[10]["cy"], marker_features[10]["area"],
            marker_features[11]["visible"], marker_features[11]["cx"], marker_features[11]["cy"], marker_features[11]["area"],
            marker_features[12]["visible"], marker_features[12]["cx"], marker_features[12]["cy"], marker_features[12]["area"],
            marker_features[13]["visible"], marker_features[13]["cx"], marker_features[13]["cy"], marker_features[13]["area"]
        ]
        rows.append(row)

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")

    # Write central labels.csv
    with open(labels_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Perform splits
    indices = list(range(num_samples))
    # Shuffle indices deterministically
    random.shuffle(indices)

    train_pct = CONFIG["dataset"]["split_train"]
    val_pct = CONFIG["dataset"]["split_val"]

    train_idx_end = int(num_samples * train_pct)
    val_idx_end = train_idx_end + int(num_samples * val_pct)

    train_indices = indices[:train_idx_end]
    val_indices = indices[train_idx_end:val_idx_end]
    test_indices = indices[val_idx_end:]

    # Save splits
    splits = {
        "train": (train_indices, os.path.join(splits_dir, "train.csv")),
        "val": (val_indices, os.path.join(splits_dir, "val.csv")),
        "test": (test_indices, os.path.join(splits_dir, "test.csv"))
    }

    for split_name, (split_idxs, split_path) in splits.items():
        with open(split_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for idx in split_idxs:
                writer.writerow(rows[idx])

    # Save summary statistics
    summary = {
        "total_samples": num_samples,
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "test_samples": len(test_indices),
        "total_boxes": total_boxes,
        "average_occupancy": average_occupancy_sum / num_samples,
        "marker_visibility_rate": marker_visible_count / num_samples,
        "marker_counts": marker_visibility_counts
    }

    summary_path = os.path.join(metrics_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("Dataset generation completed successfully!")
    print(f"  Labels: {labels_csv_path}")
    print(f"  Splits saved to {splits_dir}")
    print(f"  Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
