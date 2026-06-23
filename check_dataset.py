import os
import pandas as pd
import numpy as np

def check_dataset(data_dir="final_project/data"):
    print("==================================================")
    print("Running Dataset Integrity Checker...")
    print("==================================================")

    # 1. Check labels.csv existence
    labels_path = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(labels_path):
        print(f"[FAIL] labels.csv not found at {labels_path}")
        return False
    else:
        print("[PASS] labels.csv exists.")

    # Load dataframe
    df = pd.read_csv(labels_path)
    print(f"Loaded {len(df)} samples from labels.csv")

    # 2. Check splits files
    splits = ["train", "val", "test"]
    split_dfs = {}
    for split in splits:
        split_path = os.path.join(data_dir, "splits", f"{split}.csv")
        if not os.path.exists(split_path):
            print(f"[FAIL] Split file {split_path} not found.")
            return False
        split_dfs[split] = pd.read_csv(split_path)
        print(f"[PASS] Split '{split}' CSV exists with {len(split_dfs[split])} samples.")

    # Check split count sum
    total_split_len = sum(len(split_dfs[split]) for split in splits)
    if total_split_len != len(df):
        print(f"[WARNING] Split samples sum ({total_split_len}) does not match labels.csv ({len(df)})")
    else:
        print("[PASS] Split sizes sum matches total samples count.")

    # 3. Verify images exist
    missing_images = 0
    raw_dir = os.path.join(data_dir, "raw")
    for idx, row in df.iterrows():
        img_rel_path = row["image_path"]
        # In labels.csv image_path is relative to output_dir (e.g. raw/img_00000.png)
        img_full_path = os.path.join(data_dir, img_rel_path)
        if not os.path.exists(img_full_path):
            if missing_images < 5:
                print(f"[FAIL] Image file not found: {img_full_path}")
            missing_images += 1
            
    if missing_images > 0:
        print(f"[FAIL] Total missing images: {missing_images}")
        return False
    else:
        print("[PASS] All image files in labels.csv exist on disk.")

    # 4. Check pose label ranges and missing values
    pose_cols = ["x", "y", "z", "yaw"]
    for col in pose_cols:
        if df[col].isnull().any():
            print(f"[FAIL] Column '{col}' contains null/missing values.")
            return False
            
    # Print ranges
    print(f"Pose coordinates ranges:")
    for col in pose_cols:
        print(f"  - {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}")

    # 5. Check box count and occupancy validity
    if ((df["box_count"] < 0) | (df["box_count"] > 9)).any():
        print("[FAIL] box_count contains values outside [0, 9] range.")
        return False
    if ((df["shelf_occupancy"] < 0.0) | (df["shelf_occupancy"] > 1.0)).any():
        print("[FAIL] shelf_occupancy contains values outside [0.0, 1.0] range.")
        return False
    
    # Check if box_count/9.0 equals shelf_occupancy
    occupancy_check = np.abs(df["box_count"] / 9.0 - df["shelf_occupancy"])
    if (occupancy_check > 1e-5).any():
        print("[FAIL] shelf_occupancy is not equal to box_count / 9.0!")
        return False
    print("[PASS] Box counts and occupancy fractions are valid and aligned.")

    # 6. Check ArUco marker features formatting
    # visible_marker_ids could be null/NaN if empty, check format otherwise
    marker_ids_nan = df["visible_marker_ids"].isnull()
    for idx, row in df.iterrows():
        marker_str = row["visible_marker_ids"]
        marker_area = row["marker_area"]
        if pd.isnull(marker_str) or str(marker_str).strip() == "":
            if marker_area > 0.0:
                print(f"[FAIL] Row {idx}: marker_area={marker_area} but visible_marker_ids is empty.")
                return False
        else:
            ids = [int(float(x)) for x in str(marker_str).split(",")]
            for m_id in ids:
                if m_id not in [10, 11, 12, 13]:
                    print(f"[FAIL] Row {idx}: Invalid marker ID detected: {m_id}")
                    return False
            if marker_area == 0.0:
                print(f"[FAIL] Row {idx}: visible_marker_ids='{marker_str}' but marker_area is 0.0.")
                return False

    print("[PASS] ArUco visible marker formats and statistics are valid.")

    # Print summary statistics
    total_imgs = len(df)
    imgs_with_markers = df[df["marker_area"] > 0].shape[0]
    print("\nDataset Summary Statistics:")
    print(f"  Total Images: {total_imgs}")
    print(f"  Images with visible markers: {imgs_with_markers} ({imgs_with_markers/total_imgs*100:.2f}%)")
    print(f"  Average Box Count: {df['box_count'].mean():.2f}")
    print(f"  Average Shelf Occupancy: {df['shelf_occupancy'].mean()*100:.2f}%")
    
    # Marker-wise statistics
    for m_id in [10, 11, 12, 13]:
        col = f"m{m_id}_visible"
        if col in df.columns:
            vis_cnt = df[df[col] == 1.0].shape[0]
            print(f"  Marker {m_id} Visibility: {vis_cnt} / {total_imgs} ({vis_cnt/total_imgs*100:.2f}%)")

    print("\n[SUCCESS] All dataset integrity checks PASSED!")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset Integrity Checker")
    parser.add_argument("--dataset_source", type=str, default="opencv", choices=["opencv", "unity"], help="Dataset source (opencv or unity)")
    args = parser.parse_args()
    
    if args.dataset_source == "unity":
        check_dataset("final_project/unity_dataset")
    else:
        check_dataset("final_project/data")
