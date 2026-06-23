import os
import sys
import argparse
import subprocess
import json

def run_script(script_name, args_list=None):
    """Utility to run a python script as a subprocess in the current python interpreter"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(current_dir, script_name)]
    if args_list:
        cmd.extend(args_list)
        
    print(f"\n>>> Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n[FAIL] Script '{script_name}' failed with exit code {result.returncode}")
        return False
    print(f"[SUCCESS] Script '{script_name}' completed successfully.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Warehouse Drone Simulation Pipeline Runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true", help="Run quick pipeline (500 samples, 2 CNN epochs)")
    group.add_argument("--full", action="store_true", help="Run full pipeline (5000 samples, full training)")
    parser.add_argument("--dataset_source", type=str, default="opencv", choices=["opencv", "unity"], help="Dataset source (opencv or unity)")
    args = parser.parse_args()

    # Track stage completions
    stages = {
        "dataset_generation": False,
        "dataset_checker": False,
        "baseline_training": False,
        "cnn_training": False,
        "model_evaluation": False,
        "inventory_evaluation": False
    }

    # 1. Dataset Generation / Preprocessing
    if args.dataset_source == "unity":
        # Unity data preprocessing
        stages["dataset_generation"] = run_script("postprocess_unity.py")
    else:
        gen_args = ["--quick"] if args.quick else ["--num_samples", "5000"]
        stages["dataset_generation"] = run_script("generate_dataset.py", gen_args)
        
    if not stages["dataset_generation"]:
        sys.exit(1)

    # 2. Dataset Checker
    stages["dataset_checker"] = run_script("check_dataset.py", ["--dataset_source", args.dataset_source])
    if not stages["dataset_checker"]:
        sys.exit(1)

    # 3. Train Baseline
    stages["baseline_training"] = run_script("train_baseline.py", ["--dataset_source", args.dataset_source])
    if not stages["baseline_training"]:
        sys.exit(1)

    # 4. Train CNN
    cnn_args = ["--quick"] if args.quick else []
    cnn_args.extend(["--dataset_source", args.dataset_source])
    stages["cnn_training"] = run_script("train_cnn.py", cnn_args)
    if not stages["cnn_training"]:
        sys.exit(1)

    # 5. Evaluate models (CNN, Baseline, Hybrid)
    stages["model_evaluation"] = run_script("evaluate.py", ["--dataset_source", args.dataset_source])
    if not stages["model_evaluation"]:
        sys.exit(1)

    # 6. Evaluate inventory counting
    stages["inventory_evaluation"] = run_script("evaluate_inventory.py", ["--dataset_source", args.dataset_source])
    if not stages["inventory_evaluation"]:
        sys.exit(1)

    # 7. Print final summary and verification checklist
    print("\n" + "="*50)
    print(" PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("Final Verification Checklist:")
    print(f"  - Dataset generation:  {'PASS' if stages['dataset_generation'] else 'FAIL'}")
    print(f"  - Dataset check:       {'PASS' if stages['dataset_checker'] else 'FAIL'}")
    print(f"  - Baseline training:   {'PASS' if stages['baseline_training'] else 'FAIL'}")
    print(f"  - CNN training:        {'PASS' if stages['cnn_training'] else 'FAIL'}")
    print(f"  - Model evaluation:    {'PASS' if stages['model_evaluation'] else 'FAIL'}")
    print(f"  - Inventory evaluation: {'PASS' if stages['inventory_evaluation'] else 'FAIL'}")
    print(f"  - run_all.py execution:{'PASS'}")

    # Read and display final metrics
    metrics_dir = "final_project/results/metrics"
    
    # Baseline validation metrics summary
    b_test_path = os.path.join(metrics_dir, "baseline_test_summary.json")
    if os.path.exists(b_test_path):
        with open(b_test_path, "r") as f:
            b_test = json.load(f)
            print(f"\nFinal Baseline Test Metrics (Best Model):")
            print(f"  x MAE: {b_test.get('test_MAE_x', 0.0):.4f}m | y MAE: {b_test.get('test_MAE_y', 0.0):.4f}m | z MAE: {b_test.get('test_MAE_z', 0.0):.4f}m | yaw MAE: {b_test.get('test_MAE_yaw', 0.0):.1f}°")

    # CNN test metrics summary
    cnn_test_path = os.path.join(metrics_dir, "cnn_metrics.json")
    if os.path.exists(cnn_test_path):
        with open(cnn_test_path, "r") as f:
            cnn_test = json.load(f)
            print(f"\nFinal CNN Test Metrics:")
            print(f"  x MAE: {cnn_test.get('test_MAE_x', 0.0):.4f}m | y MAE: {cnn_test.get('test_MAE_y', 0.0):.4f}m | z MAE: {cnn_test.get('test_MAE_z', 0.0):.4f}m | yaw MAE: {cnn_test.get('test_MAE_yaw', 0.0):.1f}°")
            print(f"  Average MAE: {cnn_test.get('test_avg_MAE', 0.0):.4f} | Average R^2: {cnn_test.get('test_avg_R2', 0.0):.4f}")

    # Hybrid test metrics summary
    hybrid_test_path = os.path.join(metrics_dir, "hybrid_metrics.json")
    if os.path.exists(hybrid_test_path):
        with open(hybrid_test_path, "r") as f:
            hyb_test = json.load(f)
            h_metrics = hyb_test.get("hybrid_estimator_metrics", {})
            print(f"\nFinal Hybrid Estimator Test Metrics:")
            print(f"  x MAE: {h_metrics.get('MAE_x', 0.0):.4f}m | y MAE: {h_metrics.get('MAE_y', 0.0):.4f}m | z MAE: {h_metrics.get('MAE_z', 0.0):.4f}m | yaw MAE: {h_metrics.get('MAE_yaw', 0.0):.1f}°")
            print(f"  Average MAE: {h_metrics.get('avg_MAE', 0.0):.4f} | Average R^2: {h_metrics.get('avg_R2', 0.0):.4f}")

    # Inventory test metrics summary
    inv_test_path = os.path.join(metrics_dir, "inventory_metrics.json")
    if os.path.exists(inv_test_path):
        with open(inv_test_path, "r") as f:
            inv_test = json.load(f)
            print(f"\nFinal Inventory Counting Test Metrics:")
            print(f"  Exact Match Accuracy: {inv_test.get('accuracy', 0.0)*100:.2f}%")
            print(f"  MAE Count Error:      {inv_test.get('MAE', 0.0):.4f} boxes")

    print("\nAll pipeline outputs and metrics are available in: final_project/results/")

if __name__ == "__main__":
    main()
