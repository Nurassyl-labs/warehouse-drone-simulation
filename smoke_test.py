import os
import sys

print("==================================================")
# Smoke Test for Warehouse Drone Simulation System
print("Running Smoke Test...")
print("==================================================")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. Test basic package imports
try:
    import numpy as np
    import cv2
    import torch
    import pygame
    import joblib
    from OpenGL.GL import *
    from OpenGL.GLU import *
    print("[PASS] Core scientific and graphic packages imported successfully.")
except ImportError as e:
    print(f"[FAIL] Core import failed: {e}")
    sys.exit(1)

# 2. Test modular src/ imports
try:
    from src.simulation.path_planner import a_star_search
    from src.simulation.perspective_renderer import render_warehouse_view
    from src.vision.aruco_detector import detect_aruco_markers
    from src.vision.inventory_counter import detect_and_count_boxes
    from src.models.hybrid_estimator import HybridPoseEstimator
    from src.utils.config import CONFIG
    from src.utils.seed import set_seed
    print("[PASS] Modular source scripts imported successfully.")
except ImportError as e:
    print(f"[FAIL] Modular script import failed: {e}")
    sys.exit(1)

# 3. Test path planner on dummy grid
grid = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
]
path = a_star_search(grid, (0, 0), (2, 2))
if len(path) == 5: # (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) or (0,0)->(1,0)->(2,0)->(2,1)->(2,2)
    print("[PASS] A* path planning search runs correctly.")
else:
    print(f"[FAIL] A* path planner returned unexpected path length: {len(path)}")
    sys.exit(1)

# 4. Test perspective rendering and box counting on dummy data
try:
    boxes_active = {(0, 0): True, (1, 1): True}
    img = render_warehouse_view(0.0, 0.0, 1.8, 0.0, boxes_active)
    expected_h = CONFIG["dataset"]["image_size"][1]
    expected_w = CONFIG["dataset"]["image_size"][0]
    if img is not None and img.shape == (expected_h, expected_w, 3):
        print("[PASS] Perspective renderer rendered dummy image successfully.")
    else:
        print(f"[FAIL] Rendered image is invalid or has wrong dimensions: {img.shape if img is not None else None}")
        sys.exit(1)

    box_count, occupancy, annotated = detect_and_count_boxes(img)
    print(f"[PASS] Inventory box counter processed dummy image. Found {box_count} boxes.")
except Exception as e:
    print(f"[FAIL] Rendering or counting failed: {e}")
    sys.exit(1)

# 5. Test Hybrid Estimator setup
try:
    estimator = HybridPoseEstimator(device="cpu")
    print("[PASS] HybridPoseEstimator class initialized successfully.")
except Exception as e:
    print(f"[FAIL] Hybrid Estimator initialization failed: {e}")
    sys.exit(1)

print("\n[SUCCESS] SMOKE TEST PASSED! All modular pipelines are fully functional.")
sys.exit(0)
