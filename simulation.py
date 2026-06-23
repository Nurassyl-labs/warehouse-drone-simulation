import os
import sys
import time
import random
import math
import argparse
import pygame
import torch
from torchvision import transforms
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import cv2
import joblib

# Add final_project to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.simulation.path_planner import a_star_search
from src.simulation.perspective_renderer import render_warehouse_view
from src.vision.aruco_detector import detect_aruco_markers
from src.vision.inventory_counter import detect_and_count_boxes
from src.models.hybrid_estimator import HybridPoseEstimator

# Suppress OMP warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Grid definition (12x12)
GRID = [
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

GRID_ROWS = len(GRID)
GRID_COLS = len(GRID[0])

CLASSES = ["box"]
CNN_MODEL_PATH = "final_project/models/best_pose_model.pth"
BASELINE_MODEL_PATH = "final_project/models/best_baseline_regressor.joblib"

# Pygame Window Constants
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
MAP_CELL_SIZE = 45
MAP_OFFSET_X = 40
MAP_OFFSET_Y = 100

# 3D Constants
CELL_SIZE_3D = 20.0

class WarehouseDroneApp:
    def __init__(self, demo_mode=False, record_mode=False):
        self.demo_mode = demo_mode
        self.record_mode = record_mode
        
        # Setup recording directories
        self.output_dir = "final_project/results/demo_outputs"
        self.frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.frame_counter = 0
        
        # Track screenshot status
        self.screenshots_taken = {
            "overview": False,
            "scan": False,
            "camera": False,
            "pose": False,
            "summary": False
        }

        pygame.init()
        pygame.display.set_caption("Warehouse Drone 3D Simulation & AI Pose Dashboard")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.DOUBLEBUF | pygame.OPENGL)
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Fonts
        self.font_title = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_header = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_body = pygame.font.SysFont("Arial", 13)
        self.font_small = pygame.font.SysFont("Arial", 11)
        self.font_bold = pygame.font.SysFont("Arial", 12, bold=True)
        
        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.estimator = HybridPoseEstimator(device=self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.init_opengl()
        self.hud_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        
        # Cinematic Camera State
        self.camera_yaw = -45.0
        self.camera_pitch = 45.0
        self.camera_dist = 300.0
        self.camera_look_at = np.array([0.0, 5.0, 0.0])
        
        self.is_dragging = False
        self.last_mouse_pos = (0, 0)
        
        # Simulation State
        self.drone_grid_x = 0
        self.drone_grid_y = 0
        self.drone_pixel_x = float(MAP_OFFSET_X + self.drone_grid_x * MAP_CELL_SIZE + MAP_CELL_SIZE // 2)
        self.drone_pixel_y = float(MAP_OFFSET_Y + self.drone_grid_y * MAP_CELL_SIZE + MAP_CELL_SIZE // 2)
        self.battery = 100.0
        self.state = "IDLE"  # IDLE, NAVIGATING, SCANNING, CHARGING
        self.drone_speed = 3.0
        self.mode = "AUTO"
        self.simulation_speed = 1
        self.drone_angle = 0.0
        self.rotor_angle = 0.0
        self.temp_obstacles = set()
        
        # Path planning
        self.path = []
        self.path_index = 0
        
        # Inventory DB
        random.seed(42)
        self.shelf_boxes = {}
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if GRID[r][c] == 1:
                    boxes_active = {}
                    for l_idx in range(3):
                        for c_idx in range(3):
                            boxes_active[(l_idx, c_idx)] = random.random() < 0.6
                    self.shelf_boxes[(r, c)] = boxes_active
                    
        self.shelf_contents = {k: "box" for k in self.shelf_boxes.keys()}
        self.scanned_shelves = set()
        self.total_boxes_detected = 0
        self.average_occupancy = 0.0
        self.summary_printed = False
        
        # Real-time scan inference output
        self.scan_surf = None
        self.inference_time_ms = 0.0
        self.current_gt_pose = (0.0, 0.0, 0.0, 0.0)
        self.cnn_pred_pose = None
        self.baseline_pred_pose = None
        self.hybrid_pred_pose = None
        self.active_estimation_mode = "Standby"
        self.visible_markers = []
        self.current_box_count = 0
        self.current_occupancy = 0.0
        self.scan_timer = 0
        self.target_shelf_cell = None
        self.scan_flash_time = 0
        
        # UI controls
        self.buttons = [
            {"id": "sweep", "text": "Start Sweep Scan", "rect": pygame.Rect(40, 620, 150, 35), "color": (41, 128, 185)},
            {"id": "dock", "text": "Return to Dock", "rect": pygame.Rect(200, 620, 130, 35), "color": (211, 84, 0)},
            {"id": "query", "text": "Scan Nearest", "rect": pygame.Rect(340, 620, 130, 35), "color": (142, 68, 173)},
            {"id": "mode", "text": "Mode: AUTO", "rect": pygame.Rect(480, 620, 110, 35), "color": (39, 174, 96)},
            {"id": "speed", "text": "Speed: 1x", "rect": pygame.Rect(600, 620, 100, 35), "color": (127, 140, 141)}
        ]

        if self.demo_mode:
            self.mode = "AUTO"
            # Start sweep scan automatically in demo mode
            pygame.time.set_timer(pygame.USEREVENT + 1, 1000) # delay launch by 1s

    def init_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Enable lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 50.0, 0.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.7, 0.7, 0.7, 1.0])
        
        # Deep space dark industrial background
        glClearColor(14/255.0, 14/255.0, 18/255.0, 1.0)

    def get_drone_3d_pos(self):
        c = (self.drone_pixel_x - MAP_OFFSET_X - MAP_CELL_SIZE / 2.0) / MAP_CELL_SIZE
        r = (self.drone_pixel_y - MAP_OFFSET_Y - MAP_CELL_SIZE / 2.0) / MAP_CELL_SIZE
        x = (c - 5.5) * CELL_SIZE_3D
        z = (r - 5.5) * CELL_SIZE_3D
        y = 15.0
        if self.state == "CHARGING":
            y = 1.0
        elif self.state == "SCANNING":
            # Sync with the camera's Y sweep: -0.8 * cos(pi * t)
            # Scaling factor is 10.0 (since OpenGL 15 is center, and 1 camera unit = 10 OpenGL units)
            t_norm = (40.0 - self.scan_timer) / 40.0
            sweep_y = -0.8 * math.cos(math.pi * t_norm)
            y = 15.0 - sweep_y * 10.0
        return x, y, z

    def get_adjacent_shelves(self, r, c):
        adj = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                if GRID[nr][nc] == 1 and (nr, nc) in self.shelf_contents and (nr, nc) not in self.scanned_shelves:
                    adj.append((nr, nc))
        return adj

    def generate_sweep_path(self):
        aisles = [0, 3, 6, 9, 11]
        sweep_path = []
        current_pos = (self.drone_grid_y, self.drone_grid_x)
        
        for i, col in enumerate(aisles):
            rows = range(GRID_ROWS) if i % 2 == 0 else range(GRID_ROWS - 1, -1, -1)
            for row in rows:
                target = (row, col)
                segment = a_star_search(GRID, current_pos, target)
                if segment:
                    if sweep_path and segment[0] == sweep_path[-1]:
                        sweep_path.extend(segment[1:])
                    else:
                        sweep_path.extend(segment)
                    current_pos = target
                    
        return_segment = a_star_search(GRID, current_pos, (0, 0))
        if return_segment:
            sweep_path.extend(return_segment[1:])
        return sweep_path

    def start_sweep(self):
        self.path = self.generate_sweep_path()
        self.path_index = 0
        self.state = "NAVIGATING"
        self.battery = 100.0
        self.scanned_shelves.clear()
        self.total_boxes_detected = 0
        self.average_occupancy = 0.0
        self.summary_printed = False
        print(f"Generated sweep path of length {len(self.path)} steps.")

    def return_to_charger(self):
        start = (self.drone_grid_y, self.drone_grid_x)
        self.path = a_star_search(GRID, start, (0, 0))
        self.path_index = 0
        if self.path:
            self.state = "NAVIGATING"
            print("Returning to charging dock...")
        else:
            self.state = "IDLE"

    def query_nearest_item(self):
        candidates = []
        for (r, c) in self.shelf_boxes.keys():
            if (r, c) not in self.scanned_shelves:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and GRID[nr][nc] != 1:
                        candidates.append(((nr, nc), (r, c)))
                        
        if not candidates:
            print("No unscanned shelves remaining!")
            return

        start = (self.drone_grid_y, self.drone_grid_x)
        best_path = None
        best_shelf = None
        
        for walkable, shelf in candidates:
            p = a_star_search(GRID, start, walkable)
            if p and (best_path is None or len(p) < len(best_path)):
                best_path = p
                best_shelf = shelf
                
        if best_path:
            self.path = best_path
            self.path_index = 0
            self.state = "NAVIGATING"
            print(f"Path planned to scan shelf {best_shelf}.")
        else:
            print("No reachable path to any unscanned shelf.")

    def trigger_replan(self):
        if self.state == "NAVIGATING" and self.path:
            start = (self.drone_grid_y, self.drone_grid_x)
            dest = self.path[-1]
            new_path = a_star_search(GRID, start, dest)
            if new_path:
                self.path = new_path
                self.path_index = 0
                print("Path dynamically re-routed.")
            else:
                print("Path blocked! Stopping drone.")
                self.state = "IDLE"
                self.path = []

    def trigger_scan(self, shelf_cell):
        self.state = "SCANNING"
        self.scan_timer = 40  # Pause for 40 frames
        self.target_shelf_cell = shelf_cell
        self.scan_flash_time = 20
        
        # Centered and positioned at optimal scanning distance (z=1.5) to capture all 3 columns
        self.scan_base_x = 0.0
        self.scan_base_y = 0.0
        self.scan_base_z = 1.5
        self.scan_base_yaw = 0.0
        self.scan_max_boxes_per_level = [0, 0, 0]

    def update_scanning_inference(self):
        if not self.target_shelf_cell:
            return
            
        t = time.time()
        drift_x = 0.08 * math.sin(t * 3.0)
        drift_y = 0.04 * math.sin(t * 4.0)
        drift_z = 0.06 * math.cos(t * 2.0)
        drift_yaw = 3.0 * math.sin(t * 3.5)
        
        # Smooth vertical sweep from -0.8 (top shelf) to 0.8 (bottom shelf) during the scan timer (40 frames)
        t_norm = (40.0 - self.scan_timer) / 40.0
        sweep_y = -0.8 * math.cos(math.pi * t_norm)
        
        x_gt = self.scan_base_x + drift_x
        y_gt = sweep_y + drift_y
        z_gt = self.scan_base_z + drift_z
        yaw_gt = self.scan_base_yaw + drift_yaw
        
        self.current_gt_pose = (x_gt, y_gt, z_gt, yaw_gt)
        
        boxes_active = self.shelf_boxes.get(self.target_shelf_cell, {})
        img_bgr = render_warehouse_view(x_gt, y_gt, z_gt, yaw_gt, boxes_active)
        
        # Add a bit of brightness/contrast to camera feed for display
        img_display = cv2.convertScaleAbs(img_bgr, alpha=1.2, beta=15)
        
        self.visible_markers = detect_aruco_markers(img_display)
        
        # 1. Classical Baseline Pose Estimation
        if self.estimator.baseline_model is not None and self.visible_markers:
            marker_features = {
                10: {"visible": 0.0, "cx": 0.0, "cy": 0.0, "area": 0.0},
                11: {"visible": 0.0, "cx": 0.0, "cy": 0.0, "area": 0.0},
                12: {"visible": 0.0, "cx": 0.0, "cy": 0.0, "area": 0.0},
                13: {"visible": 0.0, "cx": 0.0, "cy": 0.0, "area": 0.0}
            }
            for m in self.visible_markers:
                m_id = m["id"]
                if m_id in marker_features:
                    marker_features[m_id] = {
                        "visible": 1.0, "cx": m["cx"], "cy": m["cy"], "area": m["area"]
                    }
            feats = []
            for mid in [10, 11, 12, 13]:
                feats.extend([
                    marker_features[mid]["visible"], marker_features[mid]["cx"],
                    marker_features[mid]["cy"], marker_features[mid]["area"]
                ])
            X = np.array([feats], dtype=np.float32)
            X_scaled = self.estimator.baseline_scaler.transform(X)
            self.baseline_pred_pose = self.estimator.baseline_model.predict(X_scaled)[0]
        else:
            self.baseline_pred_pose = None
            
        # 2. Deep PyTorch CNN Pose Regression
        if self.estimator.cnn_model is not None and self.estimator.pose_scaler is not None:
            img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            t_start = time.time()
            with torch.no_grad():
                pred = self.estimator.cnn_model(tensor).to(torch.float32)
            self.cnn_pred_pose = self.estimator.pose_scaler.inverse_transform(pred.cpu().numpy())[0]
            self.inference_time_ms = (time.time() - t_start) * 1000.0
        else:
            self.cnn_pred_pose = None

        # 3. Hybrid Pose Estimation
        self.hybrid_pred_pose, self.active_estimation_mode = self.estimator.predict(img_display, self.visible_markers)
            
        # 4. OpenCV Box Detection & Counting
        box_count, occupancy, annotated_img, detected_boxes = detect_and_count_boxes(img_display, return_boxes=True)
        
        # Determine which level the camera is currently directly in front of
        level_idx = None
        if y_gt < -0.7:
            level_idx = 0
        elif y_gt > 0.7:
            level_idx = 2
        elif -0.1 <= y_gt <= 0.1:
            level_idx = 1
            
        if level_idx is not None:
            # Count boxes whose center is on-screen
            center_band_count = 0
            for box in detected_boxes:
                bx, by, bw, bh, est_cnt = box
                # Ignore highly clipped boxes (touching top/bottom edges with height < 80)
                if (by <= 1 or by + bh >= 479) and bh < 80:
                    continue
                v_center = by + bh / 2.0
                if 40.0 <= v_center <= 240.0:
                    center_band_count += est_cnt
            
            # Update peak counts for this level
            self.scan_max_boxes_per_level[level_idx] = max(self.scan_max_boxes_per_level[level_idx], center_band_count)
            
        self.current_box_count = sum(self.scan_max_boxes_per_level)
        self.current_occupancy = min(1.0, float(self.current_box_count) / 9.0)
        
        # 5. Overlays for Cinematic HUD in Camera feed
        # Bounding box is already green. Draw detected ArUco Marker ID tags in red
        for m in self.visible_markers:
            cx, cy = int(m["cx"]), int(m["cy"])
            cv2.circle(annotated_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(annotated_img, f"ID:{m['id']}", (cx + 8, cy + 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            
        # Draw camera axis overlay
        ax_x, ax_y = 285, 35
        cv2.line(annotated_img, (ax_x, ax_y), (ax_x + 15, ax_y), (0, 0, 255), 2)
        cv2.putText(annotated_img, "X", (ax_x + 18, ax_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.line(annotated_img, (ax_x, ax_y), (ax_x, ax_y + 15), (0, 255, 0), 2)
        cv2.putText(annotated_img, "Y", (ax_x - 3, ax_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.line(annotated_img, (ax_x, ax_y), (ax_x - 10, ax_y - 10), (255, 0, 0), 2)
        cv2.putText(annotated_img, "Z", (ax_x - 16, ax_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
        
        # Draw Scanning Overlay
        if self.state == "SCANNING":
            blink = int(time.time() * 5) % 2 == 0
            if blink:
                cv2.rectangle(annotated_img, (10, 50), (120, 72), (0, 0, 255), -1)
                cv2.putText(annotated_img, "SCANNING...", (16, 66), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                
        # Draw confidence/drift bar
        if self.hybrid_pred_pose is not None:
            err_xyz = math.sqrt((self.hybrid_pred_pose[0]-x_gt)**2 + 
                                (self.hybrid_pred_pose[1]-y_gt)**2 + 
                                (self.hybrid_pred_pose[2]-z_gt)**2)
            # Drift bar at bottom
            err_pct = min(1.0, err_xyz / 1.5)
            bar_w = int(180 * err_pct)
            cv2.rectangle(annotated_img, (15, 220), (195, 230), (50, 50, 55), -1)
            bar_color = (0, 0, 255) if err_pct > 0.5 else (46, 204, 113)
            cv2.rectangle(annotated_img, (15, 220), (15 + bar_w, 230), bar_color, -1)
            cv2.putText(annotated_img, f"Estimator Drift: {err_xyz:.3f}m", (15, 215), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            # Save screenshots at key events automatically
            if self.record_mode or self.demo_mode:
                if not self.screenshots_taken["scan"] and self.state == "SCANNING" and len(self.scanned_shelves) == 0:
                    self.screenshots_taken["scan"] = True
                    cv2.imwrite(os.path.join(self.output_dir, "drone_scanning_shelf.png"), annotated_img)
                    print("Saved screenshot: drone_scanning_shelf.png")
                if not self.screenshots_taken["camera"] and len(self.visible_markers) > 0:
                    self.screenshots_taken["camera"] = True
                    cv2.imwrite(os.path.join(self.output_dir, "camera_feed_detection.png"), annotated_img)
                    print("Saved screenshot: camera_feed_detection.png")

        # Convert to Pygame
        h, w, c = annotated_img.shape
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(annotated_rgb.tobytes(), (w, h), "RGB")
        self.scan_surf = pygame.transform.scale(surf, (320, 240))

    def update(self):
        self.drone_speed = 3.5 * self.simulation_speed
        
        # Autopilot logic for AUTO mode
        if self.mode == "AUTO" and not self.demo_mode:
            # Low battery fallback: go charge if battery < 20% and not at dock
            if self.battery < 20.0 and self.state != "CHARGING" and (self.drone_grid_y, self.drone_grid_x) != (0, 0):
                if self.state == "IDLE" or (self.state == "NAVIGATING" and (not self.path or self.path[-1] != (0, 0))):
                    print("Autopilot: Low battery alert! Flying back to charging dock.")
                    self.return_to_charger()
            # Idle action: seek work or charge
            elif self.state == "IDLE":
                if (self.drone_grid_y, self.drone_grid_x) == (0, 0):
                    if self.battery > 95.0:
                        # Check if any shelves remain unscanned
                        unscanned = [k for k in self.shelf_boxes.keys() if k not in self.scanned_shelves]
                        if unscanned:
                            print("Autopilot: Battery full! Launching new sweep scan mission.")
                            self.start_sweep()
                else:
                    unscanned = [k for k in self.shelf_boxes.keys() if k not in self.scanned_shelves]
                    if unscanned:
                        print("Autopilot: Finding next nearest unscanned shelf.")
                        self.query_nearest_item()
                    else:
                        print("Autopilot: All shelves scanned! Returning to dock.")
                        self.return_to_charger()

        if self.state in ["NAVIGATING", "SCANNING"]:
            self.rotor_angle += 0.55 * self.simulation_speed
            
        # Camera orbit in demo mode
        if self.demo_mode:
            # Automatic slow pan/zoom
            t = time.time()
            self.camera_yaw = (t * 8.0) % 360
            self.camera_pitch = 30.0 + 8.0 * math.sin(t * 0.4)
            self.camera_dist = 220.0 + 30.0 * math.cos(t * 0.3)
            
            # Smoothly focus camera look_at on drone position
            dx, dy, dz = self.get_drone_3d_pos()
            self.camera_look_at = self.camera_look_at * 0.9 + np.array([dx, dy, dz]) * 0.1

        if self.state == "CHARGING":
            self.battery = min(100.0, self.battery + 1.5 * self.simulation_speed)
            if self.battery >= 100.0:
                if self.demo_mode:
                    # In demo mode, exit cleanly once fully scanned and returned
                    if len(self.scanned_shelves) > 0:
                        self.running = False
                        print("Demo sequence completed successfully.")
                else:
                    self.state = "IDLE"
            return
            
        if self.state in ["NAVIGATING", "SCANNING"]:
            # Increased battery capacity: decreased depletion rate from 0.04 to 0.008 (lasts 5x longer)
            self.battery = max(0.0, self.battery - 0.008 * self.simulation_speed)
            if self.battery <= 0:
                self.state = "IDLE"
                self.path = []
                print("Battery depleted!")
                return
                
            if self.mode == "AUTO" and self.battery < 20.0 and self.state == "NAVIGATING" and (self.drone_grid_x, self.drone_grid_y) != (0, 0):
                print("Battery Low! Heading back to dock.")
                self.return_to_charger()
                return

        if self.state == "SCANNING":
            self.update_scanning_inference()
            self.scan_timer -= 1 * self.simulation_speed
            if self.scan_timer <= 0:
                shelf = self.target_shelf_cell
                if shelf and shelf not in self.scanned_shelves:
                    self.scanned_shelves.add(shelf)
                    self.total_boxes_detected += self.current_box_count
                    self.average_occupancy = self.total_boxes_detected / (len(self.scanned_shelves) * 9.0)
                    
                    if len(self.scanned_shelves) == len(self.shelf_boxes) and not self.summary_printed:
                        self.summary_printed = True
                        print("\n" + "="*50)
                        print("         WAREHOUSE SCAN MISSION COMPLETE")
                        print("="*50)
                        print(f"Total Shelves Scanned:   {len(self.scanned_shelves)} / {len(self.shelf_boxes)} (100.0%)")
                        print(f"Total Boxes Detected:    {self.total_boxes_detected} boxes")
                        print(f"Average Shelf Occupancy:  {self.average_occupancy * 100:.1f} %")
                        avg_boxes = self.total_boxes_detected / len(self.shelf_boxes) if len(self.shelf_boxes) > 0 else 0.0
                        print(f"Average Boxes per Shelf:  {avg_boxes:.2f} boxes")
                        print(f"AI Localization Mode:     Hybrid Estimator (CNN + AdaBoost)")
                        print(f"Peak Model Inference:     {self.inference_time_ms:.1f} ms / frame")
                        print(f"Active Autopilot Status:  Scan Complete, Returning to Dock")
                        print("="*50 + "\n")
                
                unscanned = self.get_adjacent_shelves(self.drone_grid_y, self.drone_grid_x)
                if unscanned:
                    self.trigger_scan(unscanned[0])
                else:
                    self.state = "NAVIGATING"
                    self.target_shelf_cell = None
            return

        if self.state == "NAVIGATING":
            if self.path_index < len(self.path):
                target_node = self.path[self.path_index]
                target_r, target_c = target_node
                
                target_px = MAP_OFFSET_X + target_c * MAP_CELL_SIZE + MAP_CELL_SIZE // 2
                target_py = MAP_OFFSET_Y + target_r * MAP_CELL_SIZE + MAP_CELL_SIZE // 2
                
                dx = target_px - self.drone_pixel_x
                dy = target_py - self.drone_pixel_y
                dist = math.hypot(dx, dy)
                
                if dist > 0.1:
                    target_angle = math.atan2(dy, dx)
                    diff = target_angle - self.drone_angle
                    diff = (diff + math.pi) % (2 * math.pi) - math.pi
                    self.drone_angle += diff * 0.25
                
                if dist <= self.drone_speed:
                    self.drone_pixel_x = float(target_px)
                    self.drone_pixel_y = float(target_py)
                    self.drone_grid_x = target_c
                    self.drone_grid_y = target_r
                    self.path_index += 1
                    
                    # Do not dock on intermediate cells (only when the path is fully completed)
                    unscanned = self.get_adjacent_shelves(self.drone_grid_y, self.drone_grid_x)
                    if unscanned:
                        self.trigger_scan(unscanned[0])
                else:
                    self.drone_pixel_x += (dx / dist) * self.drone_speed
                    self.drone_pixel_y += (dy / dist) * self.drone_speed
            else:
                self.state = "IDLE"
                self.path = []
                if GRID[self.drone_grid_y][self.drone_grid_x] == 2:
                    self.state = "CHARGING"
                    print("Drone docked at charging station. Recharging...")
                    return
                if self.demo_mode:
                    # Finished sweep in demo, go back to dock
                    self.return_to_charger()

    def draw_rack_3d(self, px, pz, shelf_pos):
        half = (CELL_SIZE_3D - 4) / 2.0
        
        is_scanned = shelf_pos in self.scanned_shelves
        is_target = self.state == "SCANNING" and self.target_shelf_cell == shelf_pos
        is_planned = self.path and self.path[-1] == shelf_pos
        
        if is_target:
            frame_color = (231, 76, 60) # red
            glow_color = (231, 76, 60)
        elif is_scanned:
            frame_color = (46, 204, 113) # green
            glow_color = (46, 204, 113)
        elif is_planned:
            frame_color = (241, 196, 15) # yellow
            glow_color = (241, 196, 15)
        else:
            frame_color = (90, 95, 100) # grey
            glow_color = None
            
        # Draw pillars
        pw = 0.8
        self.draw_cube_local(px - half, 15.0, pz - half, pw, 30.0, pw, frame_color)
        self.draw_cube_local(px + half, 15.0, pz - half, pw, 30.0, pw, frame_color)
        self.draw_cube_local(px + half, 15.0, pz + half, pw, 30.0, pw, frame_color)
        self.draw_cube_local(px - half, 15.0, pz + half, pw, 30.0, pw, frame_color)
        
        # Horizontal shelves levels
        levels = [1.0, 11.0, 21.0, 30.0]
        for lvl in levels:
            self.draw_cube_local(px, lvl, pz - half, CELL_SIZE_3D - 4, 0.4, 0.4, frame_color)
            self.draw_cube_local(px, lvl, pz + half, CELL_SIZE_3D - 4, 0.4, 0.4, frame_color)
            self.draw_cube_local(px - half, lvl, pz, 0.4, 0.4, CELL_SIZE_3D - 4, frame_color)
            self.draw_cube_local(px + half, lvl, pz, 0.4, 0.4, CELL_SIZE_3D - 4, frame_color)
            
        # Wooden pallet on base
        self.draw_cube_local(px, 0.2, pz, CELL_SIZE_3D - 4.5, 0.4, CELL_SIZE_3D - 4.5, (140, 100, 60))
        
        # Boxes on levels
        boxes_active = self.shelf_boxes.get(shelf_pos, {})
        box_size = 4.2
        beam_heights = [1.0, 11.0, 21.0]
        col_offsets = [-5.0, 0.0, 5.0]
        
        for l_idx, lvl_y in enumerate(beam_heights):
            for c_idx, offset_x in enumerate(col_offsets):
                if boxes_active.get((l_idx, c_idx), False) or boxes_active.get(f"{l_idx},{c_idx}", False):
                    # Deterministic but slightly random size/color based on index
                    seed_idx = shelf_pos[0] * 100 + shelf_pos[1] * 10 + l_idx * 3 + c_idx
                    np.random.seed(seed_idx)
                    dw = np.random.uniform(-0.5, 0.5)
                    dh = np.random.uniform(-0.5, 0.5)
                    dd = np.random.uniform(-0.5, 0.5)
                    
                    r_col = int(135 + np.random.uniform(-20, 20))
                    g_col = int(105 + np.random.uniform(-15, 15))
                    b_col = int(75 + np.random.uniform(-10, 10))
                    
                    bx = px + offset_x
                    by = lvl_y + (box_size + dh)/2.0
                    bz = pz - 1.0 + np.random.uniform(-0.5, 0.5)
                    
                    self.draw_cube_local(bx, by, bz, box_size + dw, box_size + dh, box_size + dd, (r_col, g_col, b_col), (100, 75, 50))
                    
        # Glow marker at top of shelf
        if glow_color:
            self.draw_cube_local(px, 30.5, pz, 1.2, 0.8, 1.2, glow_color)

    def draw_3d_shelves(self):
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                cell_val = GRID[r][c]
                px = (c - 5.5) * CELL_SIZE_3D
                pz = (r - 5.5) * CELL_SIZE_3D
                
                if cell_val == 2:  # Charging Dock
                    # Metal base pad
                    self.draw_cube_local(px, 0.2, pz, CELL_SIZE_3D - 2, 0.4, CELL_SIZE_3D - 2, (35, 40, 45), (46, 204, 113))
                    self.draw_cube_local(px, 0.5, pz, 10.0, 0.2, 10.0, (110, 115, 120), (241, 196, 15))
                elif cell_val == 1:  # Shelf/Obstacle
                    shelf_pos = (r, c)
                    if shelf_pos in self.temp_obstacles:
                        # Blinking red alert cube
                        blink = int(time.time() * 5) % 2 == 0
                        color = (231, 76, 60) if blink else (150, 40, 40)
                        self.draw_cube_local(px, 10.0, pz, CELL_SIZE_3D - 4, 20.0, CELL_SIZE_3D - 4, color, (192, 57, 43))
                    else:
                        self.draw_rack_3d(px, pz, shelf_pos)

    def draw_floor_grid(self):
        glDisable(GL_LIGHTING)
        # Industrial floor lines
        glColor4f(35/255.0, 35/255.0, 40/255.0, 1.0)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        half_w = 6.0 * CELL_SIZE_3D
        for i in range(13):
            coord = (i - 6.0) * CELL_SIZE_3D
            glVertex3f(coord, 0.0, -half_w); glVertex3f(coord, 0.0, half_w)
            glVertex3f(-half_w, 0.0, coord); glVertex3f(half_w, 0.0, coord)
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_warehouse_env(self):
        glDisable(GL_LIGHTING)
        w_limit = 6.0 * CELL_SIZE_3D
        h_wall = 35.0
        
        # Walls outlines
        glColor3f(45/255.0, 45/255.0, 50/255.0)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(-w_limit, 0.0, -w_limit)
        glVertex3f(w_limit, 0.0, -w_limit)
        glVertex3f(w_limit, 0.0, w_limit)
        glVertex3f(-w_limit, 0.0, w_limit)
        glEnd()
        
        glBegin(GL_LINES)
        glVertex3f(-w_limit, 0.0, -w_limit); glVertex3f(-w_limit, h_wall, -w_limit)
        glVertex3f(w_limit, 0.0, -w_limit); glVertex3f(w_limit, h_wall, -w_limit)
        glVertex3f(w_limit, 0.0, w_limit); glVertex3f(w_limit, h_wall, w_limit)
        glVertex3f(-w_limit, 0.0, w_limit); glVertex3f(-w_limit, h_wall, w_limit)
        glEnd()
        
        glBegin(GL_LINE_LOOP)
        glVertex3f(-w_limit, h_wall, -w_limit)
        glVertex3f(w_limit, h_wall, -w_limit)
        glVertex3f(w_limit, h_wall, w_limit)
        glVertex3f(-w_limit, h_wall, w_limit)
        glEnd()
        
        # Ceiling support girders/beams
        glColor3f(35/255.0, 35/255.0, 40/255.0)
        glBegin(GL_LINES)
        for c_val in np.linspace(-w_limit, w_limit, 7):
            glVertex3f(c_val, h_wall, -w_limit); glVertex3f(c_val, h_wall, w_limit)
            glVertex3f(-w_limit, h_wall, c_val); glVertex3f(w_limit, h_wall, c_val)
        glEnd()
        
        glEnable(GL_LIGHTING)
        
        # Warehouse ceiling lights
        light_color = (255, 255, 230)
        for x_val in np.linspace(-w_limit + 40, w_limit - 40, 3):
            for z_val in np.linspace(-w_limit + 40, w_limit - 40, 3):
                self.draw_cube_local(x_val, h_wall - 0.5, z_val, 4.0, 0.6, 4.0, (70, 70, 75))
                self.draw_cube_local(x_val, h_wall - 1.0, z_val, 1.8, 0.3, 1.8, light_color)
                
        # Aisle hanging label boards
        aisle_x = [-50.0, 10.0, 70.0]
        for ax, label in zip(aisle_x, ["AISLE A1", "AISLE A2", "AISLE B1"]):
            sz = -w_limit + 15.0
            glDisable(GL_LIGHTING)
            glColor3f(90/255.0, 90/255.0, 95/255.0)
            glBegin(GL_LINES)
            glVertex3f(ax - 2.5, h_wall, sz); glVertex3f(ax - 2.5, h_wall - 6.0, sz)
            glVertex3f(ax + 2.5, h_wall, sz); glVertex3f(ax + 2.5, h_wall - 6.0, sz)
            glEnd()
            glEnable(GL_LIGHTING)
            
            # Yellow signboard
            self.draw_cube_local(ax, h_wall - 7.0, sz, 9.0, 2.2, 0.4, (241, 196, 15), (40, 40, 45))

    def draw_path_3d(self):
        if len(self.path) > 1:
            glDisable(GL_LIGHTING)
            glLineWidth(4.0)
            # Blue neon glow
            glColor4f(41/255.0, 128/255.0, 185/255.0, 0.7)
            glBegin(GL_LINE_STRIP)
            for idx in range(max(0, self.path_index - 1), len(self.path)):
                node = self.path[idx]
                px = (node[1] - 5.5) * CELL_SIZE_3D
                pz = (node[0] - 5.5) * CELL_SIZE_3D
                glVertex3f(px, 1.0, pz)
            glEnd()
            
            # White core neon line
            glLineWidth(2.0)
            glColor4f(255/255.0, 255/255.0, 255/255.0, 1.0)
            glBegin(GL_LINE_STRIP)
            for idx in range(max(0, self.path_index - 1), len(self.path)):
                node = self.path[idx]
                px = (node[1] - 5.5) * CELL_SIZE_3D
                pz = (node[0] - 5.5) * CELL_SIZE_3D
                glVertex3f(px, 1.1, pz)
            glEnd()
            glEnable(GL_LIGHTING)

    def draw_laser_3d(self):
        if self.state == "SCANNING" and self.target_shelf_cell:
            tr, tc = self.target_shelf_cell
            target_x = (tc - 5.5) * CELL_SIZE_3D
            target_z = (tr - 5.5) * CELL_SIZE_3D
            drone_x, drone_y, drone_z = self.get_drone_3d_pos()
            
            glDisable(GL_LIGHTING)
            half = CELL_SIZE_3D / 2.0
            
            # Outer transparent green scan cone
            glColor4f(46/255.0, 204/255.0, 113/255.0, 0.12)
            glBegin(GL_TRIANGLES)
            glVertex3f(drone_x, drone_y, drone_z); glVertex3f(target_x - half, 0.0, target_z - half); glVertex3f(target_x + half, 0.0, target_z - half)
            glVertex3f(drone_x, drone_y, drone_z); glVertex3f(target_x + half, 0.0, target_z - half); glVertex3f(target_x + half, 0.0, target_z + half)
            glVertex3f(drone_x, drone_y, drone_z); glVertex3f(target_x + half, 0.0, target_z + half); glVertex3f(target_x - half, 0.0, target_z + half)
            glVertex3f(drone_x, drone_y, drone_z); glVertex3f(target_x - half, 0.0, target_z + half); glVertex3f(target_x - half, 0.0, target_z - half)
            glEnd()
            
            # Inner pulsing cone
            pulse = 0.4 + 0.6 * abs(math.sin(time.time() * 7.0))
            ih = half * pulse
            glColor4f(46/255.0, 204/255.0, 113/255.0, 0.22)
            glBegin(GL_TRIANGLES)
            glVertex3f(drone_x, drone_y, drone_z); glVertex3f(target_x - ih, 0.0, target_z - ih); glVertex3f(target_x + ih, 0.0, target_z - ih)
            glVertex3f(drone_x, drone_y, drone_z); glVertex3f(target_x + ih, 0.0, target_z - ih); glVertex3f(target_x + ih, 0.0, target_z + ih)
            glVertex3f(drone_x, drone_y, drone_z); glVertex3f(target_x + ih, 0.0, target_z + ih); glVertex3f(target_x - ih, 0.0, target_z + ih)
            glVertex3f(drone_x, drone_y, drone_z); glVertex3f(target_x - ih, 0.0, target_z + ih); glVertex3f(target_x - ih, 0.0, target_z - ih)
            glEnd()
            
            # Sweep neon green line
            sweep_ratio = 0.5 + 0.5 * math.sin(time.time() * 11 * self.simulation_speed)
            sweep_x = target_x + (sweep_ratio - 0.5) * CELL_SIZE_3D
            sweep_z = target_z + (sweep_ratio - 0.5) * CELL_SIZE_3D
            glLineWidth(4.0)
            glColor4f(46/255.0, 204/255.0, 113/255.0, 1.0)
            glBegin(GL_LINES)
            glVertex3f(drone_x, drone_y, drone_z)
            glVertex3f(sweep_x, 15.0, sweep_z)
            glEnd()
            glEnable(GL_LIGHTING)

    def draw_drone_3d(self):
        dx, dy, dz = self.get_drone_3d_pos()
        
        glPushMatrix()
        glTranslatef(dx, dy, dz)
        glRotatef(-math.degrees(self.drone_angle), 0.0, 1.0, 0.0)
        
        # 1. Carbon fiber arms
        r_arm = 4.0
        arm_angles = [math.pi/4, 3*math.pi/4, 5*math.pi/4, 7*math.pi/4]
        for ang in arm_angles:
            ax = r_arm * math.cos(ang)
            az = r_arm * math.sin(ang)
            self.draw_cube_local(ax/2.0, 0.0, az/2.0, abs(ax), 0.35, abs(az), (60, 60, 65))
            
        # Navigation LEDs on tips
        self.draw_cube_local(r_arm * math.cos(3*math.pi/4), 0.25, r_arm * math.sin(3*math.pi/4), 0.4, 0.4, 0.4, (255, 0, 0)) # Red left
        self.draw_cube_local(r_arm * math.cos(5*math.pi/4), 0.25, r_arm * math.sin(5*math.pi/4), 0.4, 0.4, 0.4, (255, 0, 0))
        self.draw_cube_local(r_arm * math.cos(math.pi/4), 0.25, r_arm * math.sin(math.pi/4), 0.4, 0.4, 0.4, (0, 255, 0)) # Green right
        self.draw_cube_local(r_arm * math.cos(7*math.pi/4), 0.25, r_arm * math.sin(7*math.pi/4), 0.4, 0.4, 0.4, (0, 255, 0))
        
        # 2. Main streamlined chassis
        self.draw_cube_local(0.0, 0.0, 0.0, 3.2, 0.8, 3.2, (35, 35, 40), (80, 80, 90))
        # Streamlined battery cover
        self.draw_cube_local(0.0, 0.6, -0.2, 2.0, 0.4, 2.6, (41, 128, 185)) # Metallic blue
        
        # 3. Blinking Status LED
        led_color = (0, 255, 255)
        if self.state == "SCANNING":
            led_color = (255, 0, 0) if int(time.time() * 8) % 2 == 0 else (120, 0, 0)
        elif self.state == "CHARGING":
            led_color = (0, 255, 0) if int(time.time() * 4) % 2 == 0 else (0, 120, 0)
        elif self.state == "IDLE":
            led_color = (0, 255, 0)
        self.draw_cube_local(0.0, 0.9, 0.0, 0.6, 0.3, 0.6, led_color)
        
        # 4. Camera gimbal bottom front
        self.draw_cube_local(1.2, -0.6, 0.0, 0.8, 0.8, 0.8, (20, 20, 20))
        self.draw_cube_local(1.5, -0.6, 0.0, 0.4, 0.4, 0.4, (100, 100, 255)) # glass lens
        
        # 5. Rotors spinning
        rotor_r = 2.2
        for i, ang in enumerate(arm_angles):
            rx = r_arm * math.cos(ang)
            rz = r_arm * math.sin(ang)
            self.draw_cube_local(rx, 0.15, rz, 0.8, 0.4, 0.8, (45, 45, 45))
            
            rot_dir = 1 if i % 2 == 0 else -1
            angle_rotor = self.rotor_angle * rot_dir
            bx = rotor_r * math.cos(angle_rotor)
            bz = rotor_r * math.sin(angle_rotor)
            
            glDisable(GL_LIGHTING)
            glLineWidth(2.0)
            glColor4f(230/255.0, 230/255.0, 240/255.0, 0.95)
            glBegin(GL_LINES)
            glVertex3f(rx - bx, 0.35, rz - bz)
            glVertex3f(rx + bx, 0.35, rz + bz)
            glEnd()
            
            # Guards
            glColor4f(110/255.0, 120/255.0, 130/255.0, 0.2)
            glBegin(GL_LINE_LOOP)
            for seg in range(12):
                theta = 2.0 * math.pi * seg / 12.0
                glVertex3f(rx + rotor_r * math.cos(theta), 0.35, rz + rotor_r * math.sin(theta))
            glEnd()
            glEnable(GL_LIGHTING)
            
        glPopMatrix()

    def draw_cube_local(self, x, y, z, sx, sy, sz, color, outline_color=None):
        x1, x2 = x - sx/2.0, x + sx/2.0
        y1, y2 = y - sy/2.0, y + sy/2.0
        z1, z2 = z - sz/2.0, z + sz/2.0
        
        glColor4f(color[0]/255.0, color[1]/255.0, color[2]/255.0, 1.0)
        glBegin(GL_QUADS)
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(x1, y2, z1); glVertex3f(x1, y2, z2); glVertex3f(x2, y2, z2); glVertex3f(x2, y2, z1)
        glNormal3f(0.0, -1.0, 0.0)
        glVertex3f(x1, y1, z1); glVertex3f(x2, y1, z1); glVertex3f(x2, y1, z2); glVertex3f(x1, y1, z2)
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(x1, y1, z2); glVertex3f(x2, y1, z2); glVertex3f(x2, y2, z2); glVertex3f(x1, y2, z2)
        glNormal3f(0.0, 0.0, -1.0)
        glVertex3f(x1, y1, z1); glVertex3f(x1, y2, z1); glVertex3f(x2, y2, z1); glVertex3f(x2, y1, z1)
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(x1, y1, z1); glVertex3f(x1, y1, z2); glVertex3f(x1, y2, z2); glVertex3f(x1, y2, z1)
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(x2, y1, z1); glVertex3f(x2, y2, z1); glVertex3f(x2, y2, z2); glVertex3f(x2, y1, z2)
        glEnd()
        
        if outline_color:
            glDisable(GL_LIGHTING)
            glLineWidth(1.0)
            glColor3f(outline_color[0]/255.0, outline_color[1]/255.0, outline_color[2]/255.0)
            glBegin(GL_LINES)
            glVertex3f(x1, y2, z1); glVertex3f(x1, y2, z2)
            glVertex3f(x1, y2, z2); glVertex3f(x2, y2, z2)
            glVertex3f(x2, y2, z2); glVertex3f(x2, y2, z1)
            glVertex3f(x2, y2, z1); glVertex3f(x1, y2, z1)
            glVertex3f(x1, y1, z1); glVertex3f(x1, y1, z2)
            glVertex3f(x1, y1, z2); glVertex3f(x2, y1, z2)
            glVertex3f(x2, y1, z2); glVertex3f(x2, y1, z1)
            glVertex3f(x2, y1, z1); glVertex3f(x1, y1, z1)
            glVertex3f(x1, y1, z1); glVertex3f(x1, y2, z1)
            glVertex3f(x1, y1, z2); glVertex3f(x1, y2, z2)
            glVertex3f(x2, y1, z2); glVertex3f(x2, y2, z2)
            glVertex3f(x2, y1, z1); glVertex3f(x2, y2, z1)
            glEnd()
            glEnable(GL_LIGHTING)

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 3D Viewport
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (WINDOW_WIDTH / WINDOW_HEIGHT), 1.0, 2000.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Calculate camera spherical coords relative to target focal center
        pitch_rad = math.radians(self.camera_pitch)
        yaw_rad = math.radians(self.camera_yaw)
        
        cam_y = self.camera_dist * math.sin(pitch_rad)
        h_dist = self.camera_dist * math.cos(pitch_rad)
        cam_x = h_dist * math.sin(yaw_rad)
        cam_z = h_dist * math.cos(yaw_rad)
        
        gluLookAt(cam_x + self.camera_look_at[0], cam_y + self.camera_look_at[1], cam_z + self.camera_look_at[2],
                  self.camera_look_at[0], self.camera_look_at[1], self.camera_look_at[2],
                  0.0, 1.0, 0.0)
        
        # Draw 3D elements
        self.draw_floor_grid()
        self.draw_warehouse_env()
        self.draw_3d_shelves()
        self.draw_path_3d()
        self.draw_laser_3d()
        self.draw_drone_3d()

        # 2D HUD overlay
        self.hud_surface.fill((0, 0, 0, 0)) # transparent
        self.draw_hud_2d()
        self.render_hud_overlay()
        
        # Capture screenshots at specific moments automatically for demo requirements
        if self.record_mode or self.demo_mode:
            # 1. Overview screenshot (early flight)
            if not self.screenshots_taken["overview"] and self.state == "NAVIGATING" and self.path_index >= 5:
                self.screenshots_taken["overview"] = True
                self.capture_gl_screenshot("overview_warehouse.png")
            # 2. AI Pose Panel screenshot (scanning with active models)
            if not self.screenshots_taken["pose"] and self.state == "SCANNING" and self.baseline_pred_pose is not None:
                self.screenshots_taken["pose"] = True
                self.capture_gl_screenshot("ai_pose_panel.png")
            # 3. Drone scanning shelf screenshot (active laser scan)
            if not self.screenshots_taken["scan"] and self.state == "SCANNING" and self.scan_timer == 20:
                self.screenshots_taken["scan"] = True
                self.capture_gl_screenshot("drone_scanning_shelf.png")
            # 4. Camera feed detection screenshot (scanning with markers visible)
            if not self.screenshots_taken["camera"] and self.state == "SCANNING" and len(self.visible_markers) > 0:
                self.screenshots_taken["camera"] = True
                self.capture_gl_screenshot("camera_feed_detection.png")
        
        # Save frame to disk if recording mode is active
        if self.record_mode:
            self.save_recording_frame()
            
        pygame.display.flip()

    def capture_gl_screenshot(self, filename):
        pixels = glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(WINDOW_HEIGHT, WINDOW_WIDTH, 3)
        image = np.flipud(image)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.output_dir, filename), image_bgr)
        print(f"Saved screenshot: {filename}")

    def save_recording_frame(self):
        pixels = glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(WINDOW_HEIGHT, WINDOW_WIDTH, 3)
        image = np.flipud(image)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frame_name = f"frame_{self.frame_counter:05d}.png"
        cv2.imwrite(os.path.join(self.frames_dir, frame_name), image_bgr)
        self.frame_counter += 1

    def render_hud_overlay(self):
        data = pygame.image.tostring(self.hud_surface, "RGBA", True)
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex2f(0, 0)
        glTexCoord2f(1.0, 1.0); glVertex2f(WINDOW_WIDTH, 0)
        glTexCoord2f(1.0, 0.0); glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT)
        glTexCoord2f(0.0, 0.0); glVertex2f(0, WINDOW_HEIGHT)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glDeleteTextures([tex_id])

    def draw_hud_2d(self):
        # 1. Clean Top Status Strip (HUD Dashboard)
        pygame.draw.rect(self.hud_surface, (18, 18, 24, 230), (0, 0, WINDOW_WIDTH, 50))
        pygame.draw.rect(self.hud_surface, (45, 45, 55, 255), (0, 0, WINDOW_WIDTH, 50), 1)
        
        # Calculate dynamic FPS
        fps = int(self.clock.get_fps())
        
        # Renders text in top bar
        strip_text = [
            (f"MODE: {self.mode}", (46, 204, 113) if self.mode == "AUTO" else (231, 76, 60)),
            (f"STATUS: {self.state}", (52, 152, 219) if self.state == "NAVIGATING" else ((231, 76, 60) if self.state == "SCANNING" else (142, 68, 173))),
            (f"BATTERY: {self.battery:.1f}%", (255, 255, 255)),
            (f"FPS: {fps}", (241, 196, 15)),
            (f"CNN LATENCY: {self.inference_time_ms:.1f}ms", (155, 89, 182)),
            (f"SCANNED: {len(self.scanned_shelves)}/{len(self.shelf_boxes)}", (230, 126, 34)),
            (f"BOXES: {self.total_boxes_detected}", (46, 204, 113))
        ]
        
        spacing_x = 135
        for idx, (txt, col) in enumerate(strip_text):
            rendered = self.font_bold.render(txt, True, col)
            self.hud_surface.blit(rendered, (25 + idx * spacing_x, 18))

        # Title/Subtitle text
        title_surf = self.font_title.render("Intelligent Warehouse Autonomous 3D Inventory Dashboard", True, (255, 255, 255))
        self.hud_surface.blit(title_surf, (40, 65))
        
        # --- RIGHT PANEL: CAMERA FEED & STATS (Translucent design with thin border) ---
        pygame.draw.rect(self.hud_surface, (18, 18, 24, 210), (620, 105, 360, 500), border_radius=8)
        pygame.draw.rect(self.hud_surface, (58, 58, 68, 255), (620, 105, 360, 500), 1, border_radius=8)
        
        header_cam = self.font_header.render("Real-Time Drone Camera Feed", True, (255, 255, 255))
        self.hud_surface.blit(header_cam, (640, 115))
        
        # Camera display box
        cam_box = pygame.Rect(640, 145, 320, 240)
        pygame.draw.rect(self.hud_surface, (12, 12, 16), cam_box)
        
        if self.scan_surf:
            self.hud_surface.blit(self.scan_surf, (640, 145))
        else:
            standby_surf = self.font_body.render("[ CAMERA FEED STANDBY ]", True, (100, 100, 100))
            self.hud_surface.blit(standby_surf, (715, 250))
            
        # Compact AI Pose Estimation Panel
        pygame.draw.rect(self.hud_surface, (12, 12, 16, 220), (640, 395, 320, 195), border_radius=6)
        pygame.draw.rect(self.hud_surface, (48, 48, 58), (640, 395, 320, 195), 1, border_radius=6)
        lbl_ai = self.font_bold.render("AI DRONE POSE ESTIMATION (X, Y, Z, YAW)", True, (150, 150, 150))
        self.hud_surface.blit(lbl_ai, (650, 402))
        
        if self.state == "SCANNING":
            gt_x, gt_y, gt_z, gt_yaw = self.current_gt_pose
            txt_gt = self.font_small.render(f"Ground Truth Pose: [{gt_x:5.2f}, {gt_y:5.2f}, {gt_z:5.2f}, {gt_yaw:5.1f}°]", True, (241, 196, 15))
            self.hud_surface.blit(txt_gt, (650, 422))
            
            if self.cnn_pred_pose is not None:
                cx, cy, cz, cyaw = self.cnn_pred_pose
                txt_cnn = self.font_small.render(f"CNN Pose:          [{cx:5.2f}, {cy:5.2f}, {cz:5.2f}, {cyaw:5.1f}°]", True, (46, 204, 113))
            else:
                txt_cnn = self.font_small.render("CNN Pose:          (Not Loaded)", True, (150, 150, 150))
            self.hud_surface.blit(txt_cnn, (650, 440))
                
            if self.baseline_pred_pose is not None:
                bx, by, bz, byaw = self.baseline_pred_pose
                txt_base = self.font_small.render(f"Baseline Pose:     [{bx:5.2f}, {by:5.2f}, {bz:5.2f}, {byaw:5.1f}°]", True, (52, 152, 219))
            else:
                txt_base = self.font_small.render("Baseline Pose:     (No Markers)", True, (231, 76, 60))
            self.hud_surface.blit(txt_base, (650, 458))

            if self.hybrid_pred_pose is not None:
                hx, hy, hz, hyaw = self.hybrid_pred_pose
                err_xyz = math.sqrt((hx-gt_x)**2 + (hy-gt_y)**2 + (hz-gt_z)**2)
                err_yaw = abs(hyaw - gt_yaw)
                txt_hyb = self.font_small.render(f"Hybrid Pose:       [{hx:5.2f}, {hy:5.2f}, {hz:5.2f}, {hyaw:5.1f}°]", True, (241, 196, 15))
                self.hud_surface.blit(txt_hyb, (650, 476))
                
                txt_err = self.font_bold.render(f"Translation Error: {err_xyz:.3f}m | Yaw Error: {err_yaw:.1f}°", True, (255, 255, 255))
                self.hud_surface.blit(txt_err, (650, 496))
                
                mode_color = (52, 152, 219) if "Baseline" in self.active_estimation_mode else (46, 204, 113)
                txt_mode = self.font_bold.render(f"Active Estimator:  {self.active_estimation_mode}", True, mode_color)
                self.hud_surface.blit(txt_mode, (650, 516))
            
            m_ids = ", ".join(str(m["id"]) for m in self.visible_markers) if self.visible_markers else "None"
            txt_markers = self.font_small.render(f"Visible Marker IDs: {m_ids}", True, (180, 180, 180))
            self.hud_surface.blit(txt_markers, (650, 538))
            
            txt_lat = self.font_small.render(f"Estimator Processing Latency: {self.inference_time_ms:.1f} ms", True, (155, 89, 182))
            self.hud_surface.blit(txt_lat, (650, 554))
        else:
            txt_gt = self.font_bold.render("Drone in Transit / Hover Idle", True, (150, 150, 150))
            self.hud_surface.blit(txt_gt, (650, 440))
            txt_info = self.font_small.render("Camera and AI model processors are on standby.", True, (100, 100, 100))
            self.hud_surface.blit(txt_info, (650, 470))

        # --- LIVE INVENTORY DATA PROGRESS BAR PANEL ---
        pygame.draw.rect(self.hud_surface, (18, 18, 24, 210), (340, 495, 260, 110), border_radius=6)
        pygame.draw.rect(self.hud_surface, (58, 58, 68), (340, 495, 260, 110), 1, border_radius=6)
        lbl_counts_header = self.font_header.render("Warehouse Progress", True, (255, 255, 255))
        self.hud_surface.blit(lbl_counts_header, (350, 502))
        
        scanned_cnt = len(self.scanned_shelves)
        total_cnt = len(self.shelf_boxes)
        txt_progress = self.font_body.render(f"Shelves Scanned: {scanned_cnt} / {total_cnt}", True, (241, 196, 15))
        self.hud_surface.blit(txt_progress, (350, 526))
        
        # Mini progress bar
        pygame.draw.rect(self.hud_surface, (50, 50, 55), (350, 548, 240, 10), border_radius=2)
        prog_ratio = scanned_cnt / total_cnt if total_cnt > 0 else 0.0
        pygame.draw.rect(self.hud_surface, (46, 204, 113), (350, 548, int(240 * prog_ratio), 10), border_radius=2)
        
        txt_stats = self.font_small.render(f"Total Boxes: {self.total_boxes_detected} | Avg Occupancy: {self.average_occupancy*100:.1f}%", True, (230, 230, 235))
        self.hud_surface.blit(txt_stats, (350, 568))
        
        txt_percent = self.font_small.render(f"Scan Completion: {prog_ratio*100:.1f}%", True, (180, 180, 180))
        self.hud_surface.blit(txt_percent, (350, 584))

        # --- MINIMAP PANEL (Bottom Left) ---
        pygame.draw.rect(self.hud_surface, (18, 18, 24, 210), (30, 415, 164, 190), border_radius=6)
        pygame.draw.rect(self.hud_surface, (58, 58, 68), (30, 415, 164, 190), 1, border_radius=6)
        lbl_mini = self.font_small.render("MINIMAP (CLICK CELL)", True, (150, 150, 150))
        self.hud_surface.blit(lbl_mini, (38, 421))
        
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                cell_val = GRID[r][c]
                cell_rect = pygame.Rect(40 + c * 12, 442 + r * 12, 11, 11)
                
                if cell_val == 2:
                    pygame.draw.rect(self.hud_surface, (39, 174, 96), cell_rect)
                elif cell_val == 1:
                    shelf_pos = (r, c)
                    if shelf_pos in self.temp_obstacles:
                        pygame.draw.rect(self.hud_surface, (231, 76, 60), cell_rect) # Obstacle (Red)
                    else:
                        color = (46, 204, 113) if shelf_pos in self.scanned_shelves else (100, 100, 105)
                        pygame.draw.rect(self.hud_surface, color, cell_rect)
                else:
                    pygame.draw.rect(self.hud_surface, (40, 40, 45), cell_rect)
                    
        if len(self.path) > 1:
            points = []
            for idx in range(max(0, self.path_index - 1), len(self.path)):
                node = self.path[idx]
                points.append((40 + node[1] * 12 + 5, 442 + node[0] * 12 + 5))
            if len(points) > 1:
                pygame.draw.lines(self.hud_surface, (52, 152, 219), False, points, 2)
                
        # Draw drone dot on minimap
        drone_mini_x = 40 + self.drone_grid_x * 12 + 5
        drone_mini_y = 442 + self.drone_grid_y * 12 + 5
        pygame.draw.circle(self.hud_surface, (231, 76, 60), (drone_mini_x, drone_mini_y), 4)

        # Drag instruction
        lbl_drag = self.font_small.render("Drag left mouse on background to rotate camera | Scroll to Zoom", True, (130, 130, 140))
        self.hud_surface.blit(lbl_drag, (210, 580))

        # Draw final summary card if all shelves are scanned and we have scanned at least one shelf
        if len(self.scanned_shelves) == len(self.shelf_boxes) and len(self.shelf_boxes) > 0:
            # Translucent dark overlay over the center area
            card_rect = pygame.Rect(200, 150, 600, 400)
            pygame.draw.rect(self.hud_surface, (15, 15, 20, 245), card_rect, border_radius=12)
            pygame.draw.rect(self.hud_surface, (46, 204, 113), card_rect, 2, border_radius=12)
            
            # Title
            title_txt = self.font_title.render("WAREHOUSE SCAN MISSION COMPLETE", True, (46, 204, 113))
            self.hud_surface.blit(title_txt, (320, 180))
            
            # Draw decorative line
            pygame.draw.line(self.hud_surface, (58, 58, 68), (240, 225), (760, 225), 2)
            
            # Stats text list
            avg_boxes = self.total_boxes_detected / len(self.shelf_boxes) if len(self.shelf_boxes) > 0 else 0.0
            stats_list = [
                f"Total Shelves Scanned:   {len(self.scanned_shelves)} / {len(self.shelf_boxes)} (100.0%)",
                f"Total Boxes Detected:    {self.total_boxes_detected} boxes",
                f"Average Shelf Occupancy:  {self.average_occupancy * 100:.1f} %",
                f"Average Boxes per Shelf:  {avg_boxes:.2f} boxes",
                f"AI Localization Mode:     Hybrid Estimator (CNN + AdaBoost)",
                f"Peak Model Inference:     {self.inference_time_ms:.1f} ms / frame",
                f"Active Autopilot Status:  Docked & Fully Audited"
            ]
            
            for i, stat in enumerate(stats_list):
                txt_surf = self.font_header.render(stat, True, (255, 255, 255))
                self.hud_surface.blit(txt_surf, (260, 250 + i * 35))
                
            # Footer hint
            footer_txt = self.font_body.render("Press 'Start Sweep Scan' to run audit again | ESC to Exit", True, (150, 150, 150))
            self.hud_surface.blit(footer_txt, (340, 510))

        # --- DRAW CONTROL BUTTONS ---
        for btn in self.buttons:
            text = btn["text"]
            if btn["id"] == "query":
                text = "Scan Nearest"
            elif btn["id"] == "mode":
                text = f"Mode: {self.mode}"
                btn["color"] = (39, 174, 96) if self.mode == "AUTO" else (231, 76, 60)
            elif btn["id"] == "speed":
                text = f"Speed: {self.simulation_speed}x"
            
            m_pos = pygame.mouse.get_pos()
            btn_color = btn["color"]
            if btn["rect"].collidepoint(m_pos):
                btn_color = tuple(min(255, c + 35) for c in btn["color"])
                
            pygame.draw.rect(self.hud_surface, btn_color, btn["rect"], border_radius=6)
            pygame.draw.rect(self.hud_surface, (255, 255, 255), btn["rect"], 1, border_radius=6)
            
            btn_txt = self.font_bold.render(text, True, (255, 255, 255))
            text_rect = btn_txt.get_rect(center=btn["rect"].center)
            self.hud_surface.blit(btn_txt, text_rect)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.USEREVENT + 1:
                    # Triggered only once in demo mode
                    pygame.time.set_timer(pygame.USEREVENT + 1, 0) # stop timer
                    self.start_sweep()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    m_pos = pygame.mouse.get_pos()
                    
                    btn_clicked = False
                    for btn in self.buttons:
                        if btn["rect"].collidepoint(m_pos):
                            btn_clicked = True
                            if btn["id"] == "sweep":
                                if self.mode == "AUTO":
                                    self.start_sweep()
                            elif btn["id"] == "dock":
                                if self.mode == "AUTO":
                                    self.return_to_charger()
                            elif btn["id"] == "query":
                                if self.mode == "AUTO":
                                    self.query_nearest_item()
                            elif btn["id"] == "mode":
                                self.mode = "MANUAL" if self.mode == "AUTO" else "AUTO"
                                self.path = []
                                self.state = "IDLE"
                            elif btn["id"] == "speed":
                                if self.simulation_speed == 1:
                                    self.simulation_speed = 2
                                elif self.simulation_speed == 2:
                                    self.simulation_speed = 4
                                else:
                                    self.simulation_speed = 1
                    
                    if not btn_clicked:
                        mx, my = m_pos
                        # Minimap coordinates: (40, 442) to (40+144, 442+144)
                        if 40 <= mx < 184 and 442 <= my < 586:
                            col = (mx - 40) // 12
                            row = (my - 442) // 12
                            if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
                                if GRID[row][col] == 0 and (row, col) != (self.drone_grid_y, self.drone_grid_x):
                                    GRID[row][col] = 1
                                    self.temp_obstacles.add((row, col))
                                    print(f"Placed dynamic obstacle at ({row}, {col})")
                                    self.trigger_replan()
                                elif (row, col) in self.temp_obstacles:
                                    GRID[row][col] = 0
                                    self.temp_obstacles.remove((row, col))
                                    print(f"Removed dynamic obstacle at ({row}, {col})")
                                    self.trigger_replan()
                        else:
                            self.is_dragging = True
                            self.last_mouse_pos = m_pos
                            
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.is_dragging = False
                        
                elif event.type == pygame.MOUSEMOTION:
                    if self.is_dragging:
                        m_pos = pygame.mouse.get_pos()
                        dx = m_pos[0] - self.last_mouse_pos[0]
                        dy = m_pos[1] - self.last_mouse_pos[1]
                        
                        self.camera_yaw += dx * 0.5
                        self.camera_pitch = max(5.0, min(85.0, self.camera_pitch - dy * 0.5))
                        self.last_mouse_pos = m_pos
                        
                elif event.type == pygame.KEYDOWN:
                    if self.mode == "MANUAL":
                        dy, dx = 0, 0
                        if event.key in [pygame.K_UP, pygame.K_w]:
                            dy = -1
                        elif event.key in [pygame.K_DOWN, pygame.K_s]:
                            dy = 1
                        elif event.key in [pygame.K_LEFT, pygame.K_a]:
                            dx = -1
                        elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                            dx = 1
                        elif event.key == pygame.K_SPACE:
                            if self.state == "IDLE":
                                unscanned = self.get_adjacent_shelves(self.drone_grid_y, self.drone_grid_x)
                                if unscanned:
                                    self.trigger_scan(unscanned[0])
                                else:
                                    print("No adjacent unscanned shelves.")
                                    
                        if dx != 0 or dy != 0:
                            if self.state == "IDLE":
                                nr = self.drone_grid_y + dy
                                nc = self.drone_grid_x + dx
                                if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and GRID[nr][nc] != 1:
                                    self.path = [(nr, nc)]
                                    self.path_index = 0
                                    self.state = "NAVIGATING"
                                    
                elif event.type == pygame.MOUSEWHEEL:
                    self.camera_dist = max(100.0, min(1000.0, self.camera_dist - event.y * 20.0))
                                    
            self.update()
            self.draw()
            self.clock.tick(60)
            
        # Capture final summary screenshot
        if self.record_mode or self.demo_mode:
            self.capture_gl_screenshot("final_inventory_summary.png")
            
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warehouse Drone Simulation")
    parser.add_argument("--demo", action="store_true", help="Run in cinematic demo mode")
    args, unknown = parser.parse_known_args()
    
    app = WarehouseDroneApp(demo_mode=args.demo)
    app.run()
