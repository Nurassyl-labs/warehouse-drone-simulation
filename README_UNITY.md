# Unity Robotics Warehouse Integration and ML Pipeline Guide

This guide explains how to set up, open, and run the Unity warehouse environment integration, generate synthetic datasets, use the local HTTP inference bridge, and connect the Unity-generated data to the Python machine learning pipeline in `final_project_2.0`.

---

## 1. Unity Setup & Project Specifications

### System Requirements
* **Unity Version**: **Unity 2021.3 LTS** (or newer LTS releases like 2022.3 LTS).
* **Render Pipeline**: **Universal Render Pipeline (URP)** for optimized performance on normal laptops and Macbooks.
* **Packages Needed**: `Unity UI`, `Universal RP`.

### How to Open the Project
1. Download and open **Unity Hub**.
2. Click **Add** -> **Add project from disk**.
3. Select the folder: `final_project_2.0/Robotics-Warehouse/`.
4. If Unity Hub prompts to download/install the corresponding editor version, select **Unity 2021.3 LTS**.
5. Once opened, switch the render pipeline to **URP** if prompted, or verify that the active graphics settings use the lightweight URP assets.

---

## 2. Scene Configuration & C# Scripts

We have provided three core C# scripts in the folder `Assets/Scripts/`:

### A. Stage 1: Automatic Drone Mission (`DroneMissionController.cs`)
This script implements a waypoint-based flight route simulating an autonomous inventory scan:
1. **Setup**:
   - Create a 3D Drone GameObject (with body and 4 separate propeller children).
   - Attach the `DroneMissionController` component to the drone.
   - Reference the drone's body transform and its propeller transforms in the inspector fields.
   - Add a `LineRenderer` to the drone representing the laser scanner and drag it to the `scanLaser` field.
2. **Execution**:
   - When the scene starts, the drone takes off from its dock, flies through the warehouse aisles, sweeps a red laser over the shelves to represent scanning, and returns/lands back on the charging dock.

### B. Stage 2: C# Dataset Capture (`DatasetCapture.cs`)
This script automates synthetic image data collection inside Unity:
1. **Setup**:
   - Attach the `DatasetCapture` component to the Drone Front Camera GameObject.
   - Link the shelf marker transforms (with their corresponding IDs `10, 11, 12, 13`) in the `shelfMarkers` array list.
   - Link all cardboard box GameObjects inside the `boxGameObjects` array.
2. **Execution**:
   - Press the **'G'** key during Unity Play Mode.
   - The script will instantly loop, randomizing the camera coordinates (`x`, `y`, `z`, `yaw`), minor light intensity variation, and box active status (occupancy).
   - It captures `640x480` camera frames, encodes them to PNG, and saves them to `final_project/unity_dataset/raw/`.
   - It appends labels to `final_project/unity_dataset/labels.csv`.

### C. Stage 4: HTTP Inference Bridge HUD (`InferenceHUD.cs`)
This script connects the active Unity simulation directly to the Python neural network:
1. **Setup**:
   - Attach `InferenceHUD` to the Drone camera.
   - Link the `droneCamera` inspector field.
2. **Execution**:
   - Start the Python inference server (see below).
   - Play the scene. The script captures the camera frame every 200ms (5 FPS), posts it to `http://localhost:8080/predict`, receives predicted coordinates + box counts, and renders a translucent HUD on the top-left screen corner comparison.

---

## 3. Running the Python Inference Server

To feed the active Unity camera stream into the Python hybrid pose estimator and inventory counter:

1. In your terminal, run the inference server script:
   ```bash
   python final_project_2.0/inference_server.py
   ```
2. The server will launch on port `8080` and load the trained models.
3. Start the Unity scene. The `InferenceHUD` script will start streaming and displaying predictions inside the Unity HUD.

---

## 4. Connecting Unity Datasets to Python Training (Stage 3)

### Step 1: Preprocess/Post-process the Unity CSV
Since Unity C# scripts record raw labels (`x, y, z, yaw, visible_marker_ids, box_count, shelf_occupancy`) without the pixel coordinates and areas of individual ArUco markers, we run a python script to detect markers and enrich the dataset to match the OpenCV CSV format:
```bash
python final_project_2.0/postprocess_unity.py
```
This script will:
1. Read `final_project/unity_dataset/labels.csv`.
2. Extract the ArUco marker centers/areas from images in `unity_dataset/raw/` using Python/OpenCV.
3. Append columns (`m10_visible`, `m10_cx`, etc.) required by classical baseline regressions.
4. Perform deterministic dataset splits (`train.csv`, `val.csv`, `test.csv`).

### Step 2: Train Models on Unity Dataset
All Python training scripts have been updated to support a `--dataset_source` argument:
* **Train CNN Pose Regressor on Unity data**:
  ```bash
  python final_project_2.0/train_cnn.py --dataset_source unity
  ```
* **Train Baseline Regressor on Unity data**:
  ```bash
  python final_project_2.0/train_baseline.py --dataset_source unity
  ```
* **Run full pipeline verification on Unity data**:
  ```bash
  python final_project_2.0/run_all.py --full --dataset_source unity
  ```

---

## 5. Exporting Visual Demos

1. **Simulated Unity Demo Video**:
   To generate a visual demonstration video and output files without running the Unity Editor locally, run:
   ```bash
   python final_project_2.0/record_unity_demo.py
   ```
   This will output:
   * `final_project/results/demo_outputs/unity_demo_video.mp4` (automated scanning mission video).
   * `unity_camera_samples/` (4 raw frame images of different drone camera perspectives).
   * `unity_dataset_sample_grid.png` (2x2 stitched preview grid showing the drone view coordinates and boxes).
