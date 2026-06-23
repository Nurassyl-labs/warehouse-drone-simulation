import os
import sys
import json
import cv2
import numpy as np
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler

# Allow imports from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.hybrid_estimator import HybridPoseEstimator
from src.vision.aruco_detector import detect_aruco_markers
from src.vision.inventory_counter import detect_and_count_boxes

class InferenceHandler(BaseHTTPRequestHandler):
    estimator = None

    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        # Return online status for browser checks
        self._set_headers(200)
        response = {
            "status": "online",
            "message": "Inference server is active. Please send POST requests containing raw image bytes to /predict."
        }
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def do_POST(self):
        if self.path == "/predict":
            try:
                # 1. Read content length and load raw image bytes
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)

                # 2. Decode image from bytes
                nparr = np.frombuffer(post_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": "Failed to decode image"}).encode("utf-8"))
                    return

                # 3. Detect markers
                markers = detect_aruco_markers(img)

                # 4. Predict pose (x, y, z, yaw)
                pose, mode = self.estimator.predict(img, markers)

                # 5. Predict inventory box count
                box_count, shelf_occupancy, _ = detect_and_count_boxes(img)

                # 6. Build response dictionary
                response = {
                    "x": float(pose[0]),
                    "y": float(pose[1]),
                    "z": float(pose[2]),
                    "yaw": float(pose[3]),
                    "box_count": int(box_count),
                    "shelf_occupancy": float(shelf_occupancy),
                    "inference_mode": mode
                }

                # 7. Send success response
                self._set_headers(200)
                self.wfile.write(json.dumps(response).encode("utf-8"))
                
                print(f"Processed frame: pose=({response['x']:.2f}, {response['y']:.2f}, {response['z']:.2f}, {response['yaw']:.1f}), boxes={response['box_count']}, mode={mode}")

            except Exception as e:
                print(f"Error handling request: {e}")
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

def run_server(port=8080):
    # Setup device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Loading ML models for Hybrid Pose Estimator...")
    # Initialize estimator inside final_project_2.0 relative to its files
    InferenceHandler.estimator = HybridPoseEstimator(models_dir="final_project/models", device=device)
    
    server_address = ("", port)
    httpd = HTTPServer(server_address, InferenceHandler)
    
    print(f"Python inference server running on port {port}...")
    print(f"Send POST requests containing raw image bytes to http://localhost:{port}/predict")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
