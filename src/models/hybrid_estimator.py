import os
import sys
import numpy as np
import torch
import joblib
from PIL import Image

class HybridPoseEstimator:
    """
    Combines classical ML marker baseline with deep CNN pose regression.
    Uses classical baseline if ArUco markers are detected, otherwise falls back to deep CNN.
    """
    def __init__(self, models_dir="final_project/models", device="cpu"):
        self.device = torch.device(device)
        
        # Load baseline model and scaler
        self.baseline_model = None
        self.baseline_scaler = None
        baseline_path = os.path.join(models_dir, "best_baseline_regressor.joblib")
        baseline_scaler_path = os.path.join(models_dir, "baseline_scaler.joblib")
        
        if os.path.exists(baseline_path) and os.path.exists(baseline_scaler_path):
            self.baseline_model = joblib.load(baseline_path)
            self.baseline_scaler = joblib.load(baseline_scaler_path)
            
        # Load CNN model and scaler
        self.cnn_model = None
        self.pose_scaler = None
        cnn_path = os.path.join(models_dir, "best_pose_model.pth")
        pose_scaler_path = os.path.join(models_dir, "pose_scaler.joblib")
        
        if os.path.exists(cnn_path) and os.path.exists(pose_scaler_path):
            # Dynamic import to avoid circular dependency
            from src.models.pose_cnn import PoseRegressorCNN
            self.cnn_model = PoseRegressorCNN().to(self.device)
            try:
                self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=self.device))
            except RuntimeError:
                # If weights shape differs, catch error
                pass
            self.cnn_model.eval()
            self.pose_scaler = joblib.load(pose_scaler_path)
            
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def predict(self, img_bgr, visible_markers):
        """
        Predicts drone pose (x, y, z, yaw) from BGR camera image and list of visible markers.
        """
        # Check if markers ID in [10, 11, 12, 13] is detected
        marker_ids = [m["id"] for m in visible_markers]
        valid_marker_visible = any(mid in [10, 11, 12, 13] for mid in marker_ids)
        
        if valid_marker_visible and self.baseline_model is not None and self.baseline_scaler is not None:
            # Build 16-dimensional feature vector
            marker_features = {
                10: {"visible": 0.0, "cx": 0.0, "cy": 0.0, "area": 0.0},
                11: {"visible": 0.0, "cx": 0.0, "cy": 0.0, "area": 0.0},
                12: {"visible": 0.0, "cx": 0.0, "cy": 0.0, "area": 0.0},
                13: {"visible": 0.0, "cx": 0.0, "cy": 0.0, "area": 0.0}
            }
            for m in visible_markers:
                m_id = m["id"]
                if m_id in marker_features:
                    marker_features[m_id] = {
                        "visible": 1.0,
                        "cx": m["cx"],
                        "cy": m["cy"],
                        "area": m["area"]
                    }
            
            feats = []
            for mid in [10, 11, 12, 13]:
                feats.extend([
                    marker_features[mid]["visible"],
                    marker_features[mid]["cx"],
                    marker_features[mid]["cy"],
                    marker_features[mid]["area"]
                ])
                
            X = np.array([feats], dtype=np.float32)
            X_scaled = self.baseline_scaler.transform(X)
            pred = self.baseline_model.predict(X_scaled)[0]
            return pred, "Baseline marker mode"
            
        elif self.cnn_model is not None and self.pose_scaler is not None:
            # Fallback to CNN
            import cv2
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.cnn_model(tensor).cpu().numpy()
                
            pred = self.pose_scaler.inverse_transform(outputs)[0]
            return pred, "CNN fallback mode"
            
        else:
            return np.array([0.0, 0.0, 0.0, 0.0]), "No models loaded"
