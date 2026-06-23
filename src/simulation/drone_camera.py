import numpy as np
from src.utils.config import CONFIG

class DroneCameraModel:
    """
    Represents the drone camera properties and configuration.
    """
    def __init__(self):
        self.focal_length = CONFIG["camera"]["focal_length"]
        self.image_size = tuple(CONFIG["dataset"]["image_size"])
        self.w, self.h = self.image_size
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0
        
        # Intrinsics matrix K
        self.K = np.array([[self.focal_length, 0, self.cx],
                           [0, self.focal_length, self.cy],
                           [0, 0, 1]], dtype=np.float32)

    def get_intrinsics(self):
        return self.K
