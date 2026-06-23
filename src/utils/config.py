import os
import yaml

# Hardcoded defaults for resilience
DEFAULT_CONFIG = {
    "dataset": {
        "num_samples": 5000,
        "quick_samples": 500,
        "split_train": 0.70,
        "split_val": 0.15,
        "split_test": 0.15,
        "seed": 42,
        "image_size": [640, 480]
    },
    "camera": {
        "focal_length": 500.0
    },
    "shelf": {
        "z": 3.0,
        "levels": [-0.8, 0.0, 0.8],
        "columns": [-0.9, 0.0, 0.9],
        "markers": [
            {"id": 10, "x": -1.4, "y": -0.9},
            {"id": 11, "x": 1.4, "y": -0.9},
            {"id": 12, "x": -1.4, "y": 0.9},
            {"id": 13, "x": 1.4, "y": 0.9}
        ]
    }
}

def load_config():
    """Loads configuration dictionary from config.yaml file"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, "config.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                if cfg:
                    # Merge defaults for any missing sections
                    for k, v in DEFAULT_CONFIG.items():
                        if k not in cfg:
                            cfg[k] = v
                    return cfg
        except Exception:
            pass
    return DEFAULT_CONFIG

CONFIG = load_config()
