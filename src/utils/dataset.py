import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class WarehousePoseDataset(Dataset):
    """Custom PyTorch dataset to load warehouse images and pre-scaled pose labels"""
    def __init__(self, df, img_dir, targets_scaled, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.targets_scaled = targets_scaled
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_path"]
        img_path = os.path.join(self.img_dir, os.path.basename(img_name))
        image = Image.open(img_path).convert("RGB")

        # Pose target scaled
        target = self.targets_scaled[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(target, dtype=torch.float32)
