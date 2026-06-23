import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseRegressorCNN(nn.Module):
    """
    CNN architecture designed for continuous pose regression (predicts scaled x, y, z, yaw).
    Input image size: 3x128x128.
    """
    def __init__(self):
        super(PoseRegressorCNN, self).__init__()
        # Conv Block 1: 3x128x128 -> 16x64x64
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Conv Block 2: 16x64x64 -> 32x32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Conv Block 3: 32x32x32 -> 64x16x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Conv Block 4: 64x16x16 -> 128x8x8
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        
        # Linear Layers
        # 128 channels * 8 * 8 spatial dimension = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 4) # Output is scaled x, y, z, yaw

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
