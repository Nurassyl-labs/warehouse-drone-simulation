import os
import sys
import argparse
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set path to allow imports from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils.seed import set_seed
from src.utils.config import CONFIG
from src.models.pose_cnn import PoseRegressorCNN

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

def main():
    parser = argparse.ArgumentParser(description="Train CNN Pose Regressor")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--quick", action="store_true", help="Quick mode (2 epochs, 100 samples limit)")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/mps/auto)")
    parser.add_argument("--dataset_source", type=str, default="opencv", choices=["opencv", "unity"], help="Dataset source (opencv or unity)")
    args = parser.parse_args()

    # Set seed
    set_seed(CONFIG["dataset"]["seed"])

    if args.dataset_source == "unity":
        data_dir = "final_project/unity_dataset"
    else:
        data_dir = "final_project/data"
        
    models_dir = "final_project/models"
    metrics_dir = "final_project/results/metrics"
    figures_dir = "final_project/results/figures"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Device selection
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # 2. Load splits
    train_path = os.path.join(data_dir, "splits", "train.csv")
    val_path = os.path.join(data_dir, "splits", "val.csv")
    test_path = os.path.join(data_dir, "splits", "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        print("Error: Dataset splits not found. Please run generate_dataset.py first!")
        return

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Quick override
    epochs = args.epochs
    batch_size = args.batch_size
    if args.quick:
        epochs = 2
        batch_size = 16
        train_df = train_df.head(100)
        val_df = val_df.head(30)
        test_df = test_df.head(30)
        print(f"Quick Mode Overrides: training on {len(train_df)} samples for {epochs} epochs.")

    # 3. Fit target scaler on train set poses
    target_cols = ["x", "y", "z", "yaw"]
    train_poses = train_df[target_cols].values
    val_poses = val_df[target_cols].values
    test_poses = test_df[target_cols].values

    scaler = StandardScaler()
    train_poses_scaled = scaler.fit_transform(train_poses)
    val_poses_scaled = scaler.transform(val_poses)
    test_poses_scaled = scaler.transform(test_poses)

    # Save target scaler
    scaler_path = os.path.join(models_dir, "pose_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved pose target scaler to: {scaler_path}")

    # 4. Initialize datasets & loaders
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_dir = os.path.join(data_dir, "raw")
    train_dataset = WarehousePoseDataset(train_df, img_dir, train_poses_scaled, transform=transform_train)
    val_dataset = WarehousePoseDataset(val_df, img_dir, val_poses_scaled, transform=transform_val)
    test_dataset = WarehousePoseDataset(test_df, img_dir, test_poses_scaled, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5. Initialize model, optimizer, scheduler, criterion
    model = PoseRegressorCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training state variables
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    print("\nStarting CNN Pose Regressor training...")
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * images.size(0)

        epoch_val_loss = running_val_loss / len(val_dataset)
        scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Train MSE: {epoch_train_loss:.4f} | Val MSE: {epoch_val_loss:.4f}")

        # Best model checkpoint saving & Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(models_dir, "best_pose_model.pth"))
            print(f"  --> Saved new best model with Val MSE: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and not args.quick:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 6. Load best model for evaluation on Test Set
    best_model_path = os.path.join(models_dir, "best_pose_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()
    test_preds_scaled = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            test_preds_scaled.append(outputs.cpu().numpy())

    test_preds_scaled = np.vstack(test_preds_scaled)
    # Inverse transform to get true-scale predictions
    test_preds = scaler.inverse_transform(test_preds_scaled)

    # 7. Compute & Save Metrics
    metrics_summary = {}
    print("\nCNN Pose Regressor Test Set Evaluation:")
    for i, dim in enumerate(target_cols):
        mae = mean_absolute_error(test_poses[:, i], test_preds[:, i])
        rmse = np.sqrt(mean_squared_error(test_poses[:, i], test_preds[:, i]))
        r2 = r2_score(test_poses[:, i], test_preds[:, i])
        
        print(f"  {dim} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
        metrics_summary[f"test_MAE_{dim}"] = float(mae)
        metrics_summary[f"test_RMSE_{dim}"] = float(rmse)
        metrics_summary[f"test_R2_{dim}"] = float(r2)

    avg_mae = np.mean([metrics_summary[f"test_MAE_{d}"] for d in target_cols])
    avg_r2 = np.mean([metrics_summary[f"test_R2_{d}"] for d in target_cols])
    metrics_summary["test_avg_MAE"] = float(avg_mae)
    metrics_summary["test_avg_R2"] = float(avg_r2)

    with open(os.path.join(metrics_dir, "cnn_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved CNN metrics to: {os.path.join(metrics_dir, 'cnn_metrics.json')}")

    # 8. Save Figures
    # A. Training curves
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss (MSE)", color="blue")
    plt.plot(history["val_loss"], label="Val Loss (MSE)", color="orange")
    plt.title("CNN Pose Regressor Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    curves_path = os.path.join(figures_dir, "cnn_pose_curves.png")
    plt.savefig(curves_path)
    plt.close()
    print(f"Saved curves to: {curves_path}")

    # B. Predicted vs True for each dimension
    for i, dim in enumerate(target_cols):
        plt.figure(figsize=(6, 6))
        plt.scatter(test_poses[:, i], test_preds[:, i], alpha=0.5, color="teal")
        min_val = min(test_poses[:, i].min(), test_preds[:, i].min())
        max_val = max(test_poses[:, i].max(), test_preds[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        plt.title(f"CNN Pose: Predicted vs True ({dim.upper()})")
        plt.xlabel(f"True {dim}")
        plt.ylabel(f"Predicted {dim}")
        plt.grid(True)
        pred_vs_true_path = os.path.join(figures_dir, f"predicted_vs_true_{dim}.png")
        plt.savefig(pred_vs_true_path)
        plt.close()
        print(f"Saved predicted vs true plot for {dim} to: {pred_vs_true_path}")

    # C. Trajectory True vs Pred (X vs Z coordinates)
    plt.figure(figsize=(8, 8))
    # Select first 20 test samples for clarity
    num_plot = min(30, len(test_poses))
    plt.plot(test_poses[:num_plot, 0], test_poses[:num_plot, 2], 'go-', label="True Trajectory (X vs Z)")
    plt.plot(test_preds[:num_plot, 0], test_preds[:num_plot, 2], 'ro--', label="CNN Predicted Trajectory")
    for j in range(num_plot):
        plt.plot([test_poses[j, 0], test_preds[j, 0]], [test_poses[j, 2], test_preds[j, 2]], 'k-', alpha=0.3)
    plt.title("CNN Drone Trajectory Comparison (X vs Z, First 30 Test Samples)")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Z Position (meters)")
    plt.legend()
    plt.grid(True)
    traj_path = os.path.join(figures_dir, "trajectory_true_vs_pred.png")
    plt.savefig(traj_path)
    plt.close()
    print(f"Saved trajectory comparison to: {traj_path}")

if __name__ == "__main__":
    main()
