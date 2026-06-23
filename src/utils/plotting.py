import os
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(history, output_path):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss (MSE)", color="blue")
    plt.plot(history["val_loss"], label="Val Loss (MSE)", color="orange")
    plt.title("CNN Pose Regressor Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_predicted_vs_true(y_true, y_pred, dim_name, output_path):
    """Plots a predicted vs true scatter plot for a single coordinate dimension."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color="teal")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.title(f"CNN Pose: Predicted vs True ({dim_name.upper()})")
    plt.xlabel(f"True {dim_name}")
    plt.ylabel(f"Predicted {dim_name}")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_trajectory_comparison(y_true, y_pred, output_path, max_samples=30):
    """Plots actual vs predicted drone trajectories in 2D space (X vs Z coordinates)."""
    plt.figure(figsize=(8, 8))
    num_plot = min(max_samples, len(y_true))
    plt.plot(y_true[:num_plot, 0], y_true[:num_plot, 2], 'go-', label="True Trajectory (X vs Z)")
    plt.plot(y_pred[:num_plot, 0], y_pred[:num_plot, 2], 'ro--', label="CNN Predicted Trajectory")
    for j in range(num_plot):
        plt.plot([y_true[j, 0], y_pred[j, 0]], [y_true[j, 2], y_pred[j, 2]], 'k-', alpha=0.3)
    plt.title(f"CNN Drone Trajectory Comparison (X vs Z, First {num_plot} Test Samples)")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Z Position (meters)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
