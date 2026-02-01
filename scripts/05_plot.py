"""
Plotting Script
================
Generate visualization plots for analysis.

Plots:
1. Reconstruction error scatter plot (like paper's figure)
2. Training/Validation loss curve

Usage:
    python 05_plot.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def plot_reconstruction_error_scatter(
    errors: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    output_path: Path
):
    """
    Create scatter plot of reconstruction error by data point index.
    Similar to Figure 9 in the paper.
    """
    plt.figure(figsize=(12, 6))
    
    # Get indices for each class
    normal_idx = np.where(y_test == 0)[0]
    insider_idx = np.where(y_test == 1)[0]
    
    # Plot normal samples first (blue)
    plt.scatter(
        normal_idx, errors[normal_idx],
        c='blue', alpha=0.5, s=10, label='Normal'
    )
    
    # Plot insider samples on top (orange)
    plt.scatter(
        insider_idx, errors[insider_idx],
        c='orange', alpha=0.7, s=10, label='Insider'
    )
    
    # Add threshold line
    plt.axhline(y=threshold, color='red', linestyle='-', linewidth=2, label=f'Threshold: {threshold:.4f}')
    
    plt.xlabel('Data point index')
    plt.ylabel('Reconstruction error')
    plt.title('Reconstruction error for different classes')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_loss_curve(
    history: dict,
    output_path: Path
):
    """
    Plot training and validation loss curves.
    """
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    
    if not train_loss or not val_loss:
        print("  No training history found!")
        return
    
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    
    # Mark best epoch
    best_epoch = np.argmin(val_loss) + 1
    best_val = min(val_loss)
    plt.scatter([best_epoch], [best_val], c='green', s=100, zorder=5, label=f'Best: Epoch {best_epoch}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  Saved: {output_path}")


def main():
    import torch
    import yaml
    from models.lstm_autoencoder import create_model
    
    print("=" * 70)
    print("INSIDER THREAT DETECTION - PLOTTING")
    print("=" * 70)
    
    # Paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    outputs_dir = project_root / "outputs"
    plots_dir = outputs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    processed_dir = project_root / config['data']['processed_dir']
    
    # ===== PLOT 1: Loss Curve =====
    print("\n[1/2] Generating loss curve...")
    history_path = outputs_dir / "training_history.npy"
    
    if history_path.exists():
        history = np.load(history_path, allow_pickle=True).item()
        plot_loss_curve(history, plots_dir / "loss_curve.png")
    else:
        print("  Warning: training_history.npy not found!")
    
    # ===== PLOT 2: Reconstruction Error Scatter =====
    print("\n[2/2] Generating reconstruction error scatter plot...")
    
    # Load test data
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Insider samples: {y_test.sum():,}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = outputs_dir / "models" / "checkpoint_best.pt"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = create_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"  Loaded model from epoch {checkpoint['epoch']}")
        
        # Calculate reconstruction errors
        print("  Calculating reconstruction errors...")
        errors = []
        batch_size = 256
        
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
                reconstructed = model(batch)
                batch_errors = torch.mean((batch - reconstructed) ** 2, dim=(1, 2))
                errors.extend(batch_errors.cpu().numpy())
        
        errors = np.array(errors)
        
        # Load threshold from evaluation metrics if available
        eval_metrics_path = outputs_dir / "evaluation" / "insider_positive" / "metrics.npy"
        if eval_metrics_path.exists():
            metrics = np.load(eval_metrics_path, allow_pickle=True).item()
            threshold = metrics.get('threshold', np.percentile(errors, 90))
        else:
            # Use 90th percentile as default threshold
            threshold = np.percentile(errors, 90)
        
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Mean error (normal): {errors[y_test == 0].mean():.4f}")
        print(f"  Mean error (insider): {errors[y_test == 1].mean():.4f}")
        
        # Create scatter plot
        plot_reconstruction_error_scatter(
            errors, y_test, threshold,
            plots_dir / "reconstruction_error_scatter.png"
        )
    else:
        print("  Warning: No model checkpoint found!")
    
    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE!")
    print("=" * 70)
    print(f"\nPlots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
