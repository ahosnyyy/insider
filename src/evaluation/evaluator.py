"""
Evaluation Module
==================
Evaluate trained model and compute metrics.

Key operations:
1. Load trained model
2. Calculate reconstruction error on test set
3. Find F1-maximizing threshold
4. Generate confusion matrix and metrics
5. Create visualizations
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, auc, precision_recall_curve
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.lstm_autoencoder import LSTMAutoencoder, create_model


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config: dict, device: torch.device) -> LSTMAutoencoder:
    """Load trained model from checkpoint."""
    project_root = Path(__file__).parent.parent.parent
    checkpoint_path = project_root / "outputs" / "models" / "checkpoint_best.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint['val_loss']:.6f}")
    
    return model


def load_test_data(config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data."""
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / config['data']['processed_dir']
    
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    
    return X_test, y_test


def calculate_reconstruction_errors(
    model: LSTMAutoencoder,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256
) -> np.ndarray:
    """Calculate reconstruction error for each sample."""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
            error = model.get_reconstruction_error(batch)
            errors.append(error.cpu().numpy())
    
    return np.concatenate(errors)


def find_optimal_threshold(
    errors: np.ndarray,
    y_true: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    """
    Find threshold that maximizes F1 score.
    
    Returns:
        - Optimal threshold
        - Dictionary of metrics at that threshold
    """
    # Try different thresholds
    thresholds = np.percentile(errors, np.linspace(80, 99.9, 100))
    
    best_f1 = 0
    best_threshold = thresholds[0]
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (errors > threshold).astype(int)
        
        if y_pred.sum() == 0:  # No predictions
            continue
            
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1
            }
    
    return best_threshold, best_metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    errors: np.ndarray
) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle edge cases
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }
    
    # AUC
    fpr_curve, tpr_curve, _ = roc_curve(y_true, errors)
    metrics['auc'] = auc(fpr_curve, tpr_curve)
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Insider'],
        yticklabels=['Normal', 'Insider']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_error_distribution(
    errors: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    output_path: Path
):
    """Plot reconstruction error distribution."""
    plt.figure(figsize=(10, 6))
    
    # Separate by class
    normal_errors = errors[y_true == 0]
    insider_errors = errors[y_true == 1]
    
    plt.hist(normal_errors, bins=100, alpha=0.7, label='Normal', density=True)
    plt.hist(insider_errors, bins=100, alpha=0.7, label='Insider', density=True)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    errors: np.ndarray,
    output_path: Path
):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, errors)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_comparison_with_paper(metrics: Dict[str, float]):
    """Print comparison with paper's results."""
    print("\n" + "=" * 60)
    print("COMPARISON WITH PAPER")
    print("=" * 60)
    print("\n{:<20} {:>15} {:>15}".format("Metric", "Ours", "Paper (CM)"))
    print("-" * 50)
    print("{:<20} {:>15.2f}% {:>15.2f}%".format("Accuracy", metrics['accuracy']*100, 90.62))
    print("{:<20} {:>15.2f}% {:>15.2f}%".format("Precision", metrics['precision']*100, 11.27))
    print("{:<20} {:>15.2f}% {:>15.2f}%".format("Recall", metrics['recall']*100, 3.50))
    print("{:<20} {:>15.2f}% {:>15.2f}%".format("F1 Score", metrics['f1']*100, 5.34))
    print("{:<20} {:>15.2f}% {:>15.2f}%".format("FPR", metrics['fpr']*100, 2.25))
    print("\n{:<20}".format("Confusion Matrix (Ours):"))
    print("  TN: {:,}  FP: {:,}".format(metrics['tn'], metrics['fp']))
    print("  FN: {:,}  TP: {:,}".format(metrics['fn'], metrics['tp']))
    print("\n{:<20}".format("Confusion Matrix (Paper):"))
    print("  TN: 180,732  FP: 4,163")
    print("  FN: 14,572   TP: 529")


def main():
    """Main evaluation entry point."""
    config = load_config()
    
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(config, device)
    
    # Load test data
    print("\nLoading test data...")
    X_test, y_test = load_test_data(config)
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Insider samples: {y_test.sum():,}")
    print(f"  Normal samples: {(y_test == 0).sum():,}")
    
    # Calculate reconstruction errors
    print("\nCalculating reconstruction errors...")
    errors = calculate_reconstruction_errors(model, X_test, device)
    print(f"  Mean error (normal): {errors[y_test == 0].mean():.6f}")
    print(f"  Mean error (insider): {errors[y_test == 1].mean():.6f}")
    
    # Find optimal threshold
    print("\nFinding optimal threshold...")
    threshold, threshold_metrics = find_optimal_threshold(errors, y_test)
    print(f"  Optimal threshold: {threshold:.6f}")
    print(f"  F1 at threshold: {threshold_metrics['f1']:.4f}")
    
    # Compute final metrics
    y_pred = (errors > threshold).astype(int)
    metrics = compute_metrics(y_test, y_pred, errors)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1 Score:  {metrics['f1']*100:.2f}%")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  FPR:       {metrics['fpr']*100:.2f}%")
    
    # Create plots
    project_root = Path(__file__).parent.parent.parent
    plots_dir = project_root / "outputs" / "plots"
    log_dir = project_root / "outputs" / "logs"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating plots...")
    plot_confusion_matrix(y_test, y_pred, plots_dir / "confusion_matrix.png")
    plot_error_distribution(errors, y_test, threshold, plots_dir / "error_distribution.png")
    plot_roc_curve(y_test, errors, plots_dir / "roc_curve.png")
    print(f"  Plots saved to: {plots_dir}")
    
    # Log all metrics to TensorBoard
    print("\nLogging metrics to TensorBoard...")
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir / "evaluation")
    
    # Log scalar metrics
    writer.add_scalar('Eval/Accuracy', metrics['accuracy'], 0)
    writer.add_scalar('Eval/Precision', metrics['precision'], 0)
    writer.add_scalar('Eval/Recall', metrics['recall'], 0)
    writer.add_scalar('Eval/F1', metrics['f1'], 0)
    writer.add_scalar('Eval/FPR', metrics['fpr'], 0)
    writer.add_scalar('Eval/AUC', metrics['auc'], 0)
    writer.add_scalar('Eval/Threshold', threshold, 0)
    
    # Log confusion matrix values
    writer.add_scalar('ConfusionMatrix/TN', metrics['tn'], 0)
    writer.add_scalar('ConfusionMatrix/FP', metrics['fp'], 0)
    writer.add_scalar('ConfusionMatrix/FN', metrics['fn'], 0)
    writer.add_scalar('ConfusionMatrix/TP', metrics['tp'], 0)
    
    # Log reconstruction error stats
    writer.add_scalar('ReconError/mean_normal', errors[y_test == 0].mean(), 0)
    writer.add_scalar('ReconError/mean_insider', errors[y_test == 1].mean(), 0)
    writer.add_scalar('ReconError/std_normal', errors[y_test == 0].std(), 0)
    writer.add_scalar('ReconError/std_insider', errors[y_test == 1].std(), 0)
    
    # Add images of plots to TensorBoard
    from PIL import Image
    import torchvision.transforms as transforms
    
    for plot_name in ['confusion_matrix.png', 'error_distribution.png', 'roc_curve.png']:
        plot_path = plots_dir / plot_name
        if plot_path.exists():
            img = Image.open(plot_path)
            img_tensor = transforms.ToTensor()(img)
            writer.add_image(f'Plots/{plot_name.replace(".png", "")}', img_tensor, 0)
    
    writer.close()
    print(f"  TensorBoard logs saved to: {log_dir}/evaluation")
    
    # Comparison with paper
    print_comparison_with_paper(metrics)
    
    # Save metrics
    metrics['threshold'] = threshold
    np.save(project_root / "outputs" / "evaluation_metrics.npy", metrics)
    print(f"\nMetrics saved to: outputs/evaluation_metrics.npy")


if __name__ == "__main__":
    main()
