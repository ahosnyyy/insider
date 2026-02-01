"""
Training Script
================
Train the LSTM-Autoencoder model.

Usage:
    python 03_train.py                    # Default (uses whatever scaler was used in feature engineering)
    python 03_train.py --scaler standard  # Log that StandardScaler was used
"""

import argparse
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from training.trainer import main as run_training


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LSTM-Autoencoder for Insider Threat Detection"
    )
    parser.add_argument(
        "--scaler",
        type=str,
        choices=['none', 'minmax', 'standard'],
        default='standard',
        help="Log which scaler was used for features (for tracking)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("INSIDER THREAT DETECTION - MODEL TRAINING")
    print("=" * 70)
    print(f"  Feature scaler: {args.scaler}")
    
    run_training()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nNext step: Run 04_evaluate.py to evaluate the model")


if __name__ == "__main__":
    main()
