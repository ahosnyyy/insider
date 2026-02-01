"""
Training Script
================
Train the LSTM-Autoencoder model.

Usage:
    python 03_train.py              # Training (uses data from previous step)
    python 03_train.py --normalize  # Log that normalized data was used
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
        "--normalize",
        action="store_true",
        default=False,
        help="Log that normalized features are being used (for tracking)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("INSIDER THREAT DETECTION - MODEL TRAINING")
    print("=" * 70)
    print(f"  Data preprocessing: {'Normalized' if args.normalize else 'Raw values'}")
    
    run_training()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nNext step: Run 04_evaluate.py to evaluate the model")


if __name__ == "__main__":
    main()
