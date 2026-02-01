"""
Training Script
================
Train the LSTM-Autoencoder model.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from training.trainer import main as run_training


def main():
    print("=" * 70)
    print("INSIDER THREAT DETECTION - MODEL TRAINING")
    print("=" * 70)
    
    run_training()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nNext step: Run 04_evaluate.py to evaluate the model")


if __name__ == "__main__":
    main()
