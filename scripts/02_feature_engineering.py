"""
Feature Engineering Script
===========================
Run the complete feature engineering pipeline:
1. Sessionize events
2. Extract features
3. Create sequences
4. Split data

Usage:
    python 02_feature_engineering.py              # Default: no normalization
    python 02_feature_engineering.py --normalize  # With Min-Max normalization
"""

import argparse
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data.feature_engineering import main as run_feature_engineering
from data.sequence_creation import main as run_sequence_creation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature Engineering for Insider Threat Detection"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Apply Min-Max normalization to features (default: False)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("INSIDER THREAT DETECTION - FEATURE ENGINEERING")
    print("=" * 70)
    print(f"  Normalization: {'ENABLED' if args.normalize else 'DISABLED'}")
    
    # Step 1: Feature Engineering
    print("\n[STEP 1/2] Sessionizing events and extracting features...")
    run_feature_engineering(normalize=args.normalize)
    
    # Step 2: Sequence Creation
    print("\n[STEP 2/2] Creating LSTM sequences and splitting data...")
    run_sequence_creation()
    
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE!")
    print("=" * 70)
    print("\nNext step: Run 03_train.py to train the model")


if __name__ == "__main__":
    main()
