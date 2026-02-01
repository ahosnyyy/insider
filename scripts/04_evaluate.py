"""
Evaluation Script
==================
Evaluate the trained model and generate reports.

Usage:
    python 04_evaluate.py                        # Default: Insider as positive class
    python 04_evaluate.py --positive-class both  # Calculate both perspectives
"""

import argparse
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from evaluation.evaluator import main as run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LSTM-Autoencoder for Insider Threat Detection"
    )
    parser.add_argument(
        "--positive-class",
        type=str,
        choices=['insider', 'normal', 'both'],
        default='both',
        help="Which class to treat as positive for metrics (default: both)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("INSIDER THREAT DETECTION - MODEL EVALUATION")
    print("=" * 70)
    print(f"  Positive class: {args.positive_class}")
    
    run_evaluation(positive_class=args.positive_class)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
