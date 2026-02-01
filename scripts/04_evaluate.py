"""
Evaluation Script
==================
Evaluate the trained model and generate reports.

Usage:
    python 04_evaluate.py                              # Default: both perspectives
    python 04_evaluate.py --positive-class insider     # Insider as positive class only
    python 04_evaluate.py --exclude-scenarios 3        # Exclude scenario 3
    python 04_evaluate.py --exclude-scenarios 2,3      # Exclude multiple scenarios
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
    parser.add_argument(
        "--exclude-scenarios",
        type=str,
        default=None,
        help="Comma-separated list of scenarios to exclude (e.g., '3' or '2,3')"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse excluded scenarios
    exclude_scenarios = None
    if args.exclude_scenarios:
        exclude_scenarios = [int(s.strip()) for s in args.exclude_scenarios.split(',')]
    
    print("=" * 70)
    print("INSIDER THREAT DETECTION - MODEL EVALUATION")
    print("=" * 70)
    print(f"  Positive class: {args.positive_class}")
    if exclude_scenarios:
        print(f"  Excluding scenarios: {exclude_scenarios}")
    
    run_evaluation(
        positive_class=args.positive_class,
        exclude_scenarios=exclude_scenarios
    )
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
