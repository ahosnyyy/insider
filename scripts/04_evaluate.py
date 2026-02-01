"""
Evaluation Script
==================
Evaluate the trained model and generate reports.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from evaluation.evaluator import main as run_evaluation


def main():
    print("=" * 70)
    print("INSIDER THREAT DETECTION - MODEL EVALUATION")
    print("=" * 70)
    
    run_evaluation()
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
