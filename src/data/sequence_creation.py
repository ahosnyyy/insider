"""
Sequence Creation and Data Splitting
=====================================
Create LSTM sequences from sessions and split into train/val/test sets.

Key operations:
1. Create overlapping sequences of 20 sessions per user
2. Split: train/val (normal only) + test (normal + ALL insider)
3. Oversample test set to match paper's CM distribution
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import pickle
import yaml
from tqdm import tqdm
from sklearn.utils import resample


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_processed_data(config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load processed features, labels, and user IDs."""
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / config['data']['processed_dir']
    
    X = np.load(processed_dir / "features.npy")
    y = np.load(processed_dir / "labels.npy")
    user_ids = np.load(processed_dir / "user_ids.npy")
    
    return X, y, user_ids


def create_sequences_per_user(
    X: np.ndarray,
    y: np.ndarray,
    user_ids: np.ndarray,
    lookback: int = 20,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create LSTM sequences from sessions, keeping sequences within same user.
    
    Args:
        X: Feature matrix (n_sessions, n_features)
        y: Labels (n_sessions,)
        user_ids: User ID for each session (n_sessions,)
        lookback: Sequence length (default 20)
        stride: Stride between sequences (default 1)
        
    Returns:
        X_seq: Sequences (n_sequences, lookback, n_features)
        y_seq: Labels for each sequence (n_sequences,)
        user_seq: User ID for each sequence (n_sequences,)
    """
    print("\n--- Creating LSTM sequences ---")
    
    sequences = []
    seq_labels = []
    seq_users = []
    
    unique_users = np.unique(user_ids)
    
    for user_id in tqdm(unique_users, desc="Processing users"):
        # Get sessions for this user
        user_mask = user_ids == user_id
        user_X = X[user_mask]
        user_y = y[user_mask]
        
        # Skip if not enough sessions
        if len(user_X) < lookback:
            continue
        
        # Create sequences with stride
        for i in range(0, len(user_X) - lookback + 1, stride):
            seq = user_X[i:i + lookback]
            sequences.append(seq)
            
            # Sequence is insider if ANY session in it is insider
            # (conservative: detect sequences containing suspicious activity)
            seq_label = 1 if user_y[i:i + lookback].max() > 0 else 0
            seq_labels.append(seq_label)
            seq_users.append(user_id)
    
    X_seq = np.array(sequences, dtype=np.float32)
    y_seq = np.array(seq_labels, dtype=np.int32)
    user_seq = np.array(seq_users, dtype=np.int32)
    
    print(f"  Total sequences: {len(X_seq):,}")
    print(f"  Sequence shape: {X_seq.shape}")
    print(f"  Insider sequences: {y_seq.sum():,}")
    print(f"  Normal sequences: {(y_seq == 0).sum():,}")
    
    return X_seq, y_seq, user_seq


def split_data(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    user_seq: np.ndarray,
    config: dict
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split data into train/val/test sets.
    
    Special handling:
    - Train and Val: NORMAL sequences only (for autoencoder training)
    - Test: 20% of normal sequences + ALL insider sequences
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing (X, y)
    """
    print("\n--- Splitting data ---")
    
    train_ratio = config['split']['train']
    val_ratio = config['split']['val']
    test_ratio = config['split']['test']
    
    # Separate normal and insider
    normal_mask = y_seq == 0
    insider_mask = y_seq == 1
    
    X_normal = X_seq[normal_mask]
    y_normal = y_seq[normal_mask]
    X_insider = X_seq[insider_mask]
    y_insider = y_seq[insider_mask]
    
    print(f"  Normal sequences: {len(X_normal):,}")
    print(f"  Insider sequences: {len(X_insider):,}")
    
    # Shuffle normal sequences
    np.random.seed(42)
    perm = np.random.permutation(len(X_normal))
    X_normal = X_normal[perm]
    y_normal = y_normal[perm]
    
    # Split normal sequences
    n_normal = len(X_normal)
    n_train = int(n_normal * train_ratio)
    n_val = int(n_normal * val_ratio)
    
    X_train = X_normal[:n_train]
    y_train = y_normal[:n_train]
    
    X_val = X_normal[n_train:n_train + n_val]
    y_val = y_normal[n_train:n_train + n_val]
    
    X_test_normal = X_normal[n_train + n_val:]
    y_test_normal = y_normal[n_train + n_val:]
    
    # Test set = remaining normal + ALL insider
    X_test = np.concatenate([X_test_normal, X_insider], axis=0)
    y_test = np.concatenate([y_test_normal, y_insider], axis=0)
    
    # Shuffle test set
    perm = np.random.permutation(len(X_test))
    X_test = X_test[perm]
    y_test = y_test[perm]
    
    print(f"\n  Split summary:")
    print(f"    Train: {len(X_train):,} (all normal)")
    print(f"    Val:   {len(X_val):,} (all normal)")
    print(f"    Test:  {len(X_test):,} ({(y_test == 0).sum():,} normal, {(y_test == 1).sum():,} insider)")
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


def oversample_test_set(
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oversample test set to match paper's CM distribution.
    
    Paper's test set:
    - Total: ~200K
    - Insider: ~15K (7.5%)
    - Normal: ~185K (92.5%)
    """
    print("\n--- Oversampling test set ---")
    
    insider_factor = config['oversampling']['insider_factor']
    normal_factor = config['oversampling']['normal_factor']
    
    normal_mask = y_test == 0
    insider_mask = y_test == 1
    
    X_normal = X_test[normal_mask]
    y_normal = y_test[normal_mask]
    X_insider = X_test[insider_mask]
    y_insider = y_test[insider_mask]
    
    print(f"  Before oversampling:")
    print(f"    Normal: {len(X_normal):,}")
    print(f"    Insider: {len(X_insider):,}")
    
    # Oversample normal
    n_normal_target = int(len(X_normal) * normal_factor)
    X_normal_over, y_normal_over = resample(
        X_normal, y_normal,
        n_samples=n_normal_target,
        random_state=42,
        replace=True
    )
    
    # Oversample insider
    n_insider_target = int(len(X_insider) * insider_factor)
    X_insider_over, y_insider_over = resample(
        X_insider, y_insider,
        n_samples=n_insider_target,
        random_state=42,
        replace=True
    )
    
    # Combine
    X_test_over = np.concatenate([X_normal_over, X_insider_over], axis=0)
    y_test_over = np.concatenate([y_normal_over, y_insider_over], axis=0)
    
    # Shuffle
    perm = np.random.permutation(len(X_test_over))
    X_test_over = X_test_over[perm]
    y_test_over = y_test_over[perm]
    
    print(f"  After oversampling:")
    print(f"    Normal: {(y_test_over == 0).sum():,}")
    print(f"    Insider: {(y_test_over == 1).sum():,}")
    print(f"    Total: {len(X_test_over):,}")
    
    return X_test_over, y_test_over


def save_splits(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    config: dict
):
    """Save train/val/test splits."""
    print("\n--- Saving splits ---")
    
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / config['data']['processed_dir']
    
    for split_name, (X, y) in splits.items():
        X_path = processed_dir / f"X_{split_name}.npy"
        y_path = processed_dir / f"y_{split_name}.npy"
        np.save(X_path, X)
        np.save(y_path, y)
        print(f"  {split_name}: {X_path}")


def main():
    """Main entry point for sequence creation and splitting."""
    config = load_config()
    
    print("=" * 60)
    print("SEQUENCE CREATION & DATA SPLITTING")
    print("=" * 60)
    
    # Load processed data
    print("\nLoading processed data...")
    X, y, user_ids = load_processed_data(config)
    print(f"  Features: {X.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Insider sessions: {y.sum():,}")
    
    # Create sequences
    lookback = config['model']['lookback']
    stride = config['processing']['sequence_stride']
    X_seq, y_seq, user_seq = create_sequences_per_user(
        X, y, user_ids, lookback=lookback, stride=stride
    )
    
    # Split data
    splits = split_data(X_seq, y_seq, user_seq, config)
    
    # Oversample test set
    X_test_over, y_test_over = oversample_test_set(
        splits['test'][0], splits['test'][1], config
    )
    splits['test'] = (X_test_over, y_test_over)
    
    # Save splits
    save_splits(splits, config)
    
    print("\n" + "=" * 60)
    print("SEQUENCE CREATION & SPLITTING COMPLETE!")
    print("=" * 60)
    print(f"\nFinal summary:")
    print(f"  Train: {len(splits['train'][0]):,} sequences")
    print(f"  Val:   {len(splits['val'][0]):,} sequences")
    print(f"  Test:  {len(splits['test'][0]):,} sequences")


if __name__ == "__main__":
    main()
