"""
Feature Engineering Module
===========================
Session construction and feature extraction from events.

Key operations:
1. Sessionize events (logon → activities → logoff/next-logon)
2. Extract 12 features per session
3. Encode categorical fields
4. Handle missing values
5. Label sessions as insider/normal
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional
import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import yaml
from tqdm import tqdm
import pickle


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_db_connection(config: dict) -> duckdb.DuckDBPyConnection:
    """Get DuckDB connection."""
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / config['data']['duckdb_path']
    return duckdb.connect(str(db_path), read_only=True)


def sessionize_events(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Create sessions from events (logon → activities → logoff/next-logon).
    
    A session is defined as:
    - Starts with a Logon event
    - Ends with Logoff event OR next Logon event
    - Contains all activities in between
    
    Returns:
        DataFrame with one row per session containing:
        - user_id, session_start, session_end
        - logon_time (hour 1-24), logoff_time (hour 1-24)
        - day (0-6)
        - http_count, email_count, file_count, device_count
        - pc
    """
    print("\n--- Sessionizing events ---")
    
    # SQL to create sessions using window functions
    sql = """
    WITH ranked_events AS (
        SELECT 
            user_id,
            datetime,
            pc,
            activity,
            -- Mark session boundaries (Logon events)
            CASE WHEN activity = 'Logon' THEN 1 ELSE 0 END as is_logon,
            CASE WHEN activity = 'Logoff' THEN 1 ELSE 0 END as is_logoff
        FROM events
        ORDER BY user_id, datetime
    ),
    session_markers AS (
        SELECT 
            *,
            -- Cumulative sum of logons to create session groups
            SUM(is_logon) OVER (PARTITION BY user_id ORDER BY datetime) as session_num
        FROM ranked_events
    ),
    session_aggregates AS (
        SELECT 
            user_id,
            session_num,
            MIN(datetime) as session_start,
            MAX(datetime) as session_end,
            -- Time features (1-based hours)
            EXTRACT(HOUR FROM MIN(datetime)) + 1 as logon_time,
            EXTRACT(HOUR FROM MAX(datetime)) + 1 as logoff_time,
            EXTRACT(DOW FROM MIN(datetime)) as day,
            -- Activity counts
            SUM(CASE WHEN activity = 'Http' THEN 1 ELSE 0 END) as http_count,
            SUM(CASE WHEN activity = 'Email' THEN 1 ELSE 0 END) as email_count,
            SUM(CASE WHEN activity = 'File' THEN 1 ELSE 0 END) as file_count,
            SUM(CASE WHEN activity IN ('Connect', 'Disconnect') THEN 1 ELSE 0 END) as device_count,
            -- Primary PC (first one used)
            FIRST(pc) as pc,
            -- Session validation
            SUM(is_logon) as logon_count,
            SUM(is_logoff) as logoff_count,
            COUNT(*) as total_events
        FROM session_markers
        WHERE session_num > 0  -- Exclude events before first logon
        GROUP BY user_id, session_num
    )
    SELECT 
        user_id,
        session_num,
        session_start,
        session_end,
        logon_time,
        logoff_time,
        day,
        http_count,
        email_count,
        file_count,
        device_count,
        pc,
        total_events
    FROM session_aggregates
    WHERE logon_count >= 1  -- Must have at least one logon
    ORDER BY user_id, session_start
    """
    
    sessions_df = conn.execute(sql).fetchdf()
    
    print(f"  Total sessions created: {len(sessions_df):,}")
    print(f"  Unique users: {sessions_df['user_id'].nunique():,}")
    print(f"  Avg sessions per user: {len(sessions_df) / sessions_df['user_id'].nunique():.1f}")
    print(f"  Avg events per session: {sessions_df['total_events'].mean():.1f}")
    
    return sessions_df


def join_user_attributes(
    sessions_df: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection
) -> pd.DataFrame:
    """Join user attributes (role, functional_unit, department) from LDAP."""
    print("\n--- Joining user attributes ---")
    
    users_df = conn.execute("SELECT * FROM users").fetchdf()
    
    # Merge sessions with user attributes
    merged = sessions_df.merge(
        users_df[['user_id', 'role', 'functional_unit', 'department']],
        on='user_id',
        how='left'
    )
    
    # Check for missing user attributes
    missing = merged['role'].isna().sum()
    if missing > 0:
        print(f"  Warning: {missing} sessions missing user attributes")
    
    print(f"  Sessions with user attributes: {len(merged):,}")
    
    return merged


def label_insider_sessions(
    sessions_df: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection
) -> pd.DataFrame:
    """
    Label sessions as insider (1) or normal (0) based on ground truth.
    
    A session is insider if:
    - user_id is in insiders table
    - session_start falls within the malicious date range
    """
    print("\n--- Labeling insider sessions ---")
    
    # Get insider ground truth
    insiders_df = conn.execute("""
        SELECT user_id, start_date, end_date, scenario
        FROM insiders
    """).fetchdf()
    
    print(f"  Insider users: {len(insiders_df)}")
    
    # Initialize label column
    sessions_df['is_insider'] = 0
    sessions_df['scenario'] = None
    
    # Label sessions that fall within malicious date ranges
    for _, insider in insiders_df.iterrows():
        mask = (
            (sessions_df['user_id'] == insider['user_id']) &
            (sessions_df['session_start'] >= insider['start_date']) &
            (sessions_df['session_start'] <= insider['end_date'])
        )
        sessions_df.loc[mask, 'is_insider'] = 1
        sessions_df.loc[mask, 'scenario'] = insider['scenario']
    
    insider_count = sessions_df['is_insider'].sum()
    print(f"  Insider sessions: {insider_count:,}")
    print(f"  Normal sessions: {len(sessions_df) - insider_count:,}")
    print(f"  Insider ratio: {insider_count / len(sessions_df) * 100:.3f}%")
    
    return sessions_df


def encode_categorical_features(
    sessions_df: pd.DataFrame,
    config: dict
) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical features as integers.
    
    Returns:
        - DataFrame with encoded features
        - Dictionary of encoders for later use
    """
    print("\n--- Encoding categorical features ---")
    
    encoders = {}
    
    # Functional unit - use fixed mapping from config
    fu_mapping = config['encodings']['functional_unit']
    sessions_df['functional_unit_encoded'] = sessions_df['functional_unit'].map(fu_mapping)
    # Handle missing/unknown values
    sessions_df['functional_unit_encoded'] = sessions_df['functional_unit_encoded'].fillna(0).astype(int)
    
    # Label encode other categorical fields
    for col in ['user_id', 'pc', 'role', 'department']:
        le = LabelEncoder()
        # Handle NaN by converting to string first
        sessions_df[f'{col}_encoded'] = le.fit_transform(
            sessions_df[col].fillna('UNKNOWN').astype(str)
        ) + 1  # 1-based encoding
        encoders[col] = le
    
    print(f"  Encoded columns: user_id, pc, role, department, functional_unit")
    print(f"  Unique users: {sessions_df['user_id_encoded'].nunique()}")
    print(f"  Unique PCs: {sessions_df['pc_encoded'].nunique()}")
    print(f"  Unique roles: {sessions_df['role_encoded'].nunique()}")
    
    return sessions_df, encoders


def create_feature_matrix(sessions_df: pd.DataFrame) -> np.ndarray:
    """
    Create the final feature matrix with 12 features per session.
    
    Features (in order):
    1. logon_time (1-24)
    2. day (0-6)
    3. user_id_encoded
    4. http_count
    5. email_count
    6. file_count
    7. device_count
    8. pc_encoded
    9. logoff_time (1-24)
    10. role_encoded
    11. functional_unit_encoded
    12. department_encoded
    """
    print("\n--- Creating feature matrix ---")
    
    feature_columns = [
        'logon_time',
        'day',
        'user_id_encoded',
        'http_count',
        'email_count',
        'file_count',
        'device_count',
        'pc_encoded',
        'logoff_time',
        'role_encoded',
        'functional_unit_encoded',
        'department_encoded'
    ]
    
    # Handle missing values
    for col in feature_columns:
        if sessions_df[col].isna().any():
            if col in ['http_count', 'email_count', 'file_count', 'device_count']:
                sessions_df[col] = sessions_df[col].fillna(0)
            else:
                # Use mode for categorical, mean for numerical
                sessions_df[col] = sessions_df[col].fillna(sessions_df[col].mode()[0])
    
    X = sessions_df[feature_columns].values.astype(np.float32)
    
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Features: {feature_columns}")
    
    return X


def normalize_features(
    X: np.ndarray,
    scaler: Optional[MinMaxScaler] = None
) -> Tuple[np.ndarray, MinMaxScaler]:
    """Apply Min-Max normalization to features."""
    print("\n--- Normalizing features ---")
    
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_normalized = scaler.fit_transform(X)
    else:
        X_normalized = scaler.transform(X)
    
    print(f"  Min values: {X_normalized.min(axis=0).round(3)}")
    print(f"  Max values: {X_normalized.max(axis=0).round(3)}")
    
    return X_normalized, scaler


def save_processed_data(
    sessions_df: pd.DataFrame,
    X: np.ndarray,
    encoders: dict,
    scaler: Optional[MinMaxScaler],
    config: dict
):
    """Save processed data and artifacts."""
    print("\n--- Saving processed data ---")
    
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / config['data']['processed_dir']
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sessions DataFrame (with all columns for debugging)
    sessions_path = processed_dir / "sessions.parquet"
    sessions_df.to_parquet(sessions_path)
    print(f"  Sessions: {sessions_path}")
    
    # Save feature matrix
    features_path = processed_dir / "features.npy"
    np.save(features_path, X)
    print(f"  Features: {features_path}")
    
    # Save labels
    labels_path = processed_dir / "labels.npy"
    np.save(labels_path, sessions_df['is_insider'].values)
    print(f"  Labels: {labels_path}")
    
    # Save user IDs (for per-user sequence creation)
    users_path = processed_dir / "user_ids.npy"
    np.save(users_path, sessions_df['user_id_encoded'].values)
    print(f"  User IDs: {users_path}")
    
    # Save encoders and scaler
    artifacts_path = processed_dir / "preprocessing_artifacts.pkl"
    with open(artifacts_path, 'wb') as f:
        pickle.dump({
            'encoders': encoders,
            'scaler': scaler,
            'feature_columns': [
                'logon_time', 'day', 'user_id_encoded', 'http_count',
                'email_count', 'file_count', 'device_count', 'pc_encoded',
                'logoff_time', 'role_encoded', 'functional_unit_encoded',
                'department_encoded'
            ]
        }, f)
    print(f"  Artifacts: {artifacts_path}")


def main(normalize: bool = False):
    """
    Main entry point for feature engineering.
    
    Args:
        normalize: If True, apply Min-Max normalization. If False, use raw values.
    """
    config = load_config()
    
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    print(f"  Normalization: {'ENABLED' if normalize else 'DISABLED'}")
    
    # Connect to database
    conn = get_db_connection(config)
    
    try:
        # Step 1: Sessionize events
        sessions_df = sessionize_events(conn)
        
        # Step 2: Join user attributes
        sessions_df = join_user_attributes(sessions_df, conn)
        
        # Step 3: Label insider sessions
        sessions_df = label_insider_sessions(sessions_df, conn)
        
        # Step 4: Encode categorical features
        sessions_df, encoders = encode_categorical_features(sessions_df, config)
        
        # Step 5: Create feature matrix
        X = create_feature_matrix(sessions_df)
        
        # Step 6: Conditionally normalize features
        if normalize:
            X_final, scaler = normalize_features(X)
        else:
            print("\n--- Skipping normalization (raw values) ---")
            X_final = X
            scaler = None
        
        # Step 7: Save processed data
        save_processed_data(sessions_df, X_final, encoders, scaler, config)
        
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING COMPLETE!")
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  Total sessions: {len(sessions_df):,}")
        print(f"  Insider sessions: {sessions_df['is_insider'].sum():,}")
        print(f"  Feature dimensions: {X_final.shape[1]}")
        print(f"  Normalized: {normalize}")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
