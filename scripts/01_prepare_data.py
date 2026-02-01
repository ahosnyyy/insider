"""
Data Preparation Script
=======================
Run the complete data pipeline:
1. CSV → Parquet conversion
2. Parquet → DuckDB ingestion
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data.csv_to_parquet import main as convert_to_parquet
from data.parquet_to_duckdb import main as ingest_to_duckdb


def main():
    print("=" * 70)
    print("INSIDER THREAT DETECTION - DATA PREPARATION")
    print("=" * 70)
    
    # Step 1: Convert CSV to Parquet
    print("\n[STEP 1/2] Converting CSV files to Parquet format...")
    convert_to_parquet()
    
    # Step 2: Ingest Parquet to DuckDB
    print("\n[STEP 2/2] Ingesting Parquet files into DuckDB...")
    ingest_to_duckdb()
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 70)
    print("\nNext step: Run 02_feature_engineering.py to create sessions and features")


if __name__ == "__main__":
    main()
