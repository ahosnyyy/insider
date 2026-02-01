"""
Parquet to DuckDB Ingestion
===========================
Load Parquet files into DuckDB for SQL-based feature engineering.
Creates unified events table, users table, and labels table.
"""

import os
import sys
from pathlib import Path
import duckdb
import yaml


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_database(config: dict) -> duckdb.DuckDBPyConnection:
    """Create DuckDB database and return connection."""
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / config['data']['duckdb_path']
    
    # Create directory if needed
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing database for fresh start
    if db_path.exists():
        os.remove(db_path)
    
    print(f"Creating database: {db_path}")
    return duckdb.connect(str(db_path))


def create_events_table(conn: duckdb.DuckDBPyConnection, parquet_dir: Path):
    """Create unified events table from all activity Parquet files."""
    print("\n--- Creating events table ---")
    
    # Build SQL dynamically based on which parquet files exist
    queries = []
    
    # Logon events
    logon_path = parquet_dir / "logon.parquet"
    if logon_path.exists():
        queries.append(f"""
        SELECT 
            id,
            strptime(date, '%m/%d/%Y %H:%M:%S') as datetime,
            "user" as user_id,
            pc,
            activity,
            NULL as to_addr,
            NULL as cc,
            NULL as bcc,
            NULL as size,
            NULL as attachments,
            NULL as url,
            NULL as filename,
            NULL as content
        FROM read_parquet('{logon_path}')
        """)
        print(f"  Including: logon.parquet")
    
    # Device events
    device_path = parquet_dir / "device.parquet"
    if device_path.exists():
        queries.append(f"""
        SELECT 
            id,
            strptime(date, '%m/%d/%Y %H:%M:%S') as datetime,
            "user" as user_id,
            pc,
            activity,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
        FROM read_parquet('{device_path}')
        """)
        print(f"  Including: device.parquet")
    
    # Email events
    email_path = parquet_dir / "email.parquet"
    if email_path.exists():
        queries.append(f"""
        SELECT 
            id,
            strptime(date, '%m/%d/%Y %H:%M:%S') as datetime,
            "user" as user_id,
            pc,
            'Email' as activity,
            "to" as to_addr,
            cc,
            bcc,
            size,
            attachments,
            NULL as url,
            NULL as filename,
            content
        FROM read_parquet('{email_path}')
        """)
        print(f"  Including: email.parquet")
    
    # File events
    file_path = parquet_dir / "file.parquet"
    if file_path.exists():
        queries.append(f"""
        SELECT 
            id,
            strptime(date, '%m/%d/%Y %H:%M:%S') as datetime,
            "user" as user_id,
            pc,
            'File' as activity,
            NULL, NULL, NULL, NULL, NULL, NULL,
            filename,
            content
        FROM read_parquet('{file_path}')
        """)
        print(f"  Including: file.parquet")
    
    # HTTP events (optional - may fail to convert due to size)
    http_path = parquet_dir / "http.parquet"
    if http_path.exists():
        queries.append(f"""
        SELECT 
            id,
            strptime(date, '%m/%d/%Y %H:%M:%S') as datetime,
            "user" as user_id,
            pc,
            'Http' as activity,
            NULL, NULL, NULL, NULL, NULL,
            url,
            NULL as filename,
            content
        FROM read_parquet('{http_path}')
        """)
        print(f"  Including: http.parquet")
    else:
        print(f"  WARNING: http.parquet not found - HTTP events will be missing!")
    
    if not queries:
        raise ValueError("No parquet files found!")
    
    # Combine queries with UNION ALL
    sql = "CREATE TABLE events AS\n" + "\nUNION ALL\n".join(queries)
    
    conn.execute(sql)
    
    # Create index on user_id and datetime for fast session queries
    conn.execute("CREATE INDEX idx_events_user_time ON events(user_id, datetime)")
    
    # Get row count
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    print(f"  Total events: {count:,}")
    
    # Show activity breakdown
    activities = conn.execute("""
        SELECT activity, COUNT(*) as cnt 
        FROM events 
        GROUP BY activity
        ORDER BY cnt DESC
    """).fetchall()
    for act, cnt in activities:
        print(f"    {act}: {cnt:,}")
    
    return count


def create_users_table(conn: duckdb.DuckDBPyConnection, parquet_dir: Path):
    """Create users table from LDAP data (latest record per user)."""
    print("\n--- Creating users table ---")
    
    sql = f"""
    CREATE TABLE users AS
    WITH ranked AS (
        SELECT 
            user_id,
            employee_name,
            email,
            role,
            business_unit,
            functional_unit,
            department,
            team,
            supervisor,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY user_id DESC) as rn
        FROM read_parquet('{parquet_dir}/ldap.parquet')
    )
    SELECT 
        user_id,
        employee_name,
        email,
        role,
        business_unit,
        functional_unit,
        department,
        team,
        supervisor
    FROM ranked
    WHERE rn = 1
    """
    
    conn.execute(sql)
    
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    print(f"  Total users: {count:,}")
    
    return count


def create_labels_table(conn: duckdb.DuckDBPyConnection, raw_dir: Path):
    """Create insider labels table from ground truth."""
    print("\n--- Creating labels table ---")
    
    insiders_path = raw_dir / "answers" / "insiders.csv"
    
    sql = f"""
    CREATE TABLE insiders AS
    SELECT 
        dataset,
        scenario,
        details,
        "user" as user_id,
        strptime(start, '%m/%d/%Y %H:%M:%S') as start_date,
        strptime("end", '%m/%d/%Y %H:%M:%S') as end_date
    FROM read_csv('{insiders_path}', header=true)
    WHERE dataset = '4.2'
    """
    
    conn.execute(sql)
    
    count = conn.execute("SELECT COUNT(*) FROM insiders").fetchone()[0]
    print(f"  Insider users (dataset 4.2): {count}")
    
    # Show scenario breakdown
    scenarios = conn.execute("""
        SELECT scenario, COUNT(*) as count 
        FROM insiders 
        GROUP BY scenario
        ORDER BY scenario
    """).fetchall()
    for scenario, cnt in scenarios:
        print(f"    Scenario {scenario}: {cnt} users")
    
    return count


def create_feature_encoding_tables(conn: duckdb.DuckDBPyConnection, config: dict):
    """Create lookup tables for feature encodings."""
    print("\n--- Creating encoding tables ---")
    
    # Functional unit encoding
    fu_mapping = config['encodings']['functional_unit']
    values = [(k, v) for k, v in fu_mapping.items()]
    
    conn.execute("""
        CREATE TABLE functional_unit_encoding (
            name VARCHAR,
            code INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO functional_unit_encoding VALUES (?, ?)",
        values
    )
    print(f"  Functional unit encodings: {len(values)}")
    
    # Activity encoding (for reference)
    act_mapping = config['encodings']['activity']
    values = [(k, v) for k, v in act_mapping.items()]
    
    conn.execute("""
        CREATE TABLE activity_encoding (
            name VARCHAR,
            code INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO activity_encoding VALUES (?, ?)",
        values
    )
    print(f"  Activity encodings: {len(values)}")


def print_summary(conn: duckdb.DuckDBPyConnection):
    """Print database summary."""
    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)
    
    tables = conn.execute("SHOW TABLES").fetchall()
    for (table,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count:,} rows")


def main():
    """Main entry point for Parquet to DuckDB ingestion."""
    config = load_config()
    
    project_root = Path(__file__).parent.parent.parent
    parquet_dir = project_root / config['data']['parquet_dir']
    raw_dir = project_root / config['data']['raw_dir']
    
    print("=" * 60)
    print("Parquet to DuckDB Ingestion")
    print("=" * 60)
    print(f"Parquet dir: {parquet_dir}")
    
    # Check if parquet files exist
    if not (parquet_dir / "logon.parquet").exists():
        print("\nERROR: Parquet files not found. Run csv_to_parquet.py first.")
        sys.exit(1)
    
    # Create database
    conn = create_database(config)
    
    try:
        # Create tables
        create_events_table(conn, parquet_dir)
        create_users_table(conn, parquet_dir)
        create_labels_table(conn, raw_dir)
        create_feature_encoding_tables(conn, config)
        
        # Print summary
        print_summary(conn)
        
        print("\n" + "=" * 60)
        print("DuckDB database created successfully!")
        print("=" * 60)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
