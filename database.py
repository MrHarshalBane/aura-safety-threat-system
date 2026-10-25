import sqlite3
from datetime import datetime
import pandas as pd
import os

DB_NAME = "aura.db"

def get_connection(db_path=DB_NAME):
    """Create and return a thread-safe connection to the SQLite database."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path=DB_NAME):
    """Initializes the database schema if it does not already exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            threat_level TEXT NOT NULL,
            person_count INTEGER NOT NULL,
            trigger_reason TEXT NOT NULL,
            dispatch_id TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_incident(threat_level: str, person_count: int, trigger_reason: str, dispatch_id: str = None, db_path=DB_NAME):
    """Logs a safety incident / status transition to SQLite database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO incidents (timestamp, threat_level, person_count, trigger_reason, dispatch_id)
        VALUES (?, ?, ?, ?, ?)
    """, (now_iso, threat_level, person_count, trigger_reason, dispatch_id))
    conn.commit()
    conn.close()

def get_recent_incidents(limit: int = 12, db_path=DB_NAME):
    """Fetches the most recent incidents formatted for the Activity Log."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, threat_level, person_count, trigger_reason, dispatch_id
        FROM incidents
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        # Extract time portion for compact UI display
        dt_str = r["timestamp"]
        time_part = dt_str.split(" ")[1] if " " in dt_str else dt_str
        results.append({
            "time": time_part,
            "full_time": dt_str,
            "threat_level": r["threat_level"],
            "person_count": r["person_count"],
            "reason": r["trigger_reason"],
            "dispatch_id": r["dispatch_id"]
        })
    return results

def get_hourly_incident_stats(db_path=DB_NAME) -> pd.DataFrame:
    """Aggregates alert incidents by hour for trend visualization."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    # Count CAUTION and ALERT events grouped by hour
    query = """
        SELECT strftime('%H:00', timestamp) AS hour, COUNT(*) AS count
        FROM incidents
        WHERE threat_level IN ('ALERT', 'CAUTION')
        GROUP BY hour
        ORDER BY hour ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        # Provide clean default timeline if database is newly initialized
        current_hour = datetime.now().strftime("%H:00")
        df = pd.DataFrame([{"hour": current_hour, "count": 0}])
        
    return df
