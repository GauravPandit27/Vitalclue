import sqlite3
import time
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="vitalcue.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS session_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp DATETIME,
                    bpm REAL,
                    state TEXT,
                    confidence REAL,
                    expression TEXT,
                    gesture TEXT
                )
            ''')
            conn.commit()

    def log_vital(self, session_id: str, bpm: float, state: str, confidence: float, expression: str, gesture: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO session_logs (session_id, timestamp, bpm, state, confidence, expression, gesture)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, datetime.now(), bpm, state, confidence, expression, gesture))
            conn.commit()
