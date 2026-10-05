"""
VitalCue - Session Logging

Off by default. Continuous physiological logs are the part of this system with real
privacy weight, so persistence is opt-in rather than opt-out: with persist=False
nothing touches disk and the session is held in memory only.
"""
import sqlite3
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_path: str = "vitalcue.db", persist: bool = False):
        self.db_path = db_path
        self.persist = persist
        self._memory_log = []
        if self.persist:
            self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS session_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp DATETIME,
                    bpm REAL,
                    state TEXT,
                    confidence REAL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS recovery_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp DATETIME,
                    duration_seconds REAL,
                    resolved INTEGER
                )
                """
            )
            conn.commit()

    def log_vital(self, session_id: str, bpm: float, state: str, confidence: float):
        row = (session_id, datetime.now(), bpm, state, confidence)
        if not self.persist:
            self._memory_log.append(row)
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO session_logs (session_id, timestamp, bpm, state, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                row,
            )
            conn.commit()

    def log_recovery(self, session_id: str, duration_seconds: float, resolved: bool):
        """Outcome of one stress episode - did the intervention bring the rate back down."""
        row = (session_id, datetime.now(), duration_seconds, int(resolved))
        if not self.persist:
            self._memory_log.append(row)
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO recovery_events (session_id, timestamp, duration_seconds, resolved)
                VALUES (?, ?, ?, ?)
                """,
                row,
            )
            conn.commit()

    def clear_memory_log(self):
        self._memory_log.clear()
