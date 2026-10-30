import sqlite3
import os
import threading
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            self._shared_conn = None
        else:
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
            self._shared_conn.execute("PRAGMA foreign_keys = ON")

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    role TEXT DEFAULT 'Member',
                    thumbnail_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS face_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    photo_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    user_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    similarity_score REAL NOT NULL,
                    snapshot_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_samples_user ON face_samples(user_id);
                CREATE INDEX IF NOT EXISTS idx_logs_created ON recognition_logs(created_at DESC);
            """)

    def add_user(self, name: str, role: str = "Member", thumbnail_path: Optional[str] = None) -> int:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (name, role, thumbnail_path) VALUES (?, ?, ?)",
                (name.strip(), role.strip() if role else "Member", thumbnail_path)
            )
            return cur.lastrowid

    def update_user_thumbnail(self, user_id: int, thumbnail_path: str):
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE users SET thumbnail_path = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (thumbnail_path, user_id)
            )

    def add_face_sample(self, user_id: int, embedding: np.ndarray, photo_path: Optional[str] = None) -> int:
        emb_bytes = embedding.astype(np.float32).tobytes()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO face_samples (user_id, embedding, photo_path) VALUES (?, ?, ?)",
                (user_id, emb_bytes, photo_path)
            )
            return cur.lastrowid

    def get_all_users(self) -> List[Dict]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            rows = cur.execute("""
                SELECT u.id, u.name, u.role, u.thumbnail_path, u.created_at, u.updated_at,
                       COUNT(s.id) as sample_count
                FROM users u
                LEFT JOIN face_samples s ON u.id = s.user_id
                GROUP BY u.id
                ORDER BY u.id DESC
            """).fetchall()
            return [dict(r) for r in rows]

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None

    def delete_user(self, user_id: int) -> bool:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return cur.rowcount > 0

    def get_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """Returns list of (user_id, user_name, embedding_128d)"""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT u.id, u.name, s.embedding
                FROM face_samples s
                JOIN users u ON s.user_id = u.id
            """).fetchall()
            results = []
            for r in rows:
                emb = np.frombuffer(r["embedding"], dtype=np.float32).reshape(128)
                results.append((r["id"], r["name"], emb))
            return results

    def add_log(self, user_id: Optional[int], user_name: str, confidence: float, similarity_score: float, snapshot_path: Optional[str] = None):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO recognition_logs (user_id, user_name, confidence, similarity_score, snapshot_path) VALUES (?, ?, ?, ?, ?)",
                (user_id, user_name, round(confidence, 1), round(similarity_score, 4), snapshot_path)
            )

    def get_logs(self, limit: int = 100) -> List[Dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM recognition_logs ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            return [dict(r) for r in rows]

    def clear_logs(self):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM recognition_logs")
