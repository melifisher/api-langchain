import sqlite3
from typing import List, Tuple
from venv import logger

class ContextCache:
    def __init__(self, db_path: str = "context_cache.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_table()
        logger.info(f"Cache database initialized at {self.db_path}")

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS context (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer_full TEXT NOT NULL,
            answer_summary TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.conn.execute(query)
        self.conn.commit()

    def add_entry(self, user_id: str, question: str, answer_full: str, answer_summary: str):
        query = """
        INSERT INTO context (user_id, question, answer_full, answer_summary)
        VALUES (?, ?, ?, ?);
        """
        self.conn.execute(query, (user_id, question, answer_full, answer_summary))
        self.conn.commit()

    def get_history(self, user_id: str, limit: int = 10) -> List[Tuple[str, str, str]]:
        query = """
        SELECT question, answer_full, answer_summary
        FROM context
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?;
        """
        cursor = self.conn.execute(query, (user_id, limit))
        return cursor.fetchall()
    def get_ultimate_question(self, user_id: str) -> str | None:
        query = """
        SELECT question
        FROM context
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 1;
        """
        cursor = self.conn.execute(query, (user_id,))
        result = cursor.fetchone()
        return result[0] if result else None
    def get_ultimate_answerfull(self, user_id: str) -> str | None:
        query = """
        SELECT answer_full
        FROM context
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 1;
        """
        cursor = self.conn.execute(query, (user_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def clear_user_history(self, user_id: str):
        query = "DELETE FROM context WHERE user_id = ?;"
        self.conn.execute(query, (user_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()

