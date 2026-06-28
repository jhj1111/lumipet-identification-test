import sqlite3
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

class EmbeddingStore:
    """
    Manages loading, saving, and adding embeddings to the database.
    Stored as a SQLite database.
    """
    # SQL Query Constants for Clean Management
    SQL_CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            embedding BLOB NOT NULL,
            image_path TEXT,
            image_hash TEXT UNIQUE,
            model_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """
    SQL_CREATE_INDEX_LABEL = "CREATE INDEX IF NOT EXISTS idx_embeddings_label ON embeddings(label);"
    SQL_CREATE_INDEX_HASH = "CREATE INDEX IF NOT EXISTS idx_embeddings_hash ON embeddings(image_hash);"
    
    SQL_INSERT_EMBEDDING = """
        INSERT INTO embeddings (label, embedding, image_path, image_hash, model_name)
        VALUES (?, ?, ?, ?, ?)
    """
    SQL_INSERT_IGNORE_EMBEDDING = """
        INSERT OR IGNORE INTO embeddings (label, embedding, image_path, image_hash, model_name)
        VALUES (?, ?, ?, ?, ?)
    """
    SQL_SELECT_DIFF_MODELS = "SELECT DISTINCT model_name FROM embeddings WHERE model_name != ?"
    SQL_SELECT_ALL_BY_MODEL = "SELECT embedding, label FROM embeddings WHERE model_name = ?"
    SQL_SELECT_ALL = "SELECT embedding, label FROM embeddings"
    SQL_SELECT_LABEL_COUNTS = "SELECT label, COUNT(id) FROM embeddings GROUP BY label"
    SQL_DELETE_BY_LABEL = "DELETE FROM embeddings WHERE label = ?"
    SQL_DELETE_ALL = "DELETE FROM embeddings"

    def __init__(self, db_path: str = "embeddings/db.db"):
        path = Path(db_path)
        # Automatic mapping of .npz to .db extension
        if path.suffix == ".npz":
            path = path.with_suffix(".db")
            
        self.db_path = path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        """Create tables and indexes if they do not exist."""
        cursor = self.conn.cursor()
        cursor.execute(self.SQL_CREATE_TABLE)
        cursor.execute(self.SQL_CREATE_INDEX_LABEL)
        cursor.execute(self.SQL_CREATE_INDEX_HASH)
        self.conn.commit()

    def close(self):
        """Close connection."""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()

    def add(self, embedding: np.ndarray, label: str, image_path: Optional[str] = None, image_hash: Optional[str] = None, model_name: str = "") -> None:
        """Add a single embedding and label to the store."""
        if embedding.ndim == 2:
            embedding = embedding.flatten()
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                self.SQL_INSERT_EMBEDDING,
                (label, embedding_bytes, image_path, image_hash, model_name)
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            # Hash already exists, ignore duplication silently
            pass

    def add_batch(self, embeddings: np.ndarray, labels: List[str], image_paths: List[str], image_hashes: List[str], model_name: str) -> None:
        """Insert multiple embeddings in a single transaction."""
        cursor = self.conn.cursor()
        data = []
        for i in range(len(labels)):
            emb = embeddings[i]
            if emb.ndim == 2:
                emb = emb.flatten()
            embedding_bytes = emb.astype(np.float32).tobytes()
            data.append((labels[i], embedding_bytes, image_paths[i], image_hashes[i], model_name))
            
        try:
            cursor.executemany(self.SQL_INSERT_IGNORE_EMBEDDING, data)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e

    def get_all(self, model_name: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Return all embeddings and labels matching model_name."""
        cursor = self.conn.cursor()
        if model_name:
            # Check for different model embeddings
            cursor.execute(self.SQL_SELECT_DIFF_MODELS, (model_name,))
            diff_models = [row[0] for row in cursor.fetchall()]
            if diff_models:
                print(f"Warning: Database contains embeddings from a different model(s): {diff_models}. "
                      f"Please run 'reid migrate' to regenerate features.")
            
            cursor.execute(self.SQL_SELECT_ALL_BY_MODEL, (model_name,))
        else:
            cursor.execute(self.SQL_SELECT_ALL)
            
        rows = cursor.fetchall()
        if not rows:
            return np.empty((0, 0), dtype=np.float32), []
            
        embeddings_list = []
        labels = []
        for row in rows:
            emb_arr = np.frombuffer(row[0], dtype=np.float32)
            embeddings_list.append(emb_arr)
            labels.append(row[1])
            
        return np.vstack(embeddings_list), labels

    def list_labels(self) -> Dict[str, int]:
        """Return label counts."""
        cursor = self.conn.cursor()
        cursor.execute(self.SQL_SELECT_LABEL_COUNTS)
        return {row[0]: row[1] for row in cursor.fetchall()}

    def delete_label(self, label: str) -> int:
        """Delete entries matching label."""
        cursor = self.conn.cursor()
        cursor.execute(self.SQL_DELETE_BY_LABEL, (label,))
        self.conn.commit()
        return cursor.rowcount

    def clear(self) -> None:
        """Clear database."""
        cursor = self.conn.cursor()
        cursor.execute(self.SQL_DELETE_ALL)
        self.conn.commit()

    # Legacy/compatibility methods
    def save(self):
        pass # Not needed for SQLite as auto-commits are used

    def load(self):
        pass # Initialized in constructor
