import numpy as np
import os
from pathlib import Path
from typing import Tuple, List

class EmbeddingStore:
    """
    Manages loading, saving, and adding embeddings to the database.
    Stored as a single .npz file: { 'embeddings': ..., 'labels': ... }
    """
    def __init__(self, db_path: str = "embeddings/db.npz"):
        self.db_path = Path(db_path)
        self.embeddings = []
        self.labels = []
        self.load()

    def add(self, embedding: np.ndarray, label: str):
        """Add a new embedding and label to the store."""
        # Ensure embedding is 2D (1, D)
        if self.embeddings and (self.embeddings == embedding).any(): return
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        self.embeddings.append(embedding)
        self.labels.append(label)

    def save(self):
        """Save the database to a .npz file."""
        if not self.embeddings:
            return
            
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_array = np.vstack(self.embeddings)
        labels_array = np.array(self.labels)
        
        np.savez(self.db_path, embeddings=embeddings_array, labels=labels_array)
        print(f"Saved {len(self.labels)} embeddings to {self.db_path}")

    def load(self):
        """Load the database from a .npz file."""
        if self.db_path.exists():
            try:
                data = np.load(self.db_path, allow_pickle=True)
                self.embeddings = [data['embeddings']]
                self.labels = data['labels'].tolist()
                print(f"Loaded {len(self.labels)} embeddings from {self.db_path}")
            except Exception as e:
                print(f"Error loading database: {e}")
                self.embeddings = []
                self.labels = []

    def get_all(self) -> Tuple[np.ndarray, List[str]]:
        """Return all embeddings as a single array and all labels."""
        if not self.embeddings:
            return np.empty((0, 0)), []
        return np.vstack(self.embeddings), self.labels

    def clear(self):
        """Clear all embeddings and labels."""
        self.embeddings = []
        self.labels = []
