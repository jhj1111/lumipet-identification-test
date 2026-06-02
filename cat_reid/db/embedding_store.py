import numpy as np
import os
import pickle
from cat_reid.config import settings

class EmbeddingStore:
    def __init__(self, db_path=settings.DB_PATH, label_path=settings.LABEL_PATH):
        self.db_path = db_path
        self.label_path = label_path
        self.embeddings = []
        self.labels = []
        self.load()

    def add(self, embedding, label):
        """
        새로운 embedding과 레이블을 추가합니다.
        """
        self.embeddings.append(embedding)
        self.labels.append(label)

    def save(self):
        """
        데이터를 디스크에 저장합니다.
        """
        if not self.embeddings:
            return
            
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        np.save(self.db_path, np.array(self.embeddings))
        
        with open(self.label_path, 'wb') as f:
            pickle.dump(self.labels, f)
        print(f"Saved {len(self.embeddings)} embeddings to {self.db_path}")

    def load(self):
        """
        디스크에서 데이터를 로드합니다.
        """
        if os.path.exists(self.db_path) and os.path.exists(self.label_path):
            try:
                self.embeddings = list(np.load(self.db_path))
                with open(self.label_path, 'rb') as f:
                    self.labels = pickle.load(f)
                print(f"Loaded {len(self.embeddings)} embeddings from {self.db_path}")
            except Exception as e:
                print(f"Error loading database: {e}")
                self.embeddings = []
                self.labels = []
        else:
            self.embeddings = []
            self.labels = []

    def get_all(self):
        return np.array(self.embeddings), self.labels
