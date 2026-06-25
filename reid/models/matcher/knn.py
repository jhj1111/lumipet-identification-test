import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing import List
from reid.models.matcher.model import BaseMatcher
from reid.core.types import MatchResult

class KnnMatcher(BaseMatcher):
    """
    KNN Matcher using Scikit-Learn.
    """
    def __init__(self, k: int = 2, threshold: float = 0.7):
        super().__init__(threshold)
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        self.is_fitted = False

    def fit(self, embeddings: np.ndarray, labels: List[str]):
        if len(embeddings) == 0:
            return
        self.model.fit(embeddings, labels)
        self.labels = labels
        self.is_fitted = True

    def match(self, query_embedding: np.ndarray) -> MatchResult:
        if not self.is_fitted:
            return MatchResult(cat_id="Unknown", similarity=0.0, is_known=False)

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.model.kneighbors(query_embedding)
        
        best_idx = indices[0][0]
        best_dist = distances[0][0]
        similarity = 1.0 - best_dist # Cosine similarity = 1 - cosine distance
        
        is_known = similarity >= self.threshold
        cat_id = self.labels[best_idx] if is_known else "Unknown"
        
        return MatchResult(cat_id=cat_id, similarity=float(similarity), is_known=is_known)
