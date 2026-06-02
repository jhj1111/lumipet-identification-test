import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from cat_reid.config import settings

class KnnClassifier:
    def __init__(self, k=settings.K, threshold=settings.THRESHOLD):
        self.k = k
        self.threshold = threshold
        self.model = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        self.is_fitted = False
        self.labels = []

    def fit(self, embeddings, labels):
        """
        DB의 데이터로 KNN 모델을 준비합니다.
        """
        if len(embeddings) == 0:
            return
            
        self.model.fit(embeddings, labels)
        self.labels = labels
        self.is_fitted = True

    def predict(self, query_embedding):
        """
        가장 유사한 개체를 찾습니다.
        임계값보다 유사도가 낮으면 'Unknown'을 반환합니다.
        """
        if not self.is_fitted:
            return "Unknown", 0.0

        # KNN 결과 (거리와 인덱스)
        # sklearn의 cosine metric은 1 - cosine_similarity를 반환함
        distances, indices = self.model.kneighbors([query_embedding])
        
        best_idx = indices[0][0]
        best_dist = distances[0][0]
        similarity = 1.0 - best_dist
        
        if similarity < self.threshold:
            return "Unknown", similarity
            
        return self.labels[best_idx], similarity
