import cv2
from PIL import Image
from cat_reid.models.extractor import FeatureExtractor
from cat_reid.inference.knn_classifier import KnnClassifier
from cat_reid.db.embedding_store import EmbeddingStore

class FileInference:
    def __init__(self, extractor=None, store=None, classifier=None):
        self.extractor = extractor if extractor else FeatureExtractor()
        self.store = store if store else EmbeddingStore()
        self.classifier = classifier if classifier else KnnClassifier()
        
        # DB 로드 및 분류기 학습
        embeddings, labels = self.store.get_all()
        if len(embeddings) > 0:
            self.classifier.fit(embeddings, labels)

    def predict(self, image_path):
        """
        이미지 파일에 대해 Re-ID를 수행합니다.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            embedding = self.extractor.extract(img)
            label, similarity = self.classifier.predict(embedding)
            return label, similarity
        except Exception as e:
            print(f"Inference failed for {image_path}: {e}")
            return "Error", 0.0
