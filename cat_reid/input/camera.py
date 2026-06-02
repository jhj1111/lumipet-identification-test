import cv2
from cat_reid.models.extractor import FeatureExtractor
from cat_reid.inference.knn_classifier import KnnClassifier
from cat_reid.db.embedding_store import EmbeddingStore
from cat_reid.visualization.renderer import Renderer

class CameraInference:
    def __init__(self, extractor=None, store=None, classifier=None):
        self.extractor = extractor if extractor else FeatureExtractor()
        self.store = store if store else EmbeddingStore()
        self.classifier = classifier if classifier else KnnClassifier()
        self.renderer = Renderer()
        
        # DB 로드 및 분류기 학습
        embeddings, labels = self.store.get_all()
        if len(embeddings) > 0:
            self.classifier.fit(embeddings, labels)

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1단계에서는 ROI 지정 없이 전체 프레임에서 특징 추출 (성능은 낮을 수 있음)
            # 실제로는 고양이가 화면 중앙에 있다고 가정하거나 나중에 YOLO와 결합
            
            # BGR -> RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Embedding 추출 및 추론
            embedding = self.extractor.extract(frame_rgb)
            label, similarity = self.classifier.predict(embedding)
            
            # 결과 렌더링
            frame = self.renderer.draw_info(frame, label, similarity)
            self.renderer.show(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
