import os
import glob
from PIL import Image
from cat_reid.models.extractor import FeatureExtractor
from cat_reid.db.embedding_store import EmbeddingStore

class CatRegister:
    def __init__(self, extractor=None, store=None):
        self.extractor = extractor if extractor else FeatureExtractor()
        self.store = store if store else EmbeddingStore()

    def register_image(self, image_path, label):
        """
        단일 이미지를 등록합니다.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            embedding = self.extractor.extract(img)
            self.store.add(embedding, label)
            print(f"Registered {image_path} as {label}")
        except Exception as e:
            print(f"Failed to register {image_path}: {e}")

    def register_directory(self, dir_path):
        """
        디렉토리 구조를 기반으로 벌크 등록합니다.
        구조: dir_path/label/*.jpg
        """
        subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        
        for label in subdirs:
            label_dir = os.path.join(dir_path, label)
            image_paths = glob.glob(os.path.join(label_dir, "**", "*.*"), recursive=True)
            
            for img_path in image_paths:
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.register_image(img_path, label)
        
        self.store.save()
