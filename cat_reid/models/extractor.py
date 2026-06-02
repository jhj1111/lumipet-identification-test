import torch
import torch.nn as nn
import timm
import os
from PIL import Image
from cat_reid.preprocessing.transform import get_transform
from cat_reid.config import settings

class FeatureExtractor:
    def __init__(self, model_name=settings.MODEL_NAME, device=None, weights_path="best_projection.pth"):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {model_name} on {self.device}")
        
        # MegaDescriptor 로드 (Backbone)
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.to(self.device)
        self.model.eval()
        
        # Projection 레이어 정의 (trainer.py의 CatReIdModel 구조와 동일)
        # MegaDescriptor-L의 출력 차원은 1536입니다.
        self.projection = nn.Sequential(
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.projection.to(self.device)
        self.projection.eval()

        # 학습된 가중치 로드 시도
        if os.path.exists(weights_path):
            print(f"Found trained weights: {weights_path}. Applying custom projection.")
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.projection.load_state_dict(state_dict)
                self.has_custom_weights = True
            except Exception as e:
                print(f"Error loading weights: {e}. Falling back to base model.")
                self.has_custom_weights = False
        else:
            print(f"No custom weights found at {weights_path}. Using base MegaDescriptor features.")
            self.has_custom_weights = False
        
        self.transform = get_transform()

    @torch.no_grad()
    def extract(self, image):
        """
        이미지에서 embedding을 추출합니다.
        가중치가 로드된 경우 projection 레이어를 거친 최적화된 특징을 반환합니다.
        """
        if isinstance(image, Image.Image):
            img_tensor = self.transform(image)
        else:
            # numpy array (OpenCV 등) 처리
            img_pil = Image.fromarray(image)
            img_tensor = self.transform(img_pil)
            
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # 1. Backbone 특징 추출
        features = self.model(img_tensor)
        
        # 2. 학습된 가중치가 있다면 Projection 적용
        if self.has_custom_weights:
            embedding = self.projection(features)
        else:
            embedding = features
        
        # 3. L2 정규화 (코사인 유사도 계산을 위해)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy().flatten()
