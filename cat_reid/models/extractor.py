import torch
import timm
from PIL import Image
from cat_reid.preprocessing.transform import get_transform
from cat_reid.config import settings

class FeatureExtractor:
    def __init__(self, model_name=settings.MODEL_NAME, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {model_name} on {self.device}")
        
        # MegaDescriptor 로드
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = get_transform()

    @torch.no_grad()
    def extract(self, image):
        """
        이미지에서 embedding을 추출합니다.
        image: PIL Image or numpy array
        """
        if isinstance(image, Image.Image):
            img_tensor = self.transform(image)
        else:
            # numpy array (OpenCV 등) 처리
            img_pil = Image.fromarray(image)
            img_tensor = self.transform(img_pil)
            
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        embedding = self.model(img_tensor)
        
        # L2 정규화 (코사인 유사도 계산을 위해)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy().flatten()
