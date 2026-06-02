import torchvision.transforms as T
from cat_reid.config import settings

def get_transform():
    """
    MegaDescriptor를 위한 기본 전처리 파이프라인
    """
    return T.Compose([
        T.Resize(settings.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
