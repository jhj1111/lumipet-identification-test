from reid.core.config import get_config
from reid.models.yolo.model import YoloModel
from reid.models.extractor.model import ExtractorModel
from reid.models.matcher.knn import KnnMatcher
from reid.models.matcher.faiss import FaissMatcher

def build_detector(cfg=None):
    cfg = cfg or get_config()
    return YoloModel(model_path=cfg.detector_weights)

def build_extractor(cfg=None):
    cfg = cfg or get_config()
    return ExtractorModel(model_path=cfg.extractor_weights, model_name=cfg.model_name)

def build_matcher(cfg=None):
    cfg = cfg or get_config()
    if cfg.matcher_type == "faiss":
        return FaissMatcher(threshold=cfg.threshold)
    else:
        return KnnMatcher(k=cfg.k, threshold=cfg.threshold)
