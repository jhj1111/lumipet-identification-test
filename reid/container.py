from reid.core.config import get_config
from reid.models.yolo.model import YoloModel
from reid.models.matcher.knn import KnnMatcher
from reid.models.matcher.faiss import FaissMatcher

def build_detector(cfg=None):
    cfg = cfg or get_config()
    return YoloModel(model_path=cfg.detector_weights)

def build_extractor(cfg=None):
    cfg = cfg or get_config()
    if cfg.extractor_type == "wildlife":
        from reid.models.extractor.wildlife.model import WildlifeExtractorModel
        return WildlifeExtractorModel(model_path=cfg.extractor_weights, model_name=cfg.model_name, cfg=cfg)
    else:
        from reid.models.extractor.mega_descriptor.model import ExtractorModel
        return ExtractorModel(model_path=cfg.extractor_weights, model_name=cfg.model_name, cfg=cfg)

def build_matcher(cfg=None):
    cfg = cfg or get_config()
    if cfg.matcher_type == "faiss":
        return FaissMatcher(threshold=cfg.threshold)
    else:
        return KnnMatcher(k=cfg.k, threshold=cfg.threshold)
