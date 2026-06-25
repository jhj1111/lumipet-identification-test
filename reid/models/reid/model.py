from typing import Any, Optional

from reid.models.yolo.model import YoloModel
from reid.engine.model import BaseModel
from reid.models.extractor.model import ExtractorModel
from reid.models.matcher.model import BaseMatcher
from reid.models.reid.predict import ReIdPredictor


class ReIdModel(BaseModel):
    """
    Full Re-ID Pipeline Model.
    """
    def __init__(self, detector: YoloModel, extractor: ExtractorModel, matcher: BaseMatcher, cfg: Optional[Any] = None) -> None:
        super().__init__(task="reid_pipeline", cfg=cfg)
        self.detector = detector
        self.extractor = extractor
        self.matcher = matcher

    def _load_model(self, weights: str) -> None:
        pass

    def _get_predictor(self) -> ReIdPredictor:
        return ReIdPredictor(self.detector, self.extractor, self.matcher, self.cfg)

