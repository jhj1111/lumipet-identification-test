from typing import Any, Optional, Union
from pathlib import Path
from ultralytics import YOLO
from reid.engine.model import BaseModel
from reid.models.yolo.detect.predict import YoloPredictor

class YoloModel(BaseModel):
    """
    YOLOv8 Detection Model Wrapper.
    """
    def __init__(self, model_path: Optional[Union[str, Path]] = "yolov8n.pt", task: str = "detect"):
        super().__init__(model_path, task)
        self._load_model(model_path)

    def _load_model(self, weights: str):
        self.model = YOLO(weights)

    def _get_predictor(self, **kwargs):
        predictor = YoloPredictor()
        predictor.setup_model(self.model)
        return predictor
