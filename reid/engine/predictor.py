from abc import ABC, abstractmethod
from typing import Any, List, Union
import numpy as np
from reid.core.config import get_config

class BasePredictor(ABC):
    """
    Abstract Base Class for all Predictors.
    Standardizes the inference pipeline: Preprocess -> Inference -> Postprocess.
    """
    def __init__(self, config=None):
        self.cfg = config if config else get_config()
        self.device = self.cfg.device
        self.model = None

    def __call__(self, source: Any, **kwargs) -> Any:
        """Main entry point for prediction."""
        return self.predict(source, **kwargs)

    def predict(self, source: Any, **kwargs) -> Any:
        """Execution flow for a single or batch source."""
        # 1. Preprocess
        im = self.preprocess(source)
        
        # 2. Inference
        preds = self.inference(im)
        
        # 3. Postprocess
        results = self.postprocess(preds, im, source)
        
        return results

    @abstractmethod
    def preprocess(self, im: Any) -> Any:
        """Prepare input data for the model."""
        pass

    @abstractmethod
    def inference(self, im: Any) -> Any:
        """Run actual model inference."""
        pass

    @abstractmethod
    def postprocess(self, preds: Any, img: Any, orig_img: Any) -> Any:
        """Convert raw predictions to Results objects."""
        pass

    def setup_model(self, model: Any):
        """Assign the loaded model to the predictor."""
        self.model = model
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()
