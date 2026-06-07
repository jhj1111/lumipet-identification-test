from typing import Any, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from reid.engine.predictor import BasePredictor
from reid.data.transforms import get_transform

class ExtractorPredictor(BasePredictor):
    """
    Predictor for Feature Extraction.
    Handles image transformation and embedding extraction.
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.transform = get_transform(self.cfg.imgsz)

    def preprocess(self, im: Any) -> torch.Tensor:
        """Convert image to tensor and apply transforms."""
        if isinstance(im, (str, Path)):
            im = Image.open(im).convert('RGB')
        elif isinstance(im, np.ndarray):
            im = Image.fromarray(im)
            
        if isinstance(im, Image.Image):
            im = self.transform(im)
        
        # Add batch dimension and move to device
        return im.unsqueeze(0).to(self.device)

    def inference(self, im: torch.Tensor) -> torch.Tensor:
        """Run the backbone and projection layer."""
        with torch.no_grad():
            # self.model is ExtractorModel.model (a CombinedModel module)
            features = self.model(im)
            
            # L2 Normalize
            embedding = F.normalize(features, p=2, dim=1)
            
        return embedding

    def postprocess(self, preds: torch.Tensor, img: Any, orig_img: Any) -> np.ndarray:
        """Return flattened numpy array of the embedding."""
        return preds.cpu().numpy().flatten()
