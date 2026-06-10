from typing import Any, Optional
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
    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config)
        self.transform = get_transform(self.cfg.imgsz)

    def preprocess(self, im: Any) -> torch.Tensor:
        """Convert image to tensor and apply transforms."""
        if isinstance(im, (str, Path)):
            with Image.open(im) as img:
                im = img.convert('RGB')
                im.load()
        elif isinstance(im, np.ndarray):
            im = Image.fromarray(im)
            
        if isinstance(im, Image.Image):
            im = self.transform(im)
        
        return im.unsqueeze(0).to(self.device)

    def inference(self, im: torch.Tensor) -> torch.Tensor:
        """Extract L2 normalized projection embedding."""
        with torch.no_grad():
            features = self.model(im)
            embedding = F.normalize(features, p=2, dim=1)
        return embedding

    def postprocess(self, preds: torch.Tensor, img: Any, orig_img: Any) -> np.ndarray:
        """Format results to flattened numpy embedding array."""
        return preds.cpu().numpy().flatten()
