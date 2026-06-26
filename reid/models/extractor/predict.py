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
        self._embedding_dim = None

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
        
        im_tensor = im.unsqueeze(0).to(self.device)
        if getattr(self.cfg, "fp16", False) and "cuda" in str(self.device):
            im_tensor = im_tensor.half()
        return im_tensor

    def inference(self, im: torch.Tensor) -> torch.Tensor:
        """Extract L2 normalized projection embedding."""
        with torch.no_grad():
            features = self.model(im)
            embedding = F.normalize(features, p=2, dim=1)
        return embedding

    def postprocess(self, preds: torch.Tensor, img: Any, orig_img: Any) -> np.ndarray:
        """Format results to flattened numpy embedding array."""
        return preds.to(torch.float32).cpu().numpy().flatten()

    def preprocess_batch(self, im_list: list) -> torch.Tensor:
        """Convert each image to PIL, apply transform, stack them, send to device, and optionally convert to half."""
        tensors = []
        for im in im_list:
            if isinstance(im, (str, Path)):
                with Image.open(im) as img:
                    im = img.convert('RGB')
                    im.load()
            elif isinstance(im, np.ndarray):
                im = Image.fromarray(im)
            
            if isinstance(im, Image.Image):
                im = self.transform(im)
            
            tensors.append(im)
            
        batch_tensor = torch.stack(tensors, dim=0).to(self.device)
        
        if getattr(self.cfg, "fp16", False) and "cuda" in str(self.device):
            batch_tensor = batch_tensor.half()
            
        return batch_tensor

    def postprocess_batch(self, preds: torch.Tensor) -> np.ndarray:
        """Return preds as numpy array on CPU."""
        return preds.to(torch.float32).cpu().numpy()

    def predict_batch(self, im_list: list) -> np.ndarray:
        """Predict on a batch of images."""
        if not im_list:
            if hasattr(self, "_embedding_dim") and self._embedding_dim is not None:
                return np.empty((0, self._embedding_dim), dtype=np.float32)
            
            dim = 512
            if self.model is not None:
                try:
                    param = next(self.model.parameters())
                    device = param.device
                    dtype = param.dtype
                except StopIteration:
                    device = self.device
                    dtype = torch.float32
                
                dummy = torch.zeros((1, 3, self.cfg.imgsz, self.cfg.imgsz), device=device, dtype=dtype)
                try:
                    with torch.no_grad():
                        out = self.inference(dummy)
                        dim = out.shape[1]
                except Exception:
                    pass
            self._embedding_dim = dim
            return np.empty((0, dim), dtype=np.float32)
            
        im_prepped = self.preprocess_batch(im_list)
        preds = self.inference(im_prepped)
        self._embedding_dim = preds.shape[1]
        return self.postprocess_batch(preds)
