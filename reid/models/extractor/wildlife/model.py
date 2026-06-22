import os
from typing import Any, Optional

import timm
import torch

from reid.models.extractor.embedding import EmbeddingStore
from reid.models.extractor.model import ExtractorModel
from reid.models.extractor.predict import ExtractorPredictor


class WildlifeExtractorModel(ExtractorModel):
    """
    Feature Extractor Model wrapper using wildlife-tools and native timm backbones.
    """
    def __init__(
            self,
            model_path: Optional[str] = None,
            model_name: Optional[str] = None,
            cfg=None
        ) -> None:
        from reid.core.config import get_config
        cfg_inst = cfg or get_config()
        m_path = model_path or cfg_inst.extractor_weights
        m_name = model_name or cfg_inst.model_name
        
        super().__init__(model_path=m_path, model_name=m_name)
        self.embedding_size = 0
        self._load_model(self.model_path)

    def _load_model(self, weights: str) -> None:
        """Load raw timm backbone and infer features shape dynamically."""
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        
        # Adjust imgsz dynamically based on model's default expected input size
        if hasattr(self.model, 'default_cfg') and 'input_size' in self.model.default_cfg:
            model_imgsz = self.model.default_cfg['input_size'][1]
            if self.cfg.imgsz != model_imgsz:
                print(f"Info: Adjusting configuration imgsz from {self.cfg.imgsz} to {model_imgsz} to match model requirements.")
                self.cfg.imgsz = model_imgsz
        
        # Infer embedding size dynamically using dummy input
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.cfg.imgsz, self.cfg.imgsz)
            dummy_out = self.model(dummy)
            self.embedding_size = dummy_out.shape[1]
            
        if weights and os.path.exists(weights):
            print(f"Loading weights from {weights}")
            try:
                state_dict = torch.load(weights, map_location="cpu")
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading weights: {e}")

    def _get_trainer(self) -> "WildlifeExtractorTrainer":
        """Return the training wrapper."""
        from reid.models.extractor.wildlife.train import WildlifeExtractorTrainer
        return WildlifeExtractorTrainer(self.cfg, model_instance=self)

    def _get_validator(self) -> "ExtractorValidator":
        """Return the validation wrapper."""
        from reid.models.extractor.val import ExtractorValidator
        return ExtractorValidator(self.cfg)