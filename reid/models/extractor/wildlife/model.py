import torch
import timm
import os
from typing import Any, Optional

from reid.engine.model import BaseModel
from reid.models.extractor.predict import ExtractorPredictor
from reid.models.extractor.embedding import EmbeddingStore

class WildlifeExtractorModel(BaseModel):
    """
    Feature Extractor Model wrapper using wildlife-tools and native timm backbones.
    """
    def __init__(self, model_path: Optional[str] = None, model_name: str = None, cfg=None) -> None:
        from reid.core.config import get_config
        cfg_inst = cfg or get_config()
        m_path = model_path or cfg_inst.extractor_weights
        m_name = model_name or cfg_inst.model_name
        
        super().__init__(m_path, task="reid", cfg=cfg_inst)
        self.model_name = m_name
        self.embedding_size = 0
        self._load_model(self.model_path)
        self.store = EmbeddingStore()

    def _load_model(self, weights: str) -> None:
        """Load raw timm backbone and infer features shape dynamically."""
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        
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

    def _get_predictor(self) -> ExtractorPredictor:
        """Return the shared predictor class."""
        predictor = ExtractorPredictor(self.cfg)
        predictor.setup_model(self.model)
        return predictor

    def _get_trainer(self) -> "WildlifeExtractorTrainer":
        """Return the training wrapper."""
        from reid.models.extractor.wildlife.train import WildlifeExtractorTrainer
        return WildlifeExtractorTrainer(self.cfg, model_instance=self)

    def _get_validator(self) -> "ExtractorValidator":
        """Return the validation wrapper."""
        from reid.models.extractor.val import ExtractorValidator
        return ExtractorValidator(self.cfg)

    def register(self, image: Any, label: str, verbose: bool = True) -> None:
        """Extract features and save to embedding store."""
        embedding = self.predict(image)
        self.store.add(embedding, label)
        if verbose:
            print(f"Registered cat: {label}")

    def save_db(self) -> None:
        """Save the database of embeddings."""
        self.store.save()
