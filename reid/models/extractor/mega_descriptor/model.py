import torch
import torch.nn as nn
import timm
import os
from typing import Any, Optional

from reid.engine.model import BaseModel
from reid.models.extractor.predict import ExtractorPredictor
from reid.models.extractor.embedding import EmbeddingStore

class ExtractorModel(BaseModel):
    """
    Feature Extractor Model wrapper using MegaDescriptor.
    """
    def __init__(self, model_path: Optional[str] = None, model_name: str = None, cfg=None) -> None:
        # Prioritize cfg params if not explicitly passed
        from reid.core.config import get_config
        cfg_inst = cfg or get_config()
        m_path = model_path or cfg_inst.extractor_weights
        m_name = model_name or cfg_inst.model_name
        
        super().__init__(m_path, task="reid", cfg=cfg_inst)
        self.model_name = m_name
        self._load_model(self.model_path)
        self.store = EmbeddingStore()

    def _load_model(self, weights: str):
        backbone = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        for param in backbone.parameters():
            param.requires_grad = False
        
        num_features = getattr(backbone, 'num_features', 1536)
        projection = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        if weights and os.path.exists(weights):
            print(f"Loading weights from {weights}")
            try:
                state_dict = torch.load(weights, map_location="cpu")
                projection.load_state_dict(state_dict)
                self.has_custom_weights = True
            except Exception as e:
                print(f"Error loading weights: {e}")
                self.has_custom_weights = False
        else:
            self.has_custom_weights = False

        class CombinedModel(nn.Module):
            def __init__(self, backbone, projection, has_custom_weights):
                super().__init__()
                self.backbone = backbone
                self.projection = projection
                self.has_custom_weights = has_custom_weights
            
            def forward(self, x):
                with torch.no_grad():
                    features = self.backbone(x)
                if self.has_custom_weights:
                    return self.projection(features)
                return features

        self.model = CombinedModel(backbone, projection, self.has_custom_weights)

    def _get_predictor(self) -> ExtractorPredictor:
        predictor = ExtractorPredictor(self.cfg)
        predictor.setup_model(self.model)
        return predictor

    def _get_trainer(self) -> "ExtractorTrainer":
        from reid.models.extractor.mega_descriptor.train import ExtractorTrainer
        return ExtractorTrainer(self.cfg, model_instance=self)

    def _get_validator(self) -> "ExtractorValidator":
        from reid.models.extractor.val import ExtractorValidator
        return ExtractorValidator(self.cfg)

    def register(self, image: Any, label: str, verbose: bool = True):
        embedding = self.predict(image)
        self.store.add(embedding, label)
        if verbose:
            print(f"Registered cat: {label}")

    def save_db(self):
        self.store.save()
