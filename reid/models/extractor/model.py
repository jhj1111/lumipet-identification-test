import torch
import torch.nn as nn
import timm
import os
from pathlib import Path
from typing import Any, Optional, Union
from PIL import Image

from reid.engine.model import BaseModel
from reid.models.extractor.predict import ExtractorPredictor
from reid.models.extractor.embedding import EmbeddingStore

class ExtractorModel(BaseModel):
    """
    Feature Extractor Model using MegaDescriptor.
    """
    def __init__(self, model_path: Optional[str] = "weights/best_projection.pth", 
                 model_name: str = "hf-hub:BVRA/MegaDescriptor-L-384"):
        super().__init__(model_path, task="reid")
        self.model_name = model_name
        self._load_model(model_path)
        self.store = EmbeddingStore() # Default store

    def _load_model(self, weights: str):
        # 1. Load Backbone
        backbone = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        
        # 2. Define Projection Layer
        projection = nn.Sequential(
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # 3. Load weights if available
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

        # 4. Combine into a single module for the Predictor
        class CombinedModel(nn.Module):
            def __init__(self, backbone, projection, has_custom_weights):
                super().__init__()
                self.backbone = backbone
                self.projection = projection
                self.has_custom_weights = has_custom_weights
            
            def forward(self, x):
                features = self.backbone(x)
                if self.has_custom_weights:
                    return self.projection(features)
                return features

        self.model = CombinedModel(backbone, projection, self.has_custom_weights)

    def _get_predictor(self, **kwargs):
        predictor = ExtractorPredictor()
        predictor.setup_model(self.model)
        return predictor

    def _get_trainer(self, **kwargs):
        from reid.models.extractor.train import ExtractorTrainer
        return ExtractorTrainer()

    def _get_validator(self, **kwargs):
        from reid.models.extractor.val import ExtractorValidator
        return ExtractorValidator()

    def register(self, image: Any, label: str):
        """
        Extract embedding and save to store.
        """
        embedding = self.predict(image)
        self.store.add(embedding, label)
        print(f"Registered cat: {label}")

    def save_db(self):
        """Save the embedding store to disk."""
        self.store.save()
