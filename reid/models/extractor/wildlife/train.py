from typing import Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from reid.engine.trainer import BaseTrainer
from reid.data.loader import CatDataLoader
from wildlife_tools.train.objective import ArcFaceLoss

class WildlifeExtractorTrainer(BaseTrainer):
    """
    Trainer implementation for WildlifeExtractorModel fine-tuning using ArcFace loss.
    """
    def __init__(self, cfg: Optional[Any] = None, model_instance: Optional[Any] = None) -> None:
        super().__init__(cfg)
        self.model_instance = model_instance

    def get_model(self) -> nn.Module:
        """Return model backbone for training."""
        return self.model_instance.model

    def get_dataloader(self) -> tuple:
        """Build and return data loaders from dataset_path."""
        loader = CatDataLoader(self.cfg.dataset_path, imgsz=self.cfg.imgsz)
        return loader.get_loaders(batch_size=self.cfg.batch_size, test_size=self.cfg.test_size)

    def setup(self) -> None:
        """Initialize ArcFaceLoss and AdamW optimizer."""
        super().setup()
        num_classes = len(self.train_loader.dataset.label_to_idx)
        embedding_size = self.model_instance.embedding_size
        
        # Initialize ArcFaceLoss
        self.criterion = ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
            margin=getattr(self.cfg, "arcface_margin", 0.5),
            scale=getattr(self.cfg, "arcface_scale", 64.0)
        ).to(self.device)
        
        # Optimize model parameters and loss parameters (trainable W weights)
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=self.cfg.lr
        )

    def compute_loss(self, model_outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace angular margin loss."""
        return self.criterion(model_outputs, targets)

    def pre_epoch_hook(self) -> None:
        """Set model to training mode."""
        self.model.train()

    @torch.no_grad()
    def validate(self) -> float:
        """Evaluate current epoch accuracy on validation data."""
        self.model.eval()
        correct = 0
        total = 0
        
        # Retrieve ArcFace weights (shape: [embedding_size, num_classes])
        arcface_w = self.criterion.loss.W
        
        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            embeddings = self.model(imgs)
            
            # Predict by finding maximum cosine similarity with class weight vectors
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            norm_w = F.normalize(arcface_w, p=2, dim=0)
            logits = torch.matmul(norm_embeddings, norm_w)
            
            _, predicted = logits.max(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return 100. * correct / total

    def save_model(self) -> None:
        """Save fine-tuned backbone weights checkpoint."""
        torch.save(self.model.state_dict(), self.get_save_path())

    def get_save_path(self) -> str:
        """Return weight file save path."""
        return self.model_instance.model_path
