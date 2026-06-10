import torch
import torch.nn as nn
import torch.optim as optim
from reid.engine.trainer import BaseTrainer
from reid.data.loader import CatDataLoader

class ExtractorTrainer(BaseTrainer):
    """
    Trainer implementation for ExtractorModel projection layer fine-tuning.
    """
    def __init__(self, cfg=None, model_instance=None) -> None:
        super().__init__(cfg)
        self.model_instance = model_instance
        self.classifier = None

    def get_model(self) -> nn.Module:
        """Activate projection weights training flags and return model."""
        self.model_instance.has_custom_weights = True
        self.model_instance.model.has_custom_weights = True
        return self.model_instance.model

    def get_dataloader(self) -> tuple:
        """Build and return data loaders from dataset_path."""
        loader = CatDataLoader(self.cfg.dataset_path, imgsz=self.cfg.imgsz)
        return loader.get_loaders(batch_size=self.cfg.batch_size, test_size=self.cfg.test_size)

    def setup(self) -> None:
        """Initialize classifier layer and AdamW optimizer."""
        super().setup()
        num_classes = len(self.train_loader.dataset.label_to_idx)
        self.classifier = nn.Linear(512, num_classes).to(self.device)
        
        # Optimize projection parameters + auxiliary classifier parameters
        self.optimizer = optim.AdamW(
            list(self.model.projection.parameters()) + list(self.classifier.parameters()),
            lr=self.cfg.lr
        )

    def compute_loss(self, model_outputs, targets) -> torch.Tensor:
        """Pass projection features through linear classifier and compute entropy."""
        logits = self.classifier(model_outputs)
        return self.criterion(logits, targets)

    def pre_epoch_hook(self) -> None:
        """Maintain frozen backbone in eval mode during training."""
        if hasattr(self.model, 'backbone'):
            self.model.backbone.eval()
        self.classifier.train()

    @torch.no_grad()
    def validate(self) -> float:
        """Evaluate current epoch accuracy on validation data."""
        self.model.eval()
        self.classifier.eval()
        correct = 0
        total = 0
        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            embeddings = self.model(imgs)
            logits = self.classifier(embeddings)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return 100. * correct / total

    def save_model(self) -> None:
        """Save projection layer weights checkpoint."""
        torch.save(self.model.projection.state_dict(), self.get_save_path())

    def get_save_path(self) -> str:
        """Return weight file save path."""
        return self.model_instance.model_path
