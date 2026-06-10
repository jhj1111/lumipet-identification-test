from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn
from tqdm import tqdm
from reid.core.config import get_config

class BaseTrainer(ABC):
    """
    Abstract Base Class for model training using Template Method.
    Encapsulates standard training loop boilerplates.
    """
    def __init__(self, cfg=None) -> None:
        self.cfg = cfg or get_config()
        self.device = self.cfg.device
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the neural network model to be trained."""
        pass

    @abstractmethod
    def get_dataloader(self) -> tuple:
        """Return (train_loader, val_loader) loaders."""
        pass

    @abstractmethod
    def compute_loss(self, model_outputs, targets) -> torch.Tensor:
        """Compute model loss output."""
        pass

    def setup(self) -> None:
        """Initialize training components: model, loaders, loss, and optimizer."""
        self.model = self.get_model()
        self.model.to(self.device)
        self.train_loader, self.val_loader = self.get_dataloader()
        self.criterion = nn.CrossEntropyLoss()

    def train(self) -> None:
        """Execute the main training epochs loop (Template Method)."""
        self.setup()
        best_acc = 0.0
        epochs = int(self.cfg.epochs)

        for epoch in range(epochs):
            self.model.train()
            self.pre_epoch_hook()
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                loss = self.train_step(batch)
                pbar.set_postfix({'loss': f"{loss:.4f}"})

            val_acc = self.validate()
            print(f"Validation Accuracy: {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model()
                print(f"Saved best model weights to {self.get_save_path()}")

    def train_step(self, batch) -> float:
        """Perform single forward, loss computation, backpropagation, and weight update."""
        self.optimizer.zero_grad()
        imgs, labels = batch
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        
        outputs = self.model(imgs)
        loss = self.compute_loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def pre_epoch_hook(self) -> None:
        """Optional hook executed at the beginning of each epoch."""
        pass

    @abstractmethod
    def validate(self) -> float:
        """Calculate classification accuracy on validation loader."""
        pass

    @abstractmethod
    def save_model(self) -> None:
        """Save model state dict to checkpoint path."""
        pass

    @abstractmethod
    def get_save_path(self) -> str:
        """Get target checkpoint saving path."""
        pass


class BaseValidator(ABC):
    """
    Abstract Base Class for model validation.
    """
    def __init__(self, cfg=None) -> None:
        self.cfg = cfg or get_config()

    @abstractmethod
    def validate(self, pipeline=None) -> Any:
        """Perform pipeline validation evaluation."""
        pass
