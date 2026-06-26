from abc import ABC, abstractmethod
from typing import Any, Optional, Union
from pathlib import Path
from reid.core.config import get_config

class BaseModel(ABC):
    """
    Abstract Base Class for all models.
    Delegates tasks to specialized Predictors, Trainers, and Validators.
    """
    def __init__(self, model_path: Optional[Union[str, Path]] = None, task: Optional[str] = None, cfg=None):
        self.model_path = model_path
        self.task = task
        self.cfg = cfg or get_config()
        self.model = None
        self.predictor = None
        self.trainer = None
        self.validator = None

    @abstractmethod
    def _load_model(self, weights: str):
        """Load the underlying model weights."""
        pass

    def predict(self, source: Any) -> Any:
        """Perform inference using the assigned predictor."""
        if self.predictor is None:
            self.predictor = self._get_predictor()
        return self.predictor(source)

    def predict_batch(self, source: Any) -> Any:
        """Perform batch inference using the assigned predictor."""
        if self.predictor is None:
            self.predictor = self._get_predictor()
        return self.predictor.predict_batch(source)

    @abstractmethod
    def _get_predictor(self):
        """Return an instance of a task-specific Predictor."""
        pass

    def train(self):
        """Start training using the assigned trainer."""
        if self.trainer is None:
            self.trainer = self._get_trainer()
        return self.trainer.train()

    def _get_trainer(self) -> "BaseTrainer":
        """Return an instance of a task-specific Trainer."""
        raise NotImplementedError("Trainer not implemented for this model.")

    def val(self, pipeline=None):
        """Start validation using the assigned validator."""
        if self.validator is None:
            self.validator = self._get_validator()
        return self.validator.validate(pipeline=pipeline)

    def _get_validator(self) -> "BaseValidator":
        """Return an instance of a task-specific Validator."""
        raise NotImplementedError("Validator not implemented for this model.")
