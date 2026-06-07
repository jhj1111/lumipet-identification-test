from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

class BaseModel(ABC):
    """
    Abstract Base Class for all models (YOLO, Extractor, Pipeline).
    Acts as a Facade that delegates tasks to specialized Predictors, Trainers, etc.
    """
    def __init__(self, model_path: Optional[Union[str, Path]] = None, task: Optional[str] = None):
        self.model_path = model_path
        self.task = task
        self.model = None  # Underlying model (e.g., torch.nn.Module)
        self.predictor = None
        self.trainer = None
        self.validator = None

    @abstractmethod
    def _load_model(self, weights: str):
        """Load the underlying model weights."""
        pass

    def predict(self, source: Any, **kwargs) -> Any:
        """Perform inference using the assigned predictor."""
        if self.predictor is None:
            self.predictor = self._get_predictor(**kwargs)
        return self.predictor(source, **kwargs)

    @abstractmethod
    def _get_predictor(self, **kwargs):
        """Return an instance of a task-specific Predictor."""
        pass

    def train(self, data: Any, **kwargs):
        """Start training using the assigned trainer."""
        if self.trainer is None:
            self.trainer = self._get_trainer(**kwargs)
        return self.trainer.train(data, **kwargs)

    def _get_trainer(self, **kwargs):
        """Return an instance of a task-specific Trainer."""
        raise NotImplementedError("Trainer not implemented for this model.")

    def val(self, data: Any, **kwargs):
        """Start validation using the assigned validator."""
        if self.validator is None:
            self.validator = self._get_validator(**kwargs)
        return self.validator.validate(data, **kwargs)

    def _get_validator(self, **kwargs):
        """Return an instance of a task-specific Validator."""
        raise NotImplementedError("Validator not implemented for this model.")
