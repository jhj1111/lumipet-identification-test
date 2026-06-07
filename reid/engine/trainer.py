from abc import ABC, abstractmethod
from typing import Any

class BaseTrainer(ABC):
    """Abstract Base Class for Model Training."""
    def __init__(self, config=None):
        self.cfg = config

    @abstractmethod
    def train(self, data: Any, **kwargs):
        pass

class BaseValidator(ABC):
    """Abstract Base Class for Model Validation."""
    def __init__(self, config=None):
        self.cfg = config

    @abstractmethod
    def validate(self, data: Any, **kwargs):
        pass
