from abc import ABC, abstractmethod
from typing import Any

from reid.core.config import get_config

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
