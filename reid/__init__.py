__version__ = '0.1.0'
__author__ = 'jhj1111'

import importlib
from typing import TYPE_CHECKING

MODELS = (
    "ReIdModel",
    "YoloModel",
    "MegaDesExtractorModel", "WildlifeExtractorModel",
    "KnnMatcher", "FaissMatcher"
)

__all__ = (
    "__version__",
    "__author__",
    *MODELS,
)

if TYPE_CHECKING:
    # Enable hints for type checkers
    from reid.models import ReIdModel, YoloModel, MegaDesExtractorModel, WildlifeExtractorModel

def __getattr__(name: str):
    """Lazy-import model classes on first access."""
    if name in MODELS:
        return getattr(importlib.import_module("reid.models"), name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__():
    """Extend dir() to include lazily available model names for IDE autocompletion."""
    return sorted(set(globals()) | set(MODELS))

if __name__ == "__main__":
    print(__version__)
