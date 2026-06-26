from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from reid.core.config import get_config
from reid.core.types import MatchResult

class BaseMatcher(ABC):
    """
    Abstract Base Class for Embedding Matchers.
    """
    def __init__(self, threshold: float = 0.7, cfg = None):
        self.cfg = cfg or get_config()
        self.threshold = threshold
        self.labels = []
        self.embeddings = None

    @abstractmethod
    def fit(self, embeddings: np.ndarray, labels: List[str]):
        """Load embeddings and labels into the matcher."""
        pass

    @abstractmethod
    def match(self, query_embedding: np.ndarray) -> MatchResult:
        """Find the best match for a query embedding."""
        pass

    def match_batch(self, query_embeddings: np.ndarray) -> List[MatchResult]:
        """Match a batch of query embeddings."""
        return [self.match(q) for q in query_embeddings]
