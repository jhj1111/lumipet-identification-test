from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np
import torch

@dataclass
class BBox:
    """Bounding Box representation (x1, y1, x2, y2, confidence, class_id)"""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0
    cls: int = 0

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2, self.conf, self.cls]

    def crop(self, img: np.ndarray) -> np.ndarray:
        """Crop the image based on BBox coordinates."""
        return img[int(self.y1):int(self.y2), int(self.x1):int(self.x2)]

@dataclass
class MatchResult:
    """Result of Re-ID matching for a single detected object."""
    cat_id: str
    similarity: float
    is_known: bool = True

@dataclass
class Results:
    """
    Unified result object containing detection and Re-ID information.
    Similar to Ultralytics Results class.
    """
    orig_img: np.ndarray = field(repr=False)
    path: str
    boxes: List[BBox] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None  # Shape: (N, D)
    match_results: List[MatchResult] = field(default_factory=list)

    def __len__(self):
        return len(self.boxes)

    @property
    def names(self) -> List[str]:
        """Return list of matched cat IDs."""
        return [m.cat_id for m in self.match_results]

    def plot(self):
        """Placeholder for visualization logic (will use overlay/renderer)."""
        # TODO: Implement plotting logic using OpenCV or PIL
        pass
