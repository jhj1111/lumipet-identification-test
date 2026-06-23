import cv2
import numpy as np

class ImageQualityFilter:
    """Applies size and blur checking on a cropped region."""
    def __init__(self, min_w: int = 32, min_h: int = 32, blur_th: float = 10.0) -> None:
        self.min_w = min_w
        self.min_h = min_h
        self.blur_th = blur_th

    def is_valid(self, crop: np.ndarray) -> bool:
        """Return True if crop passes all quality filters, else False."""
        if crop is None or crop.size == 0:
            return False
        
        # Size check
        h, w = crop.shape[:2]
        if w < self.min_w or h < self.min_h:
            return False
            
        # Blur check (Laplacian variance)
        if self.blur_th > 0.0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if variance < self.blur_th:
                return False
                
        return True
