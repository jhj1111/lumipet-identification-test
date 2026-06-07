import cv2
import numpy as np
from reid.core.types import Results

class Renderer:
    """
    Renders BBoxes and labels on the image.
    """
    def __init__(self, thickness=2, font_scale=0.8):
        self.thickness = thickness
        self.font_scale = font_scale

    def draw(self, results: Results) -> np.ndarray:
        img = results.orig_img.copy()
        
        for i, box in enumerate(results.boxes):
            # 1. Draw BBox
            color = (0, 255, 0) # Green
            cv2.rectangle(img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, self.thickness)
            
            # 2. Draw Label (MatchResult)
            if i < len(results.match_results):
                match = results.match_results[i]
                label = f"{match.cat_id} {match.similarity:.2f}"
                
                # Simple background for text
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
                cv2.rectangle(img, (int(box.x1), int(box.y1) - h - 10), (int(box.x1) + w, int(box.y1)), color, -1)
                cv2.putText(img, label, (int(box.x1), int(box.y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 0, 0), 1)
                
        return img
