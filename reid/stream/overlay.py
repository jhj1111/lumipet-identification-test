import cv2
import numpy as np
from reid.core.types import Results

class Renderer:
    """
    Renders BBoxes and labels on the image.
    """
    def __init__(self, thickness=3, font_scale=1.0, dev=True):
        self.thickness = thickness
        self.font_scale = font_scale
        self.dev = dev

    def draw(self, results: Results) -> np.ndarray:
        img = results.orig_img.copy()
        
        for i, box in enumerate(results.boxes):
            # 1. Draw BBox
            color = (0, 255, 0) # Green
            cv2.rectangle(img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, self.thickness)
            
            # 2. Draw Label (MatchResult)
            if i < len(results.match_results):
                match = results.match_results[i]
                
                if self.dev:
                    track_prefix = f"ID:{box.track_id} " if box.track_id is not None else ""
                    label = f"{track_prefix}{match.cat_id} {match.similarity:.2f}"
                else:
                    label = f"{match.cat_id}"
                
                # Draw background box and text
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2)
                # Draw solid background rectangle
                cv2.rectangle(img, (int(box.x1), int(box.y1) - h - 12), (int(box.x1) + w, int(box.y1)), color, -1)
                # Draw black bold text
                cv2.putText(img, label, (int(box.x1), int(box.y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 0, 0), 2)
                
        return img
