from typing import Any, List
import numpy as np
import torch
from reid.engine.predictor import BasePredictor
from reid.core.types import Results, BBox

class YoloPredictor(BasePredictor):
    """
    YOLOv8 Predictor following the BasePredictor interface.
    """
    def preprocess(self, im: Any) -> Any:
        # Ultralytics handles most preprocessing internally, 
        # but we can do custom sizing if needed.
        return im

    def inference(self, im: Any) -> Any:
        # model is ultralytics.YOLO instance
        # results = self.model(im, conf=self.cfg.conf, iou=self.cfg.iou)
        # Note: self.model is actually the YOLO object, we call its internal model or use it directly
        return self.model(im, conf=self.cfg.conf, iou=self.cfg.iou, verbose=False)

    def postprocess(self, preds: Any, img: Any, orig_img: Any) -> Results:
        """Convert ultralytics Results to our custom Results object."""
        # preds is a list of ultralytics.engine.results.Results
        ultra_res = preds[0]
        
        results = Results(orig_img=ultra_res.orig_img, path=ultra_res.path)
        
        for box in ultra_res.boxes:
            b = box.xyxy[0].cpu().numpy()
            bbox = BBox(
                x1=float(b[0]), y1=float(b[1]), 
                x2=float(b[2]), y2=float(b[3]),
                conf=float(box.conf[0]),
                cls=int(box.cls[0])
            )
            results.boxes.append(bbox)
            
        return results
