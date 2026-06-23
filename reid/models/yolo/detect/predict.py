from typing import Any, List, Optional
from reid.engine.predictor import BasePredictor
from reid.core.types import Results, BBox

class YoloPredictor(BasePredictor):
    """
    YOLOv8 Predictor following the BasePredictor interface.
    """
    def preprocess(self, im: Any) -> Any:
        return im

    def inference(self, im: Any) -> Any:
        if self.cfg.track:
            return self.model.track(
                im,
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                persist=True,
                tracker=self.cfg.tracker,
                verbose=False
            )
        else:
            return self.model(im, conf=self.cfg.conf, iou=self.cfg.iou, verbose=False)

    def postprocess(self, preds: Any, img: Any, orig_img: Any, target_id: Optional[List[int]] = [15]) -> Optional[Results]:
        """Convert ultralytics Results object to custom Results representation."""
        ultra_res = preds[0]
        # if ultra_res.boxes.cls[0] not in target_id:
        #     return None
        results = Results(orig_img=ultra_res.orig_img, path=ultra_res.path)
        
        # Optimize CPU transfers by copying all boxes to CPU at once
        cpu_boxes = ultra_res.boxes.cpu() if hasattr(ultra_res.boxes, 'cpu') else ultra_res.boxes
        
        for box in cpu_boxes:
            if box.cls not in target_id: continue
            b = box.xyxy[0].numpy() if hasattr(box.xyxy[0], 'numpy') else box.xyxy[0]
            track_id = int(box.id[0].item()) if box.id is not None else None
            bbox = BBox(
                x1=float(b[0]), y1=float(b[1]), 
                x2=float(b[2]), y2=float(b[3]),
                conf=float(box.conf[0]),
                cls=int(box.cls[0]),
                track_id=track_id
            )
            results.boxes.append(bbox)
            
        return results

