from typing import Any, Optional
import numpy as np
from reid.engine.predictor import BasePredictor
from reid.engine.model import BaseModel
from reid.core.types import Results

class ReIdPredictor(BasePredictor):
    """
    Predictor that combines Detection, Feature Extraction, and Matching.
    """
    def __init__(self, detector: Any, extractor: Any, matcher: Any, config: Optional[Any] = None) -> None:
        super().__init__(config)
        self.detector_predictor = detector.predict
        self.extractor_predictor = extractor.predict
        self.matcher = matcher
        
        # Load DB into matcher
        embeddings, labels = extractor.store.get_all()
        if len(labels) > 0:
            # Check for dimension mismatch
            imgsz = extractor.cfg.imgsz
            dummy_emb = extractor.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))
            extractor_dim = dummy_emb.shape[0]
            db_dim = embeddings.shape[1]
            if extractor_dim != db_dim:
                raise ValueError(
                    f"Database embedding dimension ({db_dim}) does not match extractor output dimension ({extractor_dim}). "
                    f"Please delete your database file '{extractor.cfg.db_path}' and re-register the cats."
                )
            self.matcher.fit(embeddings, labels)
        else:
            print("Warning: Embedding database is empty.")

    def preprocess(self, im: Any) -> Any:
        return im

    def inference(self, im: Any) -> Results:
        """Detect bounding boxes, extract features, and compare with database."""
        results = self.detector_predictor(im)
        img_pixels = results.orig_img
        
        all_embeddings = []
        for box in results.boxes:
            crop = box.crop(img_pixels)
            if crop.size == 0:
                continue
                
            embedding = self.extractor_predictor(crop)
            all_embeddings.append(embedding)
            
            match_res = self.matcher.match(embedding)
            results.match_results.append(match_res)
            
        if all_embeddings:
            results.embeddings = np.vstack(all_embeddings)
            
        return results

    def postprocess(self, preds: Any, img: Any, orig_img: Any) -> Results:
        return preds


class ReIdModel(BaseModel):
    """
    Full Re-ID Pipeline Model.
    """
    def __init__(self, detector: Any, extractor: Any, matcher: Any, cfg: Optional[Any] = None) -> None:
        super().__init__(task="reid_pipeline", cfg=cfg)
        self.detector = detector
        self.extractor = extractor
        self.matcher = matcher

    def _load_model(self, weights: str) -> None:
        pass

    def _get_predictor(self) -> ReIdPredictor:
        return ReIdPredictor(self.detector, self.extractor, self.matcher, self.cfg)
