from typing import Any
import numpy as np
from reid.engine.predictor import BasePredictor
from reid.engine.model import BaseModel
from reid.core.types import Results

class ReIdPredictor(BasePredictor):
    """
    Predictor that combines Detection, Feature Extraction, and Matching.
    """
    def __init__(self, detector, extractor, matcher, config=None) -> None:
        super().__init__(config)
        self.detector_predictor = detector.predict
        self.extractor_predictor = extractor.predict
        self.matcher = matcher
        
        # Load DB into matcher
        embeddings, labels = extractor.store.get_all()
        if len(labels) > 0:
            # Check for dimension mismatch
            dummy_emb = extractor.predict(np.zeros((384, 384, 3), dtype=np.uint8))
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
        # Input to pipeline is the original image
        return im

    def inference(self, im: Any) -> Results:
        """
        1. Detect cats
        2. For each cat, extract embedding
        3. Match embedding against DB
        """
        # 1. Detection
        results = self.detector_predictor(im) # Returns Results object
        img_pixels = results.orig_img # Always a numpy array after detection
        
        # 2 & 3. Extraction & Matching
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
        # Inference already returns the Results object
        return preds

class ReIdModel(BaseModel):
    """
    Full Re-ID Pipeline Model.
    """
    def __init__(self, detector, extractor, matcher, cfg=None) -> None:
        super().__init__(task="reid_pipeline", cfg=cfg)
        self.detector = detector
        self.extractor = extractor
        self.matcher = matcher

    def _load_model(self, weights: str):
        pass # Underlying models are already loaded

    def _get_predictor(self) -> ReIdPredictor:
        return ReIdPredictor(self.detector, self.extractor, self.matcher, self.cfg)
