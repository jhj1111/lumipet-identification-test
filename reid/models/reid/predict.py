from typing import Any, Optional
import numpy as np
from reid.engine.predictor import BasePredictor
from reid.core.types import Results, MatchResult
from reid.core.tracker import TrackStateManager, TrackState
from reid.core.filters import ImageQualityFilter

class ReIdPredictor(BasePredictor):
    """
    Predictor that combines Detection, Feature Extraction, and Matching.
    """
    def __init__(self, detector: Any, extractor: Any, matcher: Any, config: Optional[Any] = None) -> None:
        super().__init__(config)
        self.detector_predictor = detector.predict
        self.extractor_predictor = extractor.predict
        self.matcher = matcher
        self.track_state_manager = TrackStateManager()
        
        # Instantiate Quality Filter
        self.quality_filter = ImageQualityFilter(
            min_w=getattr(self.cfg, "min_bbox_width", 32),
            min_h=getattr(self.cfg, "min_bbox_height", 32),
            blur_th=getattr(self.cfg, "blur_threshold", 10.0)
        )
        self.th_candidate = getattr(self.cfg, "threshold_candidate", 0.70)
        self.th_lock = getattr(self.cfg, "threshold_lock", 0.85)
        self.th_hysteresis = getattr(self.cfg, "threshold_hysteresis", 0.55)
        self.candidate_interval = getattr(self.cfg, "candidate_interval", 10)
        self.lock_interval = getattr(self.cfg, "lock_interval", 60)
        
        # Initialize extractor output dimension
        imgsz = extractor.cfg.imgsz
        dummy_emb = extractor.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))
        self.extractor_dim = dummy_emb.shape[0]
        
        # Load DB into matcher
        embeddings, labels = extractor.store.get_all()
        if len(labels) > 0:
            # Check for dimension mismatch
            db_dim = embeddings.shape[1]
            if self.extractor_dim != db_dim:
                raise ValueError(
                    f"Database embedding dimension ({db_dim}) does not match extractor output dimension ({self.extractor_dim}). "
                    f"Please delete your database file '{extractor.cfg.db_path}' and re-register the cats."
                )
            self.matcher.fit(embeddings, labels)
        else:
            print("Warning: Embedding database is empty.")

    def reset(self) -> None:
        """Reset the track state cache."""
        self.track_state_manager.tracks.clear()

    def preprocess(self, im: Any) -> Any:
        return im

    def inference(self, im: Any) -> Results:
        """Detect bounding boxes, crop and filter by quality, then apply state-machine Re-ID."""
        results = self.detector_predictor(im)
        img_pixels = results.orig_img
        
        all_embeddings = []
        for box in results.boxes:
            track_id = box.track_id
            crop = box.crop(img_pixels)
            
            # 1. Quality Filter check
            if not self.quality_filter.is_valid(crop):
                # Fallback: reuse cached match if exists, else Unknown
                cached = self.track_state_manager.tracks.get(track_id) if track_id is not None else None
                if cached is not None and cached.match_result is not None:
                    results.match_results.append(cached.match_result)
                    all_embeddings.append(cached.get_mean_embedding())
                else:
                    results.match_results.append(MatchResult(cat_id="Unknown", similarity=0.0, is_known=False))
                    all_embeddings.append(np.zeros(self.extractor_dim, dtype=np.float32))
                continue
            
            # 2. State Machine scheduling check
            run_matching = True
            state_obj = None
            if track_id is not None:
                if track_id not in self.track_state_manager.tracks:
                    self.track_state_manager.tracks[track_id] = TrackState(track_id)
                state_obj = self.track_state_manager.tracks[track_id]
                run_matching = state_obj.should_match(self.candidate_interval, self.lock_interval)
            
            if not run_matching and state_obj is not None and state_obj.match_result is not None:
                # Cache hit: Retrieve mean embedding and reuse match result
                self.track_state_manager.get_match(track_id)
                results.match_results.append(state_obj.match_result)
                all_embeddings.append(state_obj.get_mean_embedding())
                state_obj.frame_count += 1
                continue
                
            # Cache miss or forced re-matching frame: extract new crop features
            embedding = self.extractor_predictor(crop)
            
            # Aggregate mean embedding (current embedding + history) for matching query
            historical_embs = [obs["embedding"] for obs in state_obj.observations] if state_obj else []
            all_embs = historical_embs + [embedding]
            mean_emb = np.mean(all_embs, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 1e-6:
                mean_emb = mean_emb / norm
            
            # Run matching
            match_res = self.matcher.match(mean_emb)
            
            # Handle state transitions
            if state_obj is not None:
                state_obj.update_state(match_res.similarity, self.th_candidate, self.th_lock, self.th_hysteresis)
                state_obj.add_observation(embedding, match_res, is_match_run=True)
            
            results.match_results.append(match_res)
            all_embeddings.append(mean_emb)
            
        if all_embeddings:
            results.embeddings = np.vstack(all_embeddings)
            
        return results

    def postprocess(self, preds: Any, img: Any, orig_img: Any) -> Results:
        return preds

