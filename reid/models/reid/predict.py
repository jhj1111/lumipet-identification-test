import time
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
        self.extractor = extractor
        self.extractor_predictor = extractor.predict
        self.matcher = matcher
        max_tracks = getattr(self.cfg, "max_tracks", 1000)
        self.track_state_manager = TrackStateManager(max_tracks=max_tracks)
        
        # Instantiate Quality Filter
        self.quality_filter = ImageQualityFilter(
            min_w=getattr(self.cfg, "min_bbox_width", 32),
            min_h=getattr(self.cfg, "min_bbox_height", 32),
            blur_th=getattr(self.cfg, "blur_threshold", 10.0)
        )
        self.th_candidate = getattr(self.cfg, "threshold_candidate", 0.70)
        self.th_lock = getattr(self.cfg, "threshold_lock", 0.85)
        self.th_hysteresis = getattr(self.cfg, "threshold_hysteresis", 0.55)
        self.candidate_interval = getattr(self.cfg, "candidate_interval", 20)
        self.lock_interval = getattr(self.cfg, "lock_interval", 60)
        self.unknown_interval = getattr(self.cfg, "unknown_interval", 10)
        
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
        """Detect bounding boxes, crop and filter by quality, then apply state-machine Re-ID in batch."""
        with self.profiler.profile("detection"):
            results = self.detector_predictor(im)
        img_pixels = results.orig_img
        
        n_boxes = len(results.boxes)
        box_match_results = [None] * n_boxes
        box_embeddings = [None] * n_boxes
        
        # Pass 1: Collect crops needing extraction
        needs_extraction = []  # List of tuples: (idx, crop, track_id, state_obj)
        
        for idx, box in enumerate(results.boxes):
            track_id = box.track_id
            crop = box.crop(img_pixels)
            
            # 1. Quality Filter check
            with self.profiler.profile("quality_filter"):
                is_valid = self.quality_filter.is_valid(crop)
                
            if not is_valid:
                # Fallback: reuse cached match if exists, else Unknown
                cached = self.track_state_manager.tracks.get(track_id) if track_id is not None else None
                if cached is not None and cached.match_result is not None:
                    box_match_results[idx] = cached.match_result
                    box_embeddings[idx] = cached.get_mean_embedding()
                else:
                    box_match_results[idx] = MatchResult(cat_id="Unknown", similarity=0.0, is_known=False)
                    box_embeddings[idx] = np.zeros(self.extractor_dim, dtype=np.float32)
                continue
            
            # 2. State Machine scheduling check
            run_matching = True
            state_obj = None
            if track_id is not None:
                if track_id not in self.track_state_manager.tracks:
                    if len(self.track_state_manager.tracks) >= self.track_state_manager.max_tracks:
                        # Evict oldest track if at capacity
                        first_key = next(iter(self.track_state_manager.tracks))
                        self.track_state_manager.tracks.pop(first_key)
                    self.track_state_manager.tracks[track_id] = TrackState(track_id)
                state_obj = self.track_state_manager.tracks[track_id]
                run_matching = state_obj.should_match(self.candidate_interval, self.lock_interval, self.unknown_interval)
            
            if not run_matching and state_obj is not None and state_obj.match_result is not None:
                # Cache hit: Retrieve mean embedding and reuse match result
                self.track_state_manager.get_match(track_id)
                box_match_results[idx] = state_obj.match_result
                box_embeddings[idx] = state_obj.get_mean_embedding()
                state_obj.frame_count += 1
                continue
                
            # Cache miss or forced re-matching frame: queue for extraction
            needs_extraction.append((idx, crop, track_id, state_obj))
            
        # Pass 2: Execute batch extraction and matching
        if needs_extraction:
            crops = [item[1] for item in needs_extraction]
            with self.profiler.profile("extractor"):
                embeddings = self.extractor.predict_batch(crops)
                
            for i, (idx, crop, track_id, state_obj) in enumerate(needs_extraction):
                embedding = embeddings[i]
                
                # Aggregate mean embedding (current embedding + history) for matching query
                historical_embs = [obs["embedding"] for obs in state_obj.observations] if state_obj is not None else []
                all_embs = historical_embs + [embedding]
                mean_emb = np.mean(all_embs, axis=0)
                norm = np.linalg.norm(mean_emb)
                if norm > 1e-6:
                    mean_emb = mean_emb / norm
            
                # Run matching
                with self.profiler.profile("matcher"):
                    match_res = self.matcher.match(mean_emb)
            
                # Handle state transitions
                if state_obj is not None:
                    state_obj.update_state(match_res.similarity, self.th_candidate, self.th_lock, self.th_hysteresis)
                    state_obj.add_observation(embedding, match_res, is_match_run=True)
            
                box_match_results[idx] = match_res
                box_embeddings[idx] = mean_emb
            
        results.match_results = box_match_results
        if any(emb is not None for emb in box_embeddings):
            results.embeddings = np.vstack(box_embeddings)
            
        return results

    def postprocess(self, preds: Any, img: Any, orig_img: Any) -> Results:
        return preds


