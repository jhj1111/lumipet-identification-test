from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from reid.core.types import MatchResult

class TrackState:
    """Manages the state and history of an individual tracked object."""
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.observations: List[Dict[str, Any]] = []  # List of dicts: {"embedding": np.ndarray, "similarity": float, "frame_idx": int}
        self.match_result: Optional[MatchResult] = None
        self.frame_count = 0
        self.state = "Unknown"  # "Unknown", "Candidate", "Locked"
        self.last_match_frame = -9999
        
    def should_match(self, candidate_interval: int, lock_interval: int) -> bool:
        """Determine if a re-matching step is required."""
        if self.state == "Unknown":
            return True
        elif self.state == "Candidate":
            return (self.frame_count - self.last_match_frame) >= candidate_interval
        elif self.state == "Locked":
            return (self.frame_count - self.last_match_frame) >= lock_interval
        return True

    def update_state(self, similarity: float, th_candidate: float, th_lock: float, th_hysteresis: float):
        """Perform state transition based on similarity thresholds."""
        old_state = self.state
        if similarity >= th_lock:
            self.state = "Locked"
        elif similarity >= th_candidate:
            self.state = "Candidate"
        else:
            self.state = "Unknown"
            
        # Hysteresis unlock
        if old_state == "Locked" and similarity < th_hysteresis:
            self.state = "Unknown"

    def add_observation(self, embedding: np.ndarray, match_res: MatchResult, is_match_run: bool = True):
        sim = match_res.similarity if match_res is not None else 0.0
        new_obs = {
            "embedding": embedding,
            "similarity": sim,
            "frame_idx": self.frame_count
        }
        
        if len(self.observations) < 10:
            self.observations.append(new_obs)
        else:
            # smart eviction: keep last 3 entries unconditionally (FIFO)
            # evict the entry with the lowest similarity from the remaining 7 entries
            hist_entries = self.observations[:-3]
            min_idx = min(range(len(hist_entries)), key=lambda i: hist_entries[i]["similarity"])
            if sim > hist_entries[min_idx]["similarity"]:
                self.observations.pop(min_idx)
                self.observations.append(new_obs)
        
        self.match_result = match_res
        if is_match_run:
            self.last_match_frame = self.frame_count
        self.frame_count += 1

    def get_mean_embedding(self) -> np.ndarray:
        if not self.observations:
            return np.zeros(1, dtype=np.float32)
        embeddings = [obs["embedding"] for obs in self.observations]
        mean_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-6:
            mean_emb = mean_emb / norm
        return mean_emb

class TrackStateManager:
    """Manages states of all tracks and handles eviction strategies."""
    def __init__(self, max_tracks: int = 1000):
        self.tracks: Dict[int, TrackState] = {}
        self.max_tracks = max_tracks

    def get_match(self, track_id: int) -> Optional[Tuple[np.ndarray, MatchResult]]:
        """Return cached mean embedding and MatchResult if present. Updates LRU order."""
        if track_id in self.tracks and self.tracks[track_id].match_result is not None:
            state = self.tracks[track_id]
            self.tracks.pop(track_id)
            self.tracks[track_id] = state
            return state.get_mean_embedding(), state.match_result
        return None

    def update_track(self, track_id: int, embedding: np.ndarray, match_res: MatchResult):
        """Update track state. Evicts least recently used (LRU) track if limit is reached."""
        if track_id not in self.tracks and len(self.tracks) >= self.max_tracks:
            # Evict the oldest key in insertion order (LRU)
            first_key = next(iter(self.tracks))
            self.tracks.pop(first_key)
            
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(track_id)
        else:
            # Move key to the end of insertion order (most recently used)
            state = self.tracks.pop(track_id)
            self.tracks[track_id] = state
            
        # Add observation (since update_track is called on match frames, is_match_run is True)
        self.tracks[track_id].add_observation(embedding, match_res, is_match_run=True)
