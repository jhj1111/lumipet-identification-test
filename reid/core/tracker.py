from typing import Dict, List, Tuple, Optional
import numpy as np
from reid.core.types import MatchResult

class TrackState:
    """Manages the state and history of an individual tracked object."""
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.embeddings: List[np.ndarray] = []
        self.match_result: Optional[MatchResult] = None
        self.frame_count = 0
        
    def add_observation(self, embedding: np.ndarray, match_res: MatchResult):
        self.embeddings.append(embedding)
        # Prevent unbounded memory growth by keeping only the last 10 observations
        if len(self.embeddings) > 10:
            self.embeddings.pop(0)
        self.match_result = match_res
        self.frame_count += 1

class TrackStateManager:
    """Manages states of all tracks and handles eviction strategies."""
    def __init__(self, max_tracks: int = 1000):
        self.tracks: Dict[int, TrackState] = {}
        self.max_tracks = max_tracks

    def get_match(self, track_id: int) -> Optional[Tuple[np.ndarray, MatchResult]]:
        """Return cached embedding and MatchResult if present. Updates LRU order."""
        if track_id in self.tracks and self.tracks[track_id].match_result is not None:
            state = self.tracks[track_id]
            # LRU update: Pop and re-insert key to move it to the end of insertion order
            self.tracks.pop(track_id)
            self.tracks[track_id] = state
            return state.embeddings[-1], state.match_result
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
            
        self.tracks[track_id].add_observation(embedding, match_res)


