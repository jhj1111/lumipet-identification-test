import pytest
import sys
import tempfile
import os
import numpy as np
from unittest.mock import patch
from reid.core.config import Config

def test_config_properties():
    config = Config()
    # Assert default tracking and dev properties exist
    assert hasattr(config, "dev")
    assert hasattr(config, "track")
    assert hasattr(config, "tracker")
    assert config.dev is True
    assert config.track is True
    assert config.tracker == "bytetrack.yaml"
    
    # Assert Re-ID specific thresholds and intervals
    assert config.threshold_candidate == 0.70
    assert config.threshold_lock == 0.85
    assert config.threshold_hysteresis == 0.55
    assert config.candidate_interval == 10
    assert config.lock_interval == 60
    
    # Assert quality filters
    assert config.min_bbox_width == 32
    assert config.min_bbox_height == 32
    assert config.blur_threshold == 10.0

def test_config_cli_overrides():
    test_args = ["predict", "dev=False", "track=False", "tracker=botsort.yaml"]
    with patch.object(sys, "argv", ["cli.py"] + test_args):
        config = Config.load(yaml_path=None)
        assert config.dev is False
        assert config.track is False
        assert config.tracker == "botsort.yaml"

def test_config_yaml_loading():
    yaml_content = """
dev: False
track: False
tracker: "test_tracker.yaml"
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
        f.write(yaml_content)
        temp_path = f.name
        
    try:
        with patch.object(sys, "argv", ["cli.py"]):
            config = Config.load(yaml_path=temp_path)
            assert config.dev is False
            assert config.track is False
            assert config.tracker == "test_tracker.yaml"
    finally:
        os.remove(temp_path)

def test_bbox_track_id():
    from reid.core.types import BBox
    box = BBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0, track_id=42)
    assert box.track_id == 42

def test_yolo_predictor_tracking():
    from unittest.mock import MagicMock
    from reid.models.yolo.detect.predict import YoloPredictor
    from reid.core.config import Config
    
    dummy_img_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 1. Test when track=True
    cfg = Config()
    cfg.track = True
    cfg.tracker = "bytetrack.yaml"
    cfg.conf = 0.25
    cfg.iou = 0.7
    
    predictor = YoloPredictor(cfg)
    mock_model = MagicMock()
    # Mock model.track call return value structure
    mock_box = MagicMock()
    mock_box.xyxy = [MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=[1.0, 2.0, 3.0, 4.0]))))]
    mock_box.conf = [0.9]
    mock_box.cls = np.array([15])
    # Mock box.id to simulate track ID
    mock_id = MagicMock()
    mock_id.__getitem__ = MagicMock(return_value=MagicMock(item=MagicMock(return_value=42)))
    mock_box.id = mock_id
    
    mock_result = MagicMock()
    mock_result.orig_img = dummy_img_frame
    mock_result.path = "dummy_path"
    mock_result.boxes = [mock_box]
    
    mock_model.track.return_value = [mock_result]
    predictor.setup_model(mock_model)
    
    # Pass a numpy array so that it executes single-frame inference
    results = predictor(dummy_img_frame)
    
    mock_model.track.assert_called_once_with(
        dummy_img_frame,
        conf=0.25,
        iou=0.7,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )
    assert len(results.boxes) == 1
    assert results.boxes[0].track_id == 42
    
    # 2. Test when track=False
    cfg.track = False
    mock_model_no_track = MagicMock()
    # Mock model call return value structure
    mock_box_no_track = MagicMock()
    mock_box_no_track.xyxy = [MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=[1.0, 2.0, 3.0, 4.0]))))]
    mock_box_no_track.conf = [0.9]
    mock_box_no_track.cls = np.array([15])
    mock_box_no_track.id = None
    
    mock_result_no_track = MagicMock()
    mock_result_no_track.orig_img = dummy_img_frame
    mock_result_no_track.path = "dummy_path"
    mock_result_no_track.boxes = [mock_box_no_track]
    
    mock_model_no_track.return_value = [mock_result_no_track]
    
    predictor_no_track = YoloPredictor(cfg)
    predictor_no_track.setup_model(mock_model_no_track)
    results_no_track = predictor_no_track(dummy_img_frame)
    
    mock_model_no_track.assert_called_once_with(
        dummy_img_frame,
        conf=0.25,
        iou=0.7,
        verbose=False
    )
    assert len(results_no_track.boxes) == 1
    assert results_no_track.boxes[0].track_id is None


def test_track_state_manager():
    from unittest.mock import MagicMock
    from reid.core.tracker import TrackStateManager
    from reid.core.types import MatchResult
    import numpy as np


    manager = TrackStateManager(max_tracks=2)
    match_res1 = MatchResult(cat_id="Nabi", similarity=0.85)
    emb1 = np.ones(512)

    # 1. Test update and hit
    manager.update_track(1, emb1, match_res1)
    cached = manager.get_match(1)
    assert cached is not None
    assert np.allclose(cached[0], emb1 / np.linalg.norm(emb1))
    assert cached[1].cat_id == "Nabi"

    # 2. Test cache miss
    assert manager.get_match(99) is None

    # 3. Test LRU eviction
    match_res2 = MagicMock()
    match_res2.similarity = 0.85
    emb2 = np.ones(512) * 2
    manager.update_track(2, emb2, match_res2)
    
    # 4. Test updating existing track at capacity does NOT trigger eviction
    manager.update_track(2, emb2 * 1.5, match_res2)
    assert manager.get_match(1) is not None  # Accessing track 1 makes it most recently used (LRU)
    assert len(manager.tracks) == 2

    # Now add track 3 to trigger eviction of track 2 (least recently used)
    match_res3 = MagicMock()
    match_res3.similarity = 0.85
    emb3 = np.ones(512) * 3
    manager.update_track(3, emb3, match_res3)

    # Track 2 should be evicted (as track 1 was accessed and became recently used)
    assert manager.get_match(2) is None
    assert manager.get_match(1) is not None
    assert manager.get_match(3) is not None

    # 5. Test embedding history is capped to prevent unbounded memory growth
    state = manager.tracks[1]
    # Add observations 20 times
    for i in range(20):
        state.add_observation(np.ones(512) * i, match_res2)
    assert len(state.observations) <= 10  # Capped at 10


def test_reid_predictor_caching():
    from reid.pipeline import ReIdPredictor
    from reid.core.types import Results, BBox, MatchResult
    from reid.core.config import Config
    from unittest.mock import MagicMock

    # Mock dependencies
    detector = MagicMock()
    extractor = MagicMock()
    matcher = MagicMock()

    # Configure mock database loading
    extractor.store.get_all.return_value = (np.array([]), [])
    extractor.cfg.imgsz = 384

    # Define config
    cfg = Config()
    cfg.track = True
    cfg.blur_threshold = 0.0

    # 1. Instantiate predictor
    predictor = ReIdPredictor(detector, extractor, matcher, cfg)
    # Predictor calls extractor.predict once during __init__ to find extractor_dim
    init_calls = extractor.predict.call_count

    # 1st Frame: detector returns box with track_id=5
    box = BBox(x1=0, y1=0, x2=100, y2=100, track_id=5)
    orig_img = np.zeros((200, 200, 3), dtype=np.uint8)
    results_frame1 = Results(orig_img=orig_img, path="", boxes=[box])
    detector.predict.return_value = results_frame1

    # Mock extractor and matcher output
    extractor.predict.return_value = np.ones(512)
    matcher.match.return_value = MatchResult(cat_id="Nabi", similarity=0.95)

    # Run 1st inference (cache miss)
    res1 = predictor.inference(orig_img)
    assert extractor.predict.call_count == init_calls + 1
    assert matcher.match.call_count == 1
    assert res1.match_results[0].cat_id == "Nabi"

    # 2nd Frame: detector returns same track_id=5
    box2 = BBox(x1=10, y1=10, x2=110, y2=110, track_id=5)
    results_frame2 = Results(orig_img=orig_img, path="", boxes=[box2])
    detector.predict.return_value = results_frame2

    # Run 2nd inference (cache hit)
    res2 = predictor.inference(orig_img)
    # Call count should still be init_calls + 1 (skipped on 2nd run due to cache hit)
    assert extractor.predict.call_count == init_calls + 1
    assert matcher.match.call_count == 1
    assert res2.match_results[0].cat_id == "Nabi"

    # 3. Test caching reset
    predictor.reset()
    res3 = predictor.inference(orig_img)
    # Call count should increase (since cache was cleared)
    assert extractor.predict.call_count == init_calls + 2
    assert matcher.match.call_count == 2


    # 4. Test index alignment when crop size is zero (e.g. empty box)
    empty_box = BBox(x1=0, y1=0, x2=0, y2=0, track_id=12) # crop size 0
    results_empty = Results(orig_img=orig_img, path="", boxes=[empty_box])
    detector.predict.return_value = results_empty
    
    res_empty = predictor.inference(orig_img)
    assert len(res_empty.boxes) == 1
    assert len(res_empty.match_results) == 1
    assert res_empty.match_results[0].cat_id == "Unknown"
    assert res_empty.match_results[0].similarity == 0.0


def test_renderer_modes():
    from reid.stream.overlay import Renderer
    from reid.core.types import Results, BBox, MatchResult
    from unittest.mock import patch, ANY
    import numpy as np

    orig_img = np.zeros((100, 100, 3), dtype=np.uint8)
    box = BBox(x1=0, y1=0, x2=50, y2=50, track_id=7)
    match = MatchResult(cat_id="Nabi", similarity=0.92)
    results = Results(orig_img=orig_img, path="", boxes=[box], match_results=[match])

    # 1. Dev mode: should show ID and similarity
    renderer_dev = Renderer(dev=True)
    with patch("cv2.putText") as mock_put_text, \
         patch("cv2.rectangle"), \
         patch("cv2.getTextSize", return_value=((10, 10), 5)):
        renderer_dev.draw(results)
        mock_put_text.assert_called_with(
            ANY, "ID:7 Nabi 0.92", ANY, ANY, ANY, (0, 0, 0), 2
        )

    # 2. Service mode: should show name only
    renderer_prod = Renderer(dev=False)
    with patch("cv2.putText") as mock_put_text, \
         patch("cv2.rectangle"), \
         patch("cv2.getTextSize", return_value=((10, 10), 5)):
        renderer_prod.draw(results)
        mock_put_text.assert_called_with(
            ANY, "Nabi", ANY, ANY, ANY, (0, 0, 0), 2
        )


def test_quality_filtering():
    import numpy as np
    from reid.pipeline import ReIdPredictor
    from reid.core.types import Results, BBox, MatchResult
    from reid.core.config import Config
    from unittest.mock import MagicMock
    
    # Mock dependencies
    detector = MagicMock()
    extractor = MagicMock()
    matcher = MagicMock()
    
    extractor.store.get_all.return_value = (np.array([]), [])
    extractor.cfg.imgsz = 384
    extractor.predict.return_value = np.ones(512)
    
    cfg = Config()
    cfg.track = True
    cfg.min_bbox_width = 30
    cfg.min_bbox_height = 30
    cfg.blur_threshold = 5.0
    
    predictor = ReIdPredictor(detector, extractor, matcher, cfg)
    
    # Test 1: Tiny box size filter (20x20)
    box = BBox(x1=0, y1=0, x2=20, y2=20, track_id=8)
    orig_img = np.zeros((100, 100, 3), dtype=np.uint8)
    results = Results(orig_img=orig_img, path="", boxes=[box])
    detector.predict.return_value = results
    
    res = predictor.inference(orig_img)
    assert res.match_results[0].cat_id == "Unknown"
    assert res.match_results[0].similarity == 0.0
    assert extractor.predict.call_count == 1 # only init call
    
    # Test 2: Blurry box (Laplacian var ~ 0)
    blurry_box = BBox(x1=0, y1=0, x2=50, y2=50, track_id=9)
    results_blur = Results(orig_img=orig_img, path="", boxes=[blurry_box])
    detector.predict.return_value = results_blur
    
    res_blur = predictor.inference(orig_img)
    assert res_blur.match_results[0].cat_id == "Unknown"
    assert res_blur.match_results[0].similarity == 0.0
    assert extractor.predict.call_count == 1

def test_state_machine_transitions():
    from reid.pipeline import ReIdPredictor
    from reid.core.types import Results, BBox, MatchResult
    from reid.core.config import Config
    from unittest.mock import MagicMock
    import numpy as np
    
    detector = MagicMock()
    extractor = MagicMock()
    matcher = MagicMock()
    
    extractor.store.get_all.return_value = (np.array([]), [])
    extractor.cfg.imgsz = 384
    extractor.predict.return_value = np.ones(512)
    
    cfg = Config()
    cfg.track = True
    cfg.threshold_candidate = 0.70
    cfg.threshold_lock = 0.85
    cfg.threshold_hysteresis = 0.55
    cfg.candidate_interval = 2
    cfg.lock_interval = 5
    cfg.blur_threshold = 0.0 # disable blur check for test
    
    predictor = ReIdPredictor(detector, extractor, matcher, cfg)
    
    # Frame 1: Match with sim=0.60 (stays Unknown)
    box = BBox(x1=0, y1=0, x2=50, y2=50, track_id=12)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.predict.return_value = Results(orig_img=img, path="", boxes=[box])
    matcher.match.return_value = MatchResult(cat_id="Nabi", similarity=0.60)
    
    predictor.inference(img)
    track_state = predictor.track_state_manager.tracks[12]
    assert track_state.state == "Unknown"
    
    # Frame 2: Match with sim=0.75 (transitions to Candidate)
    matcher.match.return_value = MatchResult(cat_id="Nabi", similarity=0.75)
    predictor.inference(img)
    assert track_state.state == "Candidate"
    
    # Frame 3: Candidate cache hit (does not match, candidate_interval is 2)
    matcher.match.reset_mock()
    predictor.inference(img)
    matcher.match.assert_not_called()
    
    # Frame 4: Forced re-match on Candidate (sim=0.90 -> transitions to Locked)
    matcher.match.return_value = MatchResult(cat_id="Nabi", similarity=0.90)
    predictor.inference(img)
    assert track_state.state == "Locked"
    
    # Frame 5: Locked cache hit
    matcher.match.reset_mock()
    predictor.inference(img)
    matcher.match.assert_not_called()

def test_smart_eviction():
    from reid.core.tracker import TrackState
    from reid.core.types import MatchResult
    import numpy as np
    
    state = TrackState(track_id=1)
    # Fill observations with index 0..9
    # Set first 7 similarities low (0.5), last 3 similarity higher (0.8) to simulate recent ones
    for i in range(10):
        sim = 0.8 if i >= 7 else 0.5
        state.add_observation(np.ones(512) * i, MatchResult(cat_id="A", similarity=sim))
        
    assert len(state.observations) == 10
    
    # Insert new observation with similarity 0.7
    # It should evict one of the 0..6 entries (lowest similarity = 0.5), keeping the recent 3
    new_emb = np.ones(512) * 99
    state.add_observation(new_emb, MatchResult(cat_id="A", similarity=0.7))
    
    assert len(state.observations) == 10
    # Verify one entry of similarity 0.5 was evicted, and 0.7 is added
    similarities = [obs["similarity"] for obs in state.observations]
    assert 0.7 in similarities
    assert similarities.count(0.5) == 6
    assert similarities.count(0.8) == 3



