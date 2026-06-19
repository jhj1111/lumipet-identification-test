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
    mock_box.cls = [0]
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
    mock_box_no_track.cls = [0]
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
    assert np.array_equal(cached[0], emb1)
    assert cached[1].cat_id == "Nabi"

    # 2. Test cache miss
    assert manager.get_match(99) is None

    # 3. Test FIFO eviction
    match_res2 = MagicMock()
    emb2 = np.ones(512) * 2
    manager.update_track(2, emb2, match_res2)
    
    # 4. Test updating existing track at capacity does NOT trigger eviction
    manager.update_track(2, emb2 * 1.5, match_res2)
    assert manager.get_match(1) is not None  # Track 1 should NOT be evicted yet
    assert len(manager.tracks) == 2

    # Now add track 3 to trigger eviction of track 1
    match_res3 = MagicMock()
    emb3 = np.ones(512) * 3
    manager.update_track(3, emb3, match_res3)

    # Track 1 (oldest) should be evicted
    assert manager.get_match(1) is None
    assert manager.get_match(2) is not None
    assert manager.get_match(3) is not None

    # 5. Test embedding history is capped to prevent unbounded memory growth
    state = manager.tracks[2]
    # Add observations 20 times
    for i in range(20):
        state.add_observation(np.ones(512) * i, match_res2)
    assert len(state.embeddings) <= 10  # Capped at 10


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

    predictor = ReIdPredictor(detector, extractor, matcher, cfg)

    # 1st Frame: detector returns box with track_id=5
    box = BBox(x1=0, y1=0, x2=100, y2=100, track_id=5)
    orig_img = np.zeros((200, 200, 3), dtype=np.uint8)
    results_frame1 = Results(orig_img=orig_img, path="", boxes=[box])
    detector.predict.return_value = results_frame1

    # Mock extractor and matcher output
    extractor.predict.return_value = np.ones(512)
    matcher.match.return_value = MatchResult(cat_id="Nabi", similarity=0.95)

    # Run 1st inference
    res1 = predictor.inference(orig_img)
    assert extractor.predict.call_count == 1
    assert matcher.match.call_count == 1
    assert res1.match_results[0].cat_id == "Nabi"

    # 2nd Frame: detector returns same track_id=5
    box2 = BBox(x1=10, y1=10, x2=110, y2=110, track_id=5)
    results_frame2 = Results(orig_img=orig_img, path="", boxes=[box2])
    detector.predict.return_value = results_frame2

    # Run 2nd inference
    res2 = predictor.inference(orig_img)
    # Call count should still be 1 (skipped on 2nd run due to cache hit)
    assert extractor.predict.call_count == 1
    assert matcher.match.call_count == 1
    assert res2.match_results[0].cat_id == "Nabi"

    # 3. Test caching reset
    predictor.reset()
    res3 = predictor.inference(orig_img)
    # Call count should increase (since cache was cleared)
    assert extractor.predict.call_count == 2
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
    import numpy as np

    orig_img = np.zeros((100, 100, 3), dtype=np.uint8)
    box = BBox(x1=0, y1=0, x2=50, y2=50, track_id=7)
    match = MatchResult(cat_id="Nabi", similarity=0.92)
    results = Results(orig_img=orig_img, path="", boxes=[box], match_results=[match])

    # 1. Dev mode: should show ID and similarity
    renderer_dev = Renderer(dev=True)
    img_dev = renderer_dev.draw(results)
    assert img_dev.shape == orig_img.shape

    # 2. Service mode: should show name only
    renderer_prod = Renderer(dev=False)
    img_prod = renderer_prod.draw(results)
    assert img_prod.shape == orig_img.shape

