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




