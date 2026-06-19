import pytest
import sys
import tempfile
import os
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
