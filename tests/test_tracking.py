import pytest
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
