import yaml
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List
from pathlib import Path

@dataclass
class Config:
    # Model Paths
    detector_weights: str = "weights/yolo_cat.pt"
    extractor_weights: str = "weights/mega_finetuned.pt"
    
    # Model Settings
    model_name: str = "hf-hub:BVRA/MegaDescriptor-L-384"
    imgsz: int = 384
    conf: float = 0.25
    iou: float = 0.45
    
    # Re-ID Settings
    matcher_type: str = "faiss"  # "faiss" or "knn"
    threshold: float = 0.7
    k: int = 2
    test_size: float = 0.5  # Ratio for validation split
    
    # DB Paths
    db_path: str = "embeddings/db.npz"
    dataset_path: str = "datasets"
    
    # Run Settings
    device: str = "cuda" # or "cpu"
    save: bool = False
    show: bool = True
    
    def __post_init__(self):
        # Ensure paths are valid or relative to root
        pass

    @classmethod
    def load(cls, yaml_path: Optional[str] = "config.yaml"):
        """Load config from YAML and override with CLI arguments."""
        config = cls()
        
        # 1. Load from YAML if exists
        if yaml_path and Path(yaml_path).exists():
            with open(yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f)
                if yaml_data:
                    for k, v in yaml_data.items():
                        if hasattr(config, k):
                            setattr(config, k, v)
        
        # 2. Override with CLI arguments (key=value style)
        cli_args = cls._parse_cli(sys.argv[1:])
        for k, v in cli_args.items():
            if hasattr(config, k):
                # Type conversion based on default value type
                default_val = getattr(config, k)
                try:
                    if isinstance(default_val, bool):
                        setattr(config, k, str(v).lower() in ("true", "1", "yes"))
                    else:
                        setattr(config, k, type(default_val)(v))
                except (ValueError, TypeError):
                    setattr(config, k, v)
        
        # 3. Auto-select device (Handle CUDA fallback)
        from reid.utils.checks import select_device
        config.device = select_device(config.device)
                    
        return config

    @staticmethod
    def _parse_cli(args: List[str]) -> Dict[str, Any]:
        """Parse 'key=value' strings from CLI arguments."""
        result = {}
        for arg in args:
            if "=" in arg:
                k, v = arg.split("=", 1)
                result[k] = v
        return result

    def save_config(self, path: str):
        """Save current config to YAML."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)

# Global config instance (initialized on demand)
_config = None

def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.load()
    return _config
