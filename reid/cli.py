import sys

from reid.core.config import get_config
from reid.container import build_detector, build_extractor, build_matcher
from reid.models import ReIdModel

def main() -> None:
    # 1. Load config and handle CLI overrides (key=value)
    cfg = get_config()
    
    # 2. Set positional mode if provided
    if len(sys.argv) > 1 and "=" not in sys.argv[1]:
        cfg.mode = sys.argv[1]
    
    # 3. Convert digit source to int
    if isinstance(cfg.source, str) and cfg.source.isdigit():
        cfg.source = int(cfg.source)

    # 4. Build components
    detector = build_detector(cfg)
    extractor = build_extractor(cfg)
    matcher = build_matcher(cfg)
    
    # 5. Assemble Pipeline
    pipeline = ReIdModel(detector, extractor, matcher, cfg=cfg)

    if cfg.mode == "predict":
        pipeline.predict(source=cfg.source)
        
    elif cfg.mode == "register":
        extractor.register(source=str(cfg.source), label=cfg.label)

    elif cfg.mode == "train":
        extractor.train()

    elif cfg.mode == "val":
        extractor.val(pipeline=pipeline)
    
    else:
        print(f"Unknown mode: {cfg.mode}")
        print("Available modes: predict, register, train, val")

if __name__ == "__main__":
    main()
