import sys
import os
from tqdm import tqdm

from reid.core.config import get_config
from reid.container import build_detector, build_extractor, build_matcher
from reid.pipeline import ReIdModel

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
        predictor = pipeline._get_predictor()
        predictor(cfg.source)
        
    elif cfg.mode == "register":
        source_str = str(cfg.source)
        if not os.path.exists(source_str):
            print(f"Error: Register source {source_str} does not exist.")
            return

        if os.path.isdir(source_str):
            print(f"Bulk registering from directory: {source_str}")
            labels = [d for d in os.listdir(source_str) if os.path.isdir(os.path.join(source_str, d))]
            for s_label in tqdm(labels):
                label_dir = os.path.join(source_str, s_label)
                for root, _, files in os.walk(label_dir):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            extractor.register(os.path.join(root, f), s_label, verbose=False)
        else:
            print(f"Registering single image: {source_str} as {cfg.label}")
            extractor.register(source_str, cfg.label)
        extractor.save_db()
        print("Registration completed and database saved.")

    elif cfg.mode == "train":
        extractor.train()

    elif cfg.mode == "val":
        extractor.val(pipeline=pipeline)
    
    else:
        print(f"Unknown mode: {cfg.mode}")
        print("Available modes: predict, register, train, val")

if __name__ == "__main__":
    main()
