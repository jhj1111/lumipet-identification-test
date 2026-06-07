import sys
import os
from reid.core.config import get_config
from reid.container import build_detector, build_extractor, build_matcher
from reid.pipeline import ReIdModel
from reid.stream.service import StreamService

def main():
    # 1. Load config and handle CLI overrides (key=value)
    cfg = get_config()
    
    # 2. Build components
    detector = build_detector(cfg)
    extractor = build_extractor(cfg)
    matcher = build_matcher(cfg)
    
    # 3. Assemble Pipeline
    pipeline = ReIdModel(detector, extractor, matcher)
    
    # 4. Handle CLI Commands/Modes
    # Usage: reid mode=predict source=0
    #        reid mode=register source=path/to/cat_imgs label=my_cat
    
    mode = "predict" # default
    source = 0
    label = "Unknown"
    
    # Simple mode detection from CLI args
    for arg in sys.argv[1:]:
        if arg.startswith("mode="):
            mode = arg.split("=")[1]
        elif arg.startswith("source="):
            source = arg.split("=")[1]
            if source.isdigit(): source = int(source)
        elif arg.startswith("label="):
            label = arg.split("=")[1]

    if mode == "predict":
        service = StreamService(pipeline, cfg)
        service.run(source=source)
        
    elif mode == "register":
        # Bulk or single registration
        if os.path.isdir(str(source)):
            # datasets/label/*.jpg
            subdirs = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]
            for s_label in subdirs:
                s_dir = os.path.join(source, s_label)
                for f in os.listdir(s_dir):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        extractor.register(os.path.join(s_dir, f), s_label)
        else:
            extractor.register(source, label)
        extractor.save_db()
        print("Registration completed and database saved.")
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: predict, register")

if __name__ == "__main__":
    main()
