import sys
import os
from reid.core.config import get_config
from reid.container import build_detector, build_extractor, build_matcher
from reid.pipeline import ReIdModel
from reid.stream.service import StreamService
from reid.data.loader import CatDataLoader

def main():
    # 1. Load config and handle CLI overrides (key=value)
    cfg = get_config()
    
    # 2. Build components
    detector = build_detector(cfg)
    extractor = build_extractor(cfg)
    matcher = build_matcher(cfg)
    
    # 3. Assemble Pipeline
    pipeline = ReIdModel(detector, extractor, matcher)
    
    # 4. Improved CLI Argument Parsing
    # Usage: reid mode=predict source=0
    #        reid mode=register source=path/to/cat_imgs label=my_cat
    mode = "predict"
    source = None
    label = "Unknown"
    args_dict = {}

    # Parse positional commands (e.g., reid register source=...)
    if len(sys.argv) > 1 and "=" not in sys.argv[1]:
        mode = sys.argv[1]
        remaining_args = sys.argv[2:]
    else:
        remaining_args = sys.argv[1:]

    # Parse key=value and handle shorthand for register
    for i, arg in enumerate(remaining_args):
        if "=" in arg:
            k, v = arg.split("=", 1)
            args_dict[k] = v
        else:
            # Handle positional label/source if not using key=value
            if mode == "register":
                if source is None: source = arg
                elif label == "Unknown": label = arg

    # Apply parsed values
    mode = args_dict.get("mode", mode)
    source = args_dict.get("source", source)
    label = args_dict.get("label", label)
    
    # Defaults
    if source is None: source = 0 # Camera 0
    if source.isdigit(): source = int(source)

    if mode == "predict":
        service = StreamService(pipeline, cfg)
        service.run(source=source)
        
    elif mode == "register":
        # Bulk or single registration
        if os.path.isdir(str(source)):
            print(f"Bulk registering from directory: {source}")
            subdirs = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]
            for s_label in subdirs:
                s_dir = os.path.join(source, s_label)
                for f in os.listdir(s_dir):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        extractor.register(os.path.join(s_dir, f), s_label)
        else:
            print(f"Registering single image: {source} as {label}")
            extractor.register(source, label)
        extractor.save_db()
        print("Registration completed and database saved.")

    elif mode == "train":
        loader = CatDataLoader(cfg.dataset_path)
        train_loader, val_loader = loader.get_loaders()
        if train_loader:
            extractor.train(train_loader, val_loader)
        else:
            print("Error: Could not load training data.")

    elif mode == "val":
        loader = CatDataLoader(cfg.dataset_path)
        # Convert loader list to required format [(path, label), ...]
        val_data = list(zip(loader.image_paths, loader.labels))
        extractor.val(pipeline, val_data)
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: predict, register, train, val")

if __name__ == "__main__":
    main()
