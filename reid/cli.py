import sys
import os

from tqdm import tqdm

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

    # Parse key=value and handle positional arguments
    skip_next = False
    for i, arg in enumerate(remaining_args):
        if skip_next:
            skip_next = False
            continue
            
        if "=" in arg:
            k, v = arg.split("=", 1)
            args_dict[k] = v
        else:
            # Handle keywords used as flags followed by values
            if arg in ["source", "label", "mode"]:
                if i + 1 < len(remaining_args):
                    args_dict[arg] = remaining_args[i+1]
                    skip_next = True
                continue
                
            # Fallback to pure positional based on mode
            if mode == "register":
                if source is None: source = arg
                elif label == "Unknown": label = arg

    # Apply parsed values (priority: args_dict > positional > defaults)
    mode = args_dict.get("mode", mode)
    source = args_dict.get("source", source)
    label = args_dict.get("label", label)
    
    # Defaults
    if source is None: source = '0' # Camera 0
    if isinstance(source, str) and source.isdigit(): source = int(source)

    if mode == "predict":
        service = StreamService(pipeline, cfg)
        service.run(source=source)
        
    elif mode == "register":
        # Bulk or single registration
        if os.path.isdir(str(source)):
            print(f"Bulk registering from directory: {source}")
            if not os.path.exists(source) :
                print(f"Warning: Dataset path {source} does not exist.")
                return
            labels = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]
            for label in tqdm(labels):
                label_dir = os.path.join(source, label)
                if not os.path.isdir(label_dir):
                    continue
                for root, _, files in os.walk(label_dir):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            extractor.register(os.path.join(root, f), label, verbose=False)

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
        extractor.val(val_data, pipeline=pipeline)
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: predict, register, train, val")

if __name__ == "__main__":
    main()
