import sys
import os
from tqdm import tqdm

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

    # 4. Handle DB operations that don't need full pipeline models
    if cfg.mode == "list":
        extractor = build_extractor(cfg)
        labels_summary = extractor.store.list_labels()
        print("\n=== Registered Cats Summary ===")
        if not labels_summary:
            print("No cats registered in the database.")
        else:
            for label, count in labels_summary.items():
                print(f" - {label}: {count} embedding(s)")
        print("===============================\n")
        return

    elif cfg.mode == "delete":
        # Extract label argument from CLI overrides
        label_to_delete = None
        for arg in sys.argv[2:]:
            if arg.startswith("label="):
                label_to_delete = arg.split("=", 1)[1]
                break
        if not label_to_delete:
            print("Error: Please specify the label to delete, e.g., 'reid delete label=Nabi'")
            return
            
        extractor = build_extractor(cfg)
        count = extractor.store.delete_label(label_to_delete)
        print(f"Deleted {count} embedding(s) for label: {label_to_delete}")
        return

    elif cfg.mode == "migrate":
        extractor = build_extractor(cfg)
        cursor = extractor.store.conn.cursor()
        cursor.execute("SELECT label, image_path FROM embeddings")
        rows = cursor.fetchall()
        if not rows:
            print("No records found in database to migrate.")
            return

        print(f"Found {len(rows)} records. Verifying original files on disk...")
        valid_files = []
        for label, img_path in rows:
            if img_path and os.path.exists(img_path):
                valid_files.append((img_path, label))
            else:
                print(f"Warning: Original file missing, skipping: {img_path}")

        if not valid_files:
            print("Error: No original images exist on disk. Migration aborted.")
            return

        print(f"Migrating {len(valid_files)} embeddings using active model: {cfg.model_name}")
        
        # Clear database and re-register
        extractor.store.clear()
        
        from reid.utils import calculate_md5
        valid_registrations = []
        for img_path, img_label in valid_files:
            try:
                img_hash = calculate_md5(img_path)
                valid_registrations.append((img_path, img_label, img_hash))
            except Exception as e:
                print(f"Error calculating hash for {img_path}: {e}")

        # Call public batch registration method from extractor
        extractor.register_batch_images(valid_registrations)
        print("Migration completed successfully.")
        return

    # 5. Build components for inference modes
    detector = build_detector(cfg)
    extractor = build_extractor(cfg)
    matcher = build_matcher(cfg)
    
    # 6. Assemble Pipeline
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
        print("Available modes: predict, register, list, delete, migrate, train, val")

if __name__ == "__main__":
    main()
