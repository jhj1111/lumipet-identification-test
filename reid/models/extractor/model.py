from abc import abstractmethod
from typing import Any, Optional, List, Tuple
import os
from tqdm import tqdm

from reid.engine.model import BaseModel
from reid.engine.trainer import BaseTrainer
from reid.models.extractor.embedding import EmbeddingStore
from reid.models.extractor.predict import ExtractorPredictor


class ExtractorModel(BaseModel):
    """
    Feature Extractor Model wrapper.
    """
    def __init__(
            self,
            model_path: Optional[str] = None,
            model_name: Optional[str] = None,
            db_path: Optional[str] = None,
            cfg=None
        ) -> None:
        from reid.core.config import get_config
        cfg_inst = cfg or get_config()
        db_path = db_path or cfg_inst.db_path
        m_path = model_path or cfg_inst.extractor_weights
        m_name = model_name or cfg_inst.model_name

        super().__init__(m_path, task="reid", cfg=cfg_inst)
        self.model_name = m_name
        self._load_model(self.model_path)
        self.store = EmbeddingStore(db_path)

    @abstractmethod
    def _load_model(self, model_path: str) -> None:
        pass

    def _get_predictor(self) -> ExtractorPredictor:
        """Return the shared predictor class."""
        if getattr(self.cfg, "fp16", False) and "cuda" in str(self.cfg.device):
            if hasattr(self.model, "half"):
                self.model.half()
        predictor = ExtractorPredictor(self.cfg)
        predictor.setup_model(self.model)
        return predictor

    @abstractmethod
    def _get_trainer(self) -> BaseTrainer:
        """Return the training wrapper."""
        pass

    @abstractmethod
    def _get_validator(self) -> BaseTrainer:
        """Return the validation wrapper."""
        pass

    def register(self, source: str, label: str, verbose: Optional[bool] = None) -> None:
        """Extract features and save to embedding store using batch prediction and hash deduplication."""
        from reid.utils import calculate_md5
        from tqdm import tqdm
        
        if not os.path.exists(source):
            print(f"Error: Register source {source} does not exist.")
            return

        image_files = [] # list of (filepath, label)
        
        if os.path.isdir(source):
            print(f"Scanning directory for registration: {source}")
            labels = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]
            for s_label in labels:
                label_dir = os.path.join(source, s_label)
                for root, _, files in os.walk(label_dir):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_files.append((os.path.join(root, f), s_label))
        else:
            image_files.append((source, label))

        # Check existing hashes in store to prevent duplicates
        cursor = self.store.conn.cursor()
        cursor.execute("SELECT image_hash FROM embeddings")
        existing_hashes = {row[0] for row in cursor.fetchall() if row[0] is not None}
        
        # Calculate hashes and filter
        valid_registrations = []
        for img_path, img_label in image_files:
            try:
                img_hash = calculate_md5(img_path)
                if img_hash not in existing_hashes:
                    valid_registrations.append((img_path, img_label, img_hash))
            except Exception as e:
                print(f"Error reading {img_path}: {e}")

        if not valid_registrations:
            print("No new unique images to register.")
            return

        print(f"Registering {len(valid_registrations)} new unique image(s) in batches...")
        self.register_batch_images(valid_registrations)
        print("Registration completed successfully.")

    def register_batch_images(self, image_data: List[Tuple[str, str, str]]) -> None:
        """
        Extract features and save a list of (image_path, label, image_hash) in batches.
        """
        from tqdm import tqdm
        batch_size = getattr(self.cfg, "batch_size", 16)
        predictor = self._get_predictor()
        
        for idx in tqdm(range(0, len(image_data), batch_size)):
            chunk = image_data[idx:idx + batch_size]
            chunk_paths = [item[0] for item in chunk]
            chunk_labels = [item[1] for item in chunk]
            chunk_hashes = [item[2] for item in chunk]
            
            try:
                embeddings = predictor.predict_batch(chunk_paths)
                self.store.add_batch(
                    embeddings=embeddings,
                    labels=chunk_labels,
                    image_paths=chunk_paths,
                    image_hashes=chunk_hashes,
                    model_name=self.cfg.model_name
                )
            except Exception as e:
                print(f"Failed extracting batch {chunk_paths}: {e}")

    def save_db(self) -> None:
        """Deprecated: SQLite auto-saves."""
        pass