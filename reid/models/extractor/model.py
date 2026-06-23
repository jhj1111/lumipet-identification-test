from abc import abstractmethod
from typing import Any, Optional

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

    def register(self, image: Any, label: str, verbose: bool = True) -> None:
        """Extract features and save to embedding store."""
        embedding = self.predict(image)
        self.store.add(embedding, label)
        if verbose:
            print(f"Registered cat: {label}")

    def save_db(self) -> None:
        """Save the database of embeddings."""
        self.store.save()