
from reid.models.extractor import mega_descriptor, wildlife
from .predict import ExtractorPredictor
from .val import ExtractorValidator
from .embedding import EmbeddingStore

__all__ = "mega_descriptor", "wildlife", "ExtractorPredictor", "ExtractorValidator", "EmbeddingStore"