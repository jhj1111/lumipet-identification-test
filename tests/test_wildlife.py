import os
import shutil
import numpy as np
import torch
from reid.core.config import get_config
from reid.container import build_extractor
from reid.models.extractor.wildlife.model import WildlifeExtractorModel
from reid.models.extractor.wildlife.train import WildlifeExtractorTrainer

def test_wildlife_flow():
    print("Testing WildlifeExtractorModel initialization...")
    cfg = get_config()
    cfg.extractor_type = "wildlife"
    cfg.model_name = "hf-hub:BVRA/MegaDescriptor-T-224"
    cfg.extractor_weights = "weights/test_wildlife_weights.pth"
    cfg.db_path = "embeddings/test_db.npz"
    cfg.dataset_path = "datasets/mini_dataset"
    cfg.epochs = 1
    cfg.batch_size = 2
    cfg.device = "cpu"  # enforce CPU testing for stability
    
    # 1. Initialize model
    model = build_extractor(cfg)
    assert isinstance(model, WildlifeExtractorModel)
    assert model.embedding_size > 0
    print(f"Success! Model embedding size is {model.embedding_size}")
    
    # 2. Extract features check
    dummy_img = np.zeros((384, 384, 3), dtype=np.uint8)
    emb = model.predict(dummy_img)
    assert emb.shape[0] == model.embedding_size
    print(f"Success! Predict feature shape: {emb.shape}")
    
    # 3. Train check
    print("Testing 1 epoch training with ArcFaceLoss...")
    trainer = model._get_trainer()
    assert isinstance(trainer, WildlifeExtractorTrainer)
    trainer.train()
    assert os.path.exists(cfg.extractor_weights)
    print("Success! Checkpoint saved.")
    
    # Clean up test outputs
    if os.path.exists(cfg.extractor_weights):
        os.remove(cfg.extractor_weights)
    if os.path.exists(cfg.db_path):
        os.remove(cfg.db_path)
    db_db_path = cfg.db_path.replace(".npz", ".db")
    if os.path.exists(db_db_path):
        os.remove(db_db_path)
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_wildlife_flow()
