import os
import tempfile
import sqlite3
import numpy as np
from reid.models.extractor.embedding import EmbeddingStore

def test_db_initialization():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db_path = f.name
    
    try:
        store = EmbeddingStore(temp_db_path)
        # Check connection is active and tables exist
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings';")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "embeddings"
        conn.close()
    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

def test_db_crud_operations():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db_path = f.name
        
    try:
        store = EmbeddingStore(temp_db_path)
        
        # 1. Test insertion and retrieval
        emb1 = np.ones((1, 128), dtype=np.float32)
        store.add(emb1, "Cheesecake", image_path="cat1.jpg", image_hash="hash1", model_name="test_model")
        
        embs, labels = store.get_all(model_name="test_model")
        assert embs.shape == (1, 128)
        assert labels == ["Cheesecake"]
        
        # 2. Test duplicate hash handling
        store.add(emb1, "Cheesecake", image_path="cat1.jpg", image_hash="hash1", model_name="test_model")
        # Should gracefully ignore insertion (duplicate hash) and total count remains 1
        embs, labels = store.get_all(model_name="test_model")
        assert len(labels) == 1
        
        # 3. Test list_labels
        emb2 = np.ones((1, 128), dtype=np.float32) * 2
        store.add(emb2, "Nabi", image_path="cat2.jpg", image_hash="hash2", model_name="test_model")
        counts = store.list_labels()
        assert counts == {"Cheesecake": 1, "Nabi": 1}
        
        # 4. Test delete_label
        deleted = store.delete_label("Cheesecake")
        assert deleted == 1
        counts_after = store.list_labels()
        assert "Cheesecake" not in counts_after
        
        # 5. Test clear
        store.clear()
        assert len(store.list_labels()) == 0
        
    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

def test_extractor_model_batch_registration():
    from unittest.mock import MagicMock
    from reid.models.extractor.model import ExtractorModel
    from reid.core.config import Config
    
    cfg = Config()
    cfg.imgsz = 224
    cfg.batch_size = 2
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db_path = f.name
        
    cfg.db_path = temp_db_path
    
    # Mock original _load_model and _get_predictor
    class MockExtractorModel(ExtractorModel):
        def _load_model(self, model_path: str) -> None:
            self.model = MagicMock()
        def _get_trainer(self): pass
        def _get_validator(self): pass
        
    try:
        model = MockExtractorModel(db_path=temp_db_path, cfg=cfg)
        
        # Mock predictor batch inference
        mock_predictor = MagicMock()
        mock_predictor.predict_batch.side_effect = lambda im_list: np.ones((len(im_list), 128), dtype=np.float32)
        model._get_predictor = MagicMock(return_value=mock_predictor)
        
        # Create temp source directory for bulk registration
        with tempfile.TemporaryDirectory() as src_dir:
            nabi_dir = os.path.join(src_dir, "Nabi")
            os.makedirs(nabi_dir)
            
            # Write 3 dummy image files
            for i in range(3):
                with open(os.path.join(nabi_dir, f"img_{i}.jpg"), "wb") as f:
                    f.write(f"fake_data_{i}".encode())
                    
            model.register(src_dir, label="Unknown")
            
        # Verify 3 embeddings were added to DB
        embs, labels = model.store.get_all(model_name=cfg.model_name)
        assert len(labels) == 3
        assert labels.count("Nabi") == 3
        assert embs.shape == (3, 128)
        
    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

def test_cli_commands_integration():
    from unittest.mock import patch, MagicMock
    import sys
    from reid.cli import main
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db_path = f.name
        
    try:
        # 1. Test "list" mode
        test_args = ["reid", "list", f"db_path={temp_db_path}"]
        with patch.object(sys, "argv", test_args), patch("builtins.print") as mock_print:
            main()
            printed = "".join([call[0][0] for call in mock_print.call_args_list if call[0]])
            assert "Registered Cats Summary" in printed
            
        # 2. Test "delete" mode
        test_args_delete = ["reid", "delete", "label=Cheesecake", f"db_path={temp_db_path}"]
        with patch.object(sys, "argv", test_args_delete), patch("builtins.print") as mock_print:
            main()
            printed = "".join([call[0][0] for call in mock_print.call_args_list if call[0]])
            assert "Deleted" in printed
            
    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
