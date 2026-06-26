import os
import shutil
from reid.utils.profiler import PipelineProfiler

def test_profiler_timing_and_saving():
    profiler = PipelineProfiler(enabled=True)
    with profiler.profile("detection"):
        pass
    profiler.commit_frame()
    
    summary_str = profiler.get_summary_string()
    assert "detection" in summary_str
    
    output_dir = "results/test_source"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    profiler.save_csv("test_source", "test_run")
    csv_path = os.path.join(output_dir, "test_run.csv")
    assert os.path.exists(csv_path)


def test_base_predictor_profiler_integration():
    from reid.models.yolo.detect.predict import YoloPredictor
    from reid.core.config import Config
    import numpy as np
    
    cfg = Config()
    cfg.show = False
    cfg.save = False
    predictor = YoloPredictor(cfg)
    
    assert hasattr(predictor, "profiler")


def test_extractor_predict_batch():
    from reid.models.extractor.predict import ExtractorPredictor
    from reid.core.config import Config
    import torch
    import torch.nn as nn
    import numpy as np

    cfg = Config()
    cfg.imgsz = 224
    cfg.device = "cpu"
    cfg.fp16 = False

    predictor = ExtractorPredictor(cfg)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(128, 3 * 224 * 224))
            self.bias = nn.Parameter(torch.randn(128))
        
        def forward(self, x):
            x_flat = x.view(x.size(0), -1)
            return nn.functional.linear(x_flat, self.weight, self.bias)

    dummy_model = DummyModel()
    predictor.setup_model(dummy_model)

    # 1. Test predicting an empty list of images
    empty_res = predictor.predict_batch([])
    assert isinstance(empty_res, np.ndarray)
    assert empty_res.shape == (0, 128)

    # 2. Test predicting with a batch of images
    img1 = np.zeros((224, 224, 3), dtype=np.uint8)
    img2 = np.ones((224, 224, 3), dtype=np.uint8) * 255
    im_list = [img1, img2]

    res = predictor.predict_batch(im_list)
    assert isinstance(res, np.ndarray)
    assert res.shape == (2, 128)

    # 3. Test preprocess_batch directly
    batch_tensor = predictor.preprocess_batch(im_list)
    assert batch_tensor.shape == (2, 3, 224, 224)
    assert batch_tensor.dtype == torch.float32

    # 4. Test FP16 batch conversion on CUDA if available
    if torch.cuda.is_available():
        predictor.device = "cuda"
        predictor.cfg.fp16 = True
        predictor.model.cuda()
        # Test preprocess_batch converts to half
        cuda_batch = predictor.preprocess_batch(im_list)
        assert cuda_batch.dtype == torch.float16
        assert cuda_batch.device.type == "cuda"


def test_extractor_model_fp16_conversion():
    from reid.models.extractor.model import ExtractorModel
    from reid.core.config import Config
    import torch
    import torch.nn as nn

    class DummyExtractorModel(ExtractorModel):
        def _load_model(self, model_path: str) -> None:
            self.model = nn.Linear(10, 10)
        def _get_trainer(self):
            pass
        def _get_validator(self):
            pass

    # 1. Setup config with fp16=True, device="cuda"
    cfg = Config()
    cfg.fp16 = True
    cfg.device = "cuda"
    cfg.db_path = "embeddings/dummy_db.npz"

    # Ensure clean state for db_path
    if os.path.exists(cfg.db_path):
        os.remove(cfg.db_path)

    try:
        # Instantiate model
        model_wrapper = DummyExtractorModel(db_path=cfg.db_path, cfg=cfg)
        
        # Verify that the underlying model is converted to half
        predictor = model_wrapper._get_predictor()
        # Check if linear weights are float16
        assert next(model_wrapper.model.parameters()).dtype == torch.float16
    finally:
        # Clean up database
        if os.path.exists(cfg.db_path):
            os.remove(cfg.db_path)


def test_reid_predictor_eviction_integration():
    from reid.models.reid.predict import ReIdPredictor
    from reid.core.config import Config
    from reid.core.types import BBox, Results, MatchResult
    from unittest.mock import MagicMock
    import numpy as np
    
    detector = MagicMock()
    extractor = MagicMock()
    matcher = MagicMock()
    
    extractor.store.get_all.return_value = (np.array([]), [])
    extractor.cfg.imgsz = 384
    extractor.predict.return_value = np.ones(512)
    matcher.match.return_value = MatchResult(cat_id="Nabi", similarity=0.90)
    
    cfg = Config()
    cfg.track = True
    cfg.dev = True
    cfg.blur_threshold = 0.0
    
    predictor = ReIdPredictor(detector, extractor, matcher, cfg)
    # Trigger track creations
    for i in range(1005):
        box = BBox(x1=0, y1=0, x2=50, y2=50, track_id=i)
        results = Results(orig_img=np.zeros((100, 100, 3), dtype=np.uint8), path="", boxes=[box])
        detector.predict.return_value = results
        predictor.inference(np.zeros((100, 100, 3), dtype=np.uint8))
        
    # Enforced LRU eviction should cap track manager size at exactly 1000
    assert len(predictor.track_state_manager.tracks) == 1000


def test_full_reid_pipeline_batch_run():
    from reid.models.reid.predict import ReIdPredictor
    from reid.core.config import Config
    from reid.core.types import BBox, Results, MatchResult
    from unittest.mock import MagicMock
    import numpy as np
    
    detector = MagicMock()
    extractor = MagicMock()
    matcher = MagicMock()
    
    extractor.store.get_all.return_value = (np.array([]), [])
    extractor.cfg.imgsz = 384
    extractor.predict.return_value = np.ones(512)
    extractor.predict_batch = MagicMock(return_value=np.ones((2, 512)))
    matcher.match.return_value = MatchResult(cat_id="Cheesecake", similarity=0.90)
    
    cfg = Config()
    cfg.track = True
    cfg.dev = True
    cfg.fp16 = True
    cfg.blur_threshold = 0.0
    
    predictor = ReIdPredictor(detector, extractor, matcher, cfg)
    
    # 2 boxes requiring feature extraction
    box1 = BBox(x1=0, y1=0, x2=50, y2=50, track_id=1)
    box2 = BBox(x1=10, y1=10, x2=60, y2=60, track_id=2)
    results = Results(orig_img=np.zeros((100, 100, 3), dtype=np.uint8), path="", boxes=[box1, box2])
    detector.predict.return_value = results
    
    res = predictor.inference(np.zeros((100, 100, 3), dtype=np.uint8))
    
    # Extractor predict_batch should be called exactly once
    extractor.predict_batch.assert_called_once()
    assert len(res.match_results) == 2
    assert res.match_results[0].cat_id == "Cheesecake"
    assert res.match_results[1].cat_id == "Cheesecake"
    
    # Verify profiling output keys exist and timings are recorded
    assert "detection" in predictor.profiler.current_frame
    assert "quality_filter" in predictor.profiler.current_frame
    assert "extractor" in predictor.profiler.current_frame
    assert "matcher" in predictor.profiler.current_frame
    assert predictor.profiler.current_frame["detection"] >= 0.0
    assert predictor.profiler.current_frame["quality_filter"] >= 0.0
    assert predictor.profiler.current_frame["extractor"] >= 0.0
    assert predictor.profiler.current_frame["matcher"] >= 0.0

