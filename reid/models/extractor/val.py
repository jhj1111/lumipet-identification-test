import time
import numpy as np
from tqdm import tqdm
from reid.engine.validator import BaseValidator
from reid.data.loader import CatDataLoader

class ExtractorValidator(BaseValidator):
    """
    Validator for Re-ID performance (Accuracy & Latency) using Dynamic Split.
    """
    def validate(self, pipeline=None) -> dict:
        """
        Validate dynamic gallery/query split accuracy.
        """
        loader = CatDataLoader(self.cfg.dataset_path)
        gallery_data, query_data = loader.get_reid_split(test_size=self.cfg.test_size)
        
        if not pipeline or gallery_data is None or query_data is None:
            raise ValueError("pipeline, gallery, and query data are required for dynamic validation.")

        # Save original visualization flags to suppress display during validation
        orig_show = pipeline.cfg.show
        orig_save = pipeline.cfg.save
        pipeline.cfg.show = False
        pipeline.cfg.save = False

        try:
            # Ensure pipeline predictor is initialized
            if pipeline.predictor is None:
                pipeline.predictor = pipeline._get_predictor()

            # 1. Clear current matcher state to ensure a fair test
            pipeline.predictor.matcher.is_fitted = False
            pipeline.predictor.matcher.embeddings = None
            pipeline.predictor.matcher.labels = []
            if hasattr(pipeline.predictor.matcher, 'index') and pipeline.predictor.matcher.index is not None:
                pipeline.predictor.matcher.index.reset()

            # 2. Build Temporary Gallery
            print(f"Building gallery DB with {len(gallery_data)} images...")
            gallery_embeddings = []
            gallery_labels = []
            
            # Disable verbose output during gallery build if possible
            for img_path, label in tqdm(gallery_data, desc="Gallery"):
                # We use the pipeline's individual components directly to bypass the matcher during DB build
                res = pipeline.detector.predict(img_path)
                if not res.boxes: 
                    continue
                    
                # Safely take the largest box for registration
                box = max(res.boxes, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
                crop = box.crop(res.orig_img)
                
                if crop.size > 0:
                    emb = pipeline.extractor.predict(crop)
                    gallery_embeddings.append(emb)
                    gallery_labels.append(label)
                    
            if gallery_embeddings:
                # Fit the matcher with our temporary gallery
                pipeline.predictor.matcher.fit(np.vstack(gallery_embeddings), gallery_labels)
            else:
                print("Error: Could not extract any embeddings for the gallery.")
                return {}

            # 3. Evaluate on Query
            correct = 0
            total = len(query_data)
            latencies = []
            
            print(f"Validating on {total} query images...")
            for img_path, true_label in tqdm(query_data, desc="Query"):
                start_time = time.time()
                
                # Use the full pipeline to predict (Detect -> Extract -> Match against temporary DB)
                results = pipeline.predict(img_path)
                
                latencies.append(time.time() - start_time)
                
                if len(results.match_results) > 0:
                    # Check if the true cat was identified in ANY of the detected boxes
                    detected_labels = [m.cat_id for m in results.match_results]
                    if true_label in detected_labels:
                        correct += 1
                        
            accuracy = correct / total if total > 0 else 0
            avg_latency = np.mean(latencies)
            
            print(f"\nResult -> Accuracy: {accuracy:.4f}, Avg Latency: {avg_latency:.4f}s")
            return {"accuracy": accuracy, "latency": avg_latency}
        finally:
            # Restore original flags
            pipeline.cfg.show = orig_show
            pipeline.cfg.save = orig_save
