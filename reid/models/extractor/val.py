import time
import numpy as np
from tqdm import tqdm
from reid.engine.validator import BaseValidator

class ExtractorValidator(BaseValidator):
    """
    Validator for Re-ID performance (Accuracy & Latency).
    """
    def validate(self, data, **kwargs):
        """
        data: [(img_path, true_label), ...]
        kwargs['pipeline']: The ReIdModel instance to use for prediction.
        """
        pipeline = kwargs.get('pipeline')
        if pipeline is None:
            raise ValueError("Pipeline instance is required for validation.")
            
        correct = 0
        total = len(data)
        latencies = []
        
        print(f"Validating on {total} images...")
        for img_path, true_label in tqdm(data):
            start_time = time.time()
            
            # Use the pipeline to predict
            results = pipeline.predict(img_path)
            
            latencies.append(time.time() - start_time)
            
            if len(results.match_results) > 0:
                pred_label = results.match_results[0].cat_id
                if pred_label == true_label:
                    correct += 1
            
        accuracy = correct / total if total > 0 else 0
        avg_latency = np.mean(latencies)
        
        print(f"Result -> Accuracy: {accuracy:.4f}, Avg Latency: {avg_latency:.4f}s")
        return {"accuracy": accuracy, "latency": avg_latency}
