import time
import numpy as np
from collections import defaultdict
from cat_reid.input.file import FileInference

class Evaluator:
    def __init__(self, inferencer=None):
        self.inferencer = inferencer if inferencer else FileInference()

    def evaluate(self, query_data):
        """
        Query 데이터셋에 대해 정확도와 평균 처리 시간을 측정합니다.
        query_data: [(image_path, label), ...]
        """
        correct = 0
        total = len(query_data)
        latencies = []
        
        # 고양이별 결과 통계
        cat_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        results = []
        
        print(f"Starting evaluation on {total} images...")
        
        for img_path, true_label in query_data:
            start_time = time.time()
            pred_label, score = self.inferencer.predict(img_path)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
            
            is_correct = (pred_label == true_label)
            if is_correct:
                correct += 1
                
            cat_stats[true_label]['total'] += 1
            if is_correct:
                cat_stats[true_label]['correct'] += 1
                
            results.append({
                'path': img_path,
                'true': true_label,
                'pred': pred_label,
                'score': score,
                'correct': is_correct
            })
            
        accuracy = correct / total if total > 0 else 0
        avg_latency = np.mean(latencies)
        
        print("\n" + "="*50)
        print(f"{'Cat ID':<30} | {'Accuracy':<10} | {'Count'}")
        print("-"*50)
        for label, stats in sorted(cat_stats.items()):
            cat_acc = stats['correct'] / stats['total']
            print(f"{label:<30} | {cat_acc:8.2%} | {stats['total']}")
        
        print("="*50)
        print("Overall Performance")
        print("-"*50)
        print(f"Total Images: {total}")
        print(f"Accuracy:     {accuracy:.4f} ({correct}/{total})")
        print(f"Avg Latency:  {avg_latency:.4f} sec/image")
        print(f"FPS (est.):   {1/avg_latency:.2f}")
        print("="*50)
        
        return accuracy, avg_latency, results
