import os
import time
import logging
import numpy as np
from contextlib import contextmanager
from typing import List, Optional, Generator

logger = logging.getLogger(__name__)

class PipelineProfiler:
    def __init__(self, enabled: bool = True, stages: Optional[List[str]] = None) -> None:
        self.enabled = enabled
        self.stages = stages or ["detection", "quality_filter", "extractor", "matcher", "rendering"]
        self.current_frame = {stage: 0.0 for stage in self.stages}
        self.current_frame["total"] = 0.0
        self.history = []

    @contextmanager
    def profile(self, stage: str) -> Generator[None, None, None]:
        if not self.enabled or stage not in self.stages:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0  # ms
            self.current_frame[stage] += elapsed

    def commit_frame(self, total_time_ms: float = 0.0) -> None:
        if not self.enabled:
            return
        if total_time_ms > 0.0:
            self.current_frame["total"] = total_time_ms
        else:
            self.current_frame["total"] = sum(self.current_frame[s] for s in self.stages)
        self.history.append(dict(self.current_frame))
        self.current_frame = {stage: 0.0 for stage in self.stages}
        self.current_frame["total"] = 0.0

    def get_summary_string(self) -> str:
        if not self.history:
            return "No profiling history recorded."
        
        summary = []
        summary.append("=" * 20 + " Re-ID Profiler Summary " + "=" * 21)
        summary.append(f"{'Stage':<18} | {'Avg Time (ms)':>13} | {'Std Dev (ms)':>12} | {'Ratio (%)':>9}")
        summary.append("-" * 65)
        
        totals = [frame["total"] for frame in self.history]
        mean_total = np.mean(totals)
        
        for stage in self.stages:
            times = [frame[stage] for frame in self.history]
            mean_val = np.mean(times)
            std_val = np.std(times)
            ratio = (mean_val / mean_total * 100.0) if mean_total > 0.0 else 0.0
            summary.append(f"{stage:<18} | {mean_val:>13.2f} | {std_val:>12.2f} | {ratio:>8.1f}%")
            
        summary.append("-" * 65)
        mean_fps = 1000.0 / mean_total if mean_total > 0.0 else 0.0
        summary.append(f"Total Frames: {len(self.history):<5} | Mean Frame Time: {mean_total:.2f} ms | Mean FPS: {mean_fps:.1f}")
        summary.append("=" * 65)
        return "\n".join(summary)

    def save_csv(self, source_name: str, timestamp: str) -> None:
        if not self.enabled or not self.history:
            return
        
        # Clean source name
        clean_source = source_name.replace(" ", "_").replace(".", "_").replace("/", "_").replace("\\", "_")
        out_dir = os.path.join("results", clean_source)
        os.makedirs(out_dir, exist_ok=True)
        
        csv_path = os.path.join(out_dir, f"{timestamp}.csv")
        headers = self.stages + ["total"]
        
        with open(csv_path, "w") as f:
            f.write(",".join(headers) + "\n")
            for frame in self.history:
                row = [f"{frame[h]:.4f}" for h in headers]
                f.write(",".join(row) + "\n")
        logger.info(f"Profiling logs saved to: {csv_path}")

