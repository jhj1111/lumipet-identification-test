from abc import ABC, abstractmethod
from typing import Any, Optional, Union
from pathlib import Path
import numpy as np
import cv2
import time
from reid.core.config import get_config
from reid.stream.input import StreamLoader
from reid.utils.profiler import PipelineProfiler


class BasePredictor(ABC):
    """
    Abstract Base Class for all Predictors.
    Manages the inference loop over frames/sources, overlay rendering, and video saving.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        self.cfg = config or get_config()
        self.device = self.cfg.device
        self.model = None
        self.video_writer = None
        self.fps_ema = None
        self.loader: Optional[StreamLoader] = None
        self.profiler = PipelineProfiler(enabled=getattr(self.cfg, "dev", True))

    def setup_model(self, model: Any) -> None:
        """Bind model to device and configure eval mode."""
        self.model = model
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()

    def __call__(self, source: Any) -> Any:
        # self.loader = StreamLoader(source)
        return self.predict(source)

    def predict(self, source: Union[str, Path, int]) -> Any:
        """Perform predictions on single input or stream loader sources."""
        import os

        # Reset state/history for this specific run
        self.video_writer = None
        self.fps_ema = None
        if self.profiler:
            self.profiler.history = []
            self.profiler.current_frame = {stage: 0.0 for stage in self.profiler.stages}
            self.profiler.current_frame["total"] = 0.0

        # Identify source_name and timestamp
        source_str = str(source)
        if isinstance(source, int):
            source_name = f"webcam_{source}"
        else:
            source_name = os.path.splitext(os.path.basename(source_str))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Reset cache/state if the subclass supports it (e.g. ReIdPredictor)
        if hasattr(self, "reset"):
            self.reset()

        # Check if the source is a single image file path
        is_image_file = False
        if isinstance(source, (str, Path)):
            source_str = str(source)
            if os.path.isfile(source_str) and source_str.lower().endswith(('.png', '.jpg', '.jpeg')):
                is_image_file = True

        # Setup resizable display window
        if self.cfg.show:
            cv2.namedWindow("Lumipet Re-ID", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Lumipet Re-ID", 1280, 720)

        # Execute single-frame inference if input is raw image pixels or a single image file path
        if not isinstance(source, (str, Path, int)) or is_image_file:
            if is_image_file:
                frame = cv2.imread(str(source))
                if frame is None:
                    raise ValueError(f"Could not read image file: {source}")
            else:
                frame = source

            res = self.predict_once(frame)

            # Draw overlay and show/save if Results object is returned
            if hasattr(res, 'boxes') and (self.cfg.show or self.cfg.save):
                annotated_frame = self.draw_overlay(res)

                if self.cfg.show and is_image_file:
                    cv2.imshow("Lumipet Re-ID", annotated_frame)
                    print("Press 's' to save, 'q' to quit (no save).")
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('s'):
                            cv2.imwrite("output_result.png", annotated_frame)
                            print("Saved prediction result to: output_result.png")
                            break
                        elif key == ord('q'):
                            break
                    cv2.destroyAllWindows()
                elif self.cfg.save:
                    # show=False but save=True: save unconditionally, no key wait
                    cv2.imwrite("output_result.png", annotated_frame)
                    print("Saved prediction result to: output_result.png")

            return res

        self.loader = StreamLoader(source)

        # Setup video writer output if save=True
        if self.cfg.save:
            fps = self.loader.get_fps()
            w, h = self.loader.get_size()
            save_path = "output.mp4"
            self.video_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
            )
            print(f"Saving results to: {save_path}")

        paused = False
        results_list = []
        results_list_break = False

        try:
            for path, frame in self.loader:
                start_time = time.perf_counter()

                # 1. Run inference
                res = self.predict_once(frame)
                if res is not None:
                    results_list.append(res)

                annotated_frame = None
                if hasattr(res, 'boxes'):
                    annotated_frame = self.draw_overlay(res)

                # 2. Record processing execution time (excludes display waits and I/O)
                total_time_ms = (time.perf_counter() - start_time) * 1000.0
                self.profiler.commit_frame(total_time_ms)

                # 3. Post-processing visualization, saving, and GUI updates
                if annotated_frame is not None:
                    # Calculate FPS based on monotonic timers and actual processing time
                    current_fps = 1000.0 / max(total_time_ms, 1e-3)
                    if self.fps_ema is None:
                        self.fps_ema = current_fps
                    else:
                        self.fps_ema = 0.9 * self.fps_ema + 0.1 * current_fps

                    cv2.putText(annotated_frame, f"FPS: {self.fps_ema:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Display if show=True
                    if self.cfg.show:
                        cv2.imshow("Lumipet Re-ID", annotated_frame)
                        key = cv2.waitKey(int(self.loader.get_fps())) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('p'):
                            paused = True

                        # Pause loop: keep window responsive, wait for 'p' (resume) or 'q' (quit)
                        while paused:
                            key2 = cv2.waitKey(50) & 0xFF
                            if key2 == ord('p'):
                                paused = False
                            elif key2 == ord('q'):
                                paused = False
                                results_list_break = True
                                break

                        if results_list_break:
                            break

                    # Write if save=True
                    if self.video_writer:
                        self.video_writer.write(annotated_frame)

        finally:
            # Safe resource cleanup and reference deletion
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            if self.cfg.show:
                cv2.destroyAllWindows()
            if self.profiler and self.profiler.enabled:
                print(self.profiler.get_summary_string())
                self.profiler.save_csv(source_name, timestamp)


        return results_list

    def predict_once(self, im: Any) -> Any:
        """Execute single-item preprocess -> inference -> postprocess pipeline."""
        im_prepped = self.preprocess(im)
        preds = self.inference(im_prepped)
        results = self.postprocess(preds=preds, img=im_prepped, orig_img=im)
        return results

    @abstractmethod
    def preprocess(self, im: Any) -> Any:
        pass

    @abstractmethod
    def inference(self, im: Any) -> Any:
        pass

    @abstractmethod
    def postprocess(self, preds: Any, img: Any, orig_img: Any) -> Any:
        pass

    def draw_overlay(self, results: Any) -> np.ndarray:
        """Render detections and classifications using overlay Renderer."""
        from reid.stream.overlay import Renderer
        dev_mode = getattr(self.cfg, 'dev', True)
        renderer = Renderer(dev=dev_mode)
        return renderer.draw(results)
