from abc import ABC, abstractmethod
from typing import Any, Optional
from pathlib import Path
import numpy as np
import cv2
import time
from reid.core.config import get_config
from reid.stream.input import StreamLoader


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

    def predict(self, source: Any) -> Any:
        """Perform predictions on single input or stream loader sources."""
        import os

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
                annotated_frame = self.draw_overlay(res, frame)

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

        # from reid.stream.input import StreamLoader
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
        last_annotated_frame = None

        results_list = []
        try:
            for path, frame in self.loader:
                start_time = time.time()

                # 1. Run inference
                res = self.predict_once(frame)
                if res is not None : results_list.append(res)

                # 2. Draw overlay if results contains bounding boxes
                if hasattr(res, 'boxes'):
                    annotated_frame = self.draw_overlay(res, frame)

                    # Add FPS count on top (smoothed with EMA)
                    current_fps = 1.0 / max(time.time() - start_time, 1e-6)
                    if self.fps_ema is None:
                        self.fps_ema = current_fps
                    else:
                        self.fps_ema = 0.9 * self.fps_ema + 0.1 * current_fps
                    cv2.putText(annotated_frame, f"FPS: {self.fps_ema:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    last_annotated_frame = annotated_frame

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
                                # Use a flag since we can't `break` an outer for-loop from here
                                results_list_break = True
                                break
                        else:
                            results_list_break = False

                        if 'results_list_break' in dir() and results_list_break:
                            break

                    # Write if save=True
                    if self.video_writer:
                        self.video_writer.write(annotated_frame)
        finally:
            if self.video_writer:
                self.video_writer.release()
            if self.cfg.show:
                cv2.destroyAllWindows()

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

    def draw_overlay(self, results: Any, frame: np.ndarray) -> np.ndarray:
        """Render detections and classifications using overlay Renderer."""
        from reid.stream.overlay import Renderer
        dev_mode = getattr(self.cfg, 'dev', True)
        renderer = Renderer(dev=dev_mode)
        return renderer.draw(results)
