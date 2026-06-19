from abc import ABC, abstractmethod
from typing import Any, Optional
from pathlib import Path
import numpy as np
import cv2
import time
from reid.core.config import get_config


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

    def setup_model(self, model: Any) -> None:
        """Bind model to device and configure eval mode."""
        self.model = model
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()

    def __call__(self, source: Any) -> Any:
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
                if self.cfg.save:
                    cv2.imwrite("output_result.png", annotated_frame)
                    print("Saved prediction result to: output_result.png")
                if self.cfg.show:
                    cv2.imshow("Lumipet Re-ID", annotated_frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            return res

        from reid.stream.input import StreamLoader
        loader = StreamLoader(source)

        # Setup video writer output if save=True
        if self.cfg.save:
            fps = loader.get_fps()
            w, h = loader.get_size()
            save_path = "output.mp4"
            self.video_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
            )
            print(f"Saving results to: {save_path}")

        results_list = []
        try:
            for path, frame in loader:
                start_time = time.time()

                # 1. Run inference
                res = self.predict_once(frame)
                results_list.append(res)

                # 2. Draw overlay if results contains bounding boxes
                if hasattr(res, 'boxes'):
                    annotated_frame = self.draw_overlay(res, frame)

                    # Add FPS count on top
                    fps_val = 1.0 / max(time.time() - start_time, 1e-6)
                    cv2.putText(annotated_frame, f"FPS: {fps_val:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Display if show=True
                    if self.cfg.show:
                        cv2.imshow("Lumipet Re-ID", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
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
        results = self.postprocess(preds, im_prepped, im)
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
        renderer = Renderer()
        return renderer.draw(results)
