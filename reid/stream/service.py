import cv2
import time
from reid.stream.input import StreamLoader
from reid.stream.overlay import Renderer
from reid.core.config import get_config

class StreamService:
    """
    Orchestrates the frame loop: Input -> Pipeline -> Rendering -> Output.
    """
    def __init__(self, pipeline, config=None):
        self.pipeline = pipeline
        self.cfg = config or get_config()
        self.renderer = Renderer()
        self.video_writer = None

    def run(self, source=0):
        loader = StreamLoader(source)
        
        # Setup Video Writer if save=True
        if self.cfg.save:
            fps = loader.get_fps()
            w, h = loader.get_size()
            save_path = "output.mp4"
            self.video_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
            )
            print(f"Saving results to: {save_path}")

        try:
            for path, frame in loader:
                start_time = time.time()
                
                # 1. Pipeline Inference
                results = self.pipeline.predict(frame)
                
                # 2. Render Results
                annotated_frame = self.renderer.draw(results)
                
                # 3. Handle FPS display
                fps_val = 1.0 / (time.time() - start_time)
                cv2.putText(annotated_frame, f"FPS: {fps_val:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 4. Output: Show
                if self.cfg.show:
                    cv2.imshow("Lumipet Re-ID", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 5. Output: Save
                if self.video_writer:
                    self.video_writer.write(annotated_frame)

        finally:
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()
