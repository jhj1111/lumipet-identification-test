import cv2
import os
from pathlib import Path
from typing import Union, Iterator, Optional

class StreamLoader:
    """
    Abstracts input sources: Camera (index), Video file, or Directory/Image.
    """
    def __init__(self, source: Union[str, Path, int] = 0):
        self.source = source
        self.cap = None
        self.is_video = False
        self.is_dir = False
        self.image_files = []
        
        self._setup_source()

    def _setup_source(self):
        source_str = str(self.source)
        if source_str.isdigit():
            self.cap = cv2.VideoCapture(int(source_str))
            self.is_video = True
        elif os.path.isfile(self.source):
            if source_str.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.cap = cv2.VideoCapture(source_str)
                self.is_video = True
            else:
                self.image_files = [self.source]
        elif os.path.isdir(self.source):
            self.is_dir = True
            self.image_files = sorted([
                os.path.join(source_str, f) for f in os.listdir(source_str)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])

    def __iter__(self) -> Iterator[tuple[Optional[str], cv2.Mat]]:
        if self.is_video:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                yield str(self.source), frame
            self.cap.release()
        else:
            for img_path in self.image_files:
                frame = cv2.imread(img_path)
                if frame is not None:
                    yield img_path, frame

    def get_fps(self) -> float:
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 30.0

    def get_size(self) -> tuple[int, int]:
        if self.cap:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (640, 480)
