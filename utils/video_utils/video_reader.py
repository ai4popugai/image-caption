import cv2
import numpy as np


class VideoReader:
    """
    If video ends, reader starts to read frames from 0.
    """

    def __init__(self, video_path: str, fps: int, scale_factor: float = 1.):
        super().__init__()
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor), \
                                  int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_factor)
        self.frames_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0

    def _get_next_frame(self) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        _, frame = self.cap.read()
        frame = cv2.resize(frame, (self.width, self.height))
        self.frame_idx += 1
        return frame

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self.frame_idx >= self.frames_count:
            raise StopIteration
        frame = self._get_next_frame()
        return frame
