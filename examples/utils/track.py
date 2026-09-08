import logging
import re
import sys
import warnings
from contextlib import contextmanager

import numpy as np
from ultralytics import YOLO

_MATCHING_POINTS_RE = re.compile(r"not enough matching points", re.IGNORECASE)


@contextmanager
def silence_botsort_noise():
    """Suppress noisy BoTSORT tracker warnings during YOLO tracking."""
    warnings.filterwarnings("ignore", message=".*matching points.*")
    warnings.filterwarnings("ignore", message=".*not enough matching.*")

    for logger_name in ("ultralytics", "boxmot", "torch"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    class _FilteredStream:
        def __init__(self, stream):
            self._stream = stream

        def write(self, text: str) -> int:
            if text and _MATCHING_POINTS_RE.search(text):
                return len(text)
            return self._stream.write(text)

        def flush(self) -> None:
            self._stream.flush()

        def __getattr__(self, name):
            return getattr(self._stream, name)

    old_stderr = sys.stderr
    old_stdout = sys.stdout
    sys.stderr = _FilteredStream(old_stderr)
    sys.stdout = _FilteredStream(old_stdout)
    try:
        yield
    finally:
        sys.stderr = old_stderr
        sys.stdout = old_stdout


class YoloDroneTracker:
    def __init__(self, model_path: str, tracker_config: str = "botsort.yaml", conf_thresh: float = 0.3):

        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.conf_thresh = conf_thresh
        self.locked_target_id = None

    def __call__(self, rgb_image: np.ndarray) -> tuple[tuple[int, int] | None, np.ndarray | None]:

        if rgb_image.dtype != np.uint8:
            rgb_image = (np.clip(rgb_image, 0.0, 1.0) * 255.0).astype(np.uint8)

        with silence_botsort_noise():
            results = self.model.track(
                source=rgb_image,
                tracker=self.tracker_config,
                conf=self.conf_thresh,
                persist=True,
                verbose=False,
            )

        centroid = None
        debug_image = None
        
        if len(results) > 0:
            result = results[0]
            debug_image = result.plot() 
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                best_box = None
                if boxes.id is not None:
                    ids = boxes.id.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()

                    if self.locked_target_id is not None and self.locked_target_id in ids:
                        idx = np.where(ids == self.locked_target_id)[0][0]
                        best_box = boxes[idx]
                    else:
                        idx = np.argmax(confs)
                        best_box = boxes[idx]
                        self.locked_target_id = ids[idx] # Zapisujemy ID do śledzenia w kolejnych klatkach
                else:
                    idx = np.argmax(boxes.conf.cpu().numpy())
                    best_box = boxes[idx]
                
                if best_box is not None:
                    xyxy = best_box.xyxy[0].cpu().numpy()
                    x_center = int((xyxy[0] + xyxy[2]) / 2)
                    y_center = int((xyxy[1] + xyxy[3]) / 2)
                    
                    centroid = (x_center, y_center)
        
        if debug_image is None:
            debug_image = rgb_image.copy()

        return centroid, debug_image