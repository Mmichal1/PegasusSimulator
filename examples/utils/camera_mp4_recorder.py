#!/usr/bin/env python
"""
| File: camera_mp4_recorder.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Write a raw onboard-camera RGB stream to an MP4 (no overlays).
"""

from __future__ import annotations

import os

import numpy as np

from camera_preview import import_cv2


class CameraMp4Recorder:
    """Append raw camera frames to an MP4 via OpenCV VideoWriter."""

    def __init__(self, path: str, fps: float = 30.0):
        self.path = os.path.abspath(path)
        self.fps = float(fps)
        self.frame_count = 0
        self._writer = None
        self._size: tuple[int, int] | None = None
        self._cv2 = None

    def write(self, rgba: np.ndarray | None) -> None:
        rgb = _rgb_from_rgba(rgba)
        if rgb is None:
            return

        height, width = rgb.shape[:2]
        if self._writer is None:
            self._open(width, height)
        if self._writer is None or self._size != (width, height):
            return

        bgr = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)
        self.frame_count += 1

    def _open(self, width: int, height: int) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._cv2 = import_cv2()
        self._size = (width, height)
        for codec in ("mp4v", "avc1", "XVID"):
            fourcc = self._cv2.VideoWriter_fourcc(*codec)
            writer = self._cv2.VideoWriter(self.path, fourcc, self.fps, (width, height))
            if writer is not None and writer.isOpened():
                self._writer = writer
                return
            if writer is not None:
                writer.release()
        raise RuntimeError(
            f"OpenCV could not open an MP4 writer at {self.path} "
            f"({width}x{height} @ {self.fps:.1f} fps)"
        )

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def _rgb_from_rgba(rgba: np.ndarray | None) -> np.ndarray | None:
    if rgba is None or len(getattr(rgba, "shape", ())) < 3:
        return None
    rgb = np.asarray(rgba[..., :3])
    if rgb.dtype != np.uint8:
        if rgb.size and float(rgb.max()) <= 1.0:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)
