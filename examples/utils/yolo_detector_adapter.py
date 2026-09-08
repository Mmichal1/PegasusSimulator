#!/usr/bin/env python
"""
| File: yolo_detector_adapter.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Adapters around YoloDroneTracker for richer vision outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from track import YoloDroneTracker, silence_botsort_noise

RgbImage = np.ndarray
PixelCoord = tuple[int, int]


@dataclass
class VisionDetection:
    centroid: PixelCoord | None
    bbox_width_px: float | None
    bbox_height_px: float | None
    debug_image: RgbImage | None


class YoloDroneDetectorAdapter:
    """Wraps YoloDroneTracker and also exposes bounding-box size for distance estimation."""

    def __init__(self, tracker: YoloDroneTracker):
        self._tracker = tracker

    def detect(self, rgb_image: RgbImage) -> VisionDetection:
        if rgb_image.dtype != np.uint8:
            rgb_image = (np.clip(rgb_image, 0.0, 1.0) * 255.0).astype(np.uint8)

        with silence_botsort_noise():
            results = self._tracker.model.track(
                source=rgb_image,
                tracker=self._tracker.tracker_config,
                conf=self._tracker.conf_thresh,
                persist=True,
                verbose=False,
            )

        centroid: PixelCoord | None = None
        bbox_width_px: float | None = None
        bbox_height_px: float | None = None
        debug_image = rgb_image.copy()

        if len(results) > 0:
            result = results[0]
            debug_image = result.plot()
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                best_box = None
                if boxes.id is not None:
                    ids = boxes.id.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    if (
                        self._tracker.locked_target_id is not None
                        and self._tracker.locked_target_id in ids
                    ):
                        idx = int(np.where(ids == self._tracker.locked_target_id)[0][0])
                        best_box = boxes[idx]
                    else:
                        idx = int(np.argmax(confs))
                        best_box = boxes[idx]
                        self._tracker.locked_target_id = int(ids[idx])
                else:
                    idx = int(np.argmax(boxes.conf.cpu().numpy()))
                    best_box = boxes[idx]

                if best_box is not None:
                    xyxy = best_box.xyxy[0].cpu().numpy()
                    x_center = int((xyxy[0] + xyxy[2]) / 2)
                    y_center = int((xyxy[1] + xyxy[3]) / 2)
                    centroid = (x_center, y_center)
                    bbox_width_px = float(xyxy[2] - xyxy[0])
                    bbox_height_px = float(xyxy[3] - xyxy[1])

        return VisionDetection(
            centroid=centroid,
            bbox_width_px=bbox_width_px,
            bbox_height_px=bbox_height_px,
            debug_image=debug_image,
        )

    def __call__(self, rgb_image: RgbImage) -> tuple[PixelCoord | None, RgbImage | None]:
        detection = self.detect(rgb_image)
        return detection.centroid, detection.debug_image
