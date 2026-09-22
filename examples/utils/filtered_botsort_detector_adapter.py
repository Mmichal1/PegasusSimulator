#!/usr/bin/env python
"""Adapter from filtered_detection's BoT-SORT tracker to interceptor detections."""

from __future__ import annotations

from typing import Any

import numpy as np

from yolo_detector_adapter import VisionDetection

RgbImage = np.ndarray


class FilteredBotSortDetectorAdapter:
    """Expose BotSortFilteredTracker through the PN controller's detector API."""

    def __init__(self, tracker: Any):
        if not callable(tracker):
            raise ValueError("tracker must be callable")
        self._tracker = tracker

    def detect(self, rgb_image: RgbImage) -> VisionDetection:
        centroid, debug_image = self._tracker(rgb_image)

        bbox_width_px: float | None = None
        bbox_height_px: float | None = None
        bbox = self._tracker.bbox
        if centroid is not None and bbox is not None:
            bbox_width_px = float(bbox[2])
            bbox_height_px = float(bbox[3])

        return VisionDetection(
            centroid=centroid,
            bbox_width_px=bbox_width_px,
            bbox_height_px=bbox_height_px,
            debug_image=debug_image,
        )

    def __call__(self, rgb_image: RgbImage):
        """Compatibility with PluggableTorpedoVisionController's detector contract."""
        detection = self.detect(rgb_image)
        return detection.centroid, detection.debug_image

    def reset(self) -> None:
        """Clear BoT-SORT and filter state for a new episode."""
        self._tracker.reset()
