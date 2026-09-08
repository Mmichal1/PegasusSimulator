#!/usr/bin/env python
"""
| File: upward_intercept_torpedo_controller_pluggable.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Upward pursuit with monocular intercept prediction.
"""

from __future__ import annotations

from typing import Any, Protocol

import carb
import numpy as np

from upward_torpedo_controller_pluggable import UpwardTorpedoVisionController
from vision_intercept import MonocularInterceptPredictor, IRIS_DRONE_WIDTH_M

try:
    from camera_preview import import_cv2
except ImportError:  # pragma: no cover
    import_cv2 = None

RgbImage = np.ndarray
PixelCoord = tuple[int, int]


class RichTargetDetector(Protocol):
    def detect(self, rgb_image: RgbImage) -> Any: ...


def _draw_intercept_overlay(
    rgb: RgbImage,
    centroid: PixelCoord | None,
    predicted: PixelCoord | None,
    distance_m: float | None,
    lead_time_s: float,
) -> RgbImage:
    if import_cv2 is None:
        return rgb

    cv2 = import_cv2()
    out = np.asarray(rgb[..., :3]).copy()
    if out.dtype != np.uint8:
        if out.max() <= 1.0:
            out = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            out = out.astype(np.uint8)

    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.drawMarker(bgr, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)

    if centroid is not None:
        u, v = centroid
        cv2.circle(bgr, (int(u), int(v)), 10, (0, 255, 0), 2)
        cv2.line(bgr, (cx, cy), (int(u), int(v)), (255, 200, 0), 2)

    if predicted is not None and predicted != centroid:
        pu, pv = predicted
        cv2.circle(bgr, (int(pu), int(pv)), 12, (255, 0, 255), 2)
        if centroid is not None:
            cv2.arrowedLine(
                bgr,
                (int(centroid[0]), int(centroid[1])),
                (int(pu), int(pv)),
                (255, 0, 255),
                2,
                tipLength=0.3,
            )

    if distance_m is not None:
        label = f"~{distance_m:.1f} m  lead {lead_time_s:.2f}s"
        cv2.putText(bgr, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class UpwardInterceptTorpedoVisionController(UpwardTorpedoVisionController):
    """
    Upward pursuit that steers toward a predicted image position.

    Requires a detector exposing `detect(rgb) -> VisionDetection` with bbox width.
    """

    def __init__(
        self,
        detector: RichTargetDetector,
        target_width_m: float = IRIS_DRONE_WIDTH_M,
        fov_deg: float = 80.0,
        predictor_history_length: int = 24,
        min_velocity_window_s: float = 0.45,
        velocity_smoothing: float = 0.12,
        prediction_smoothing: float = 0.15,
        max_prediction_offset_px: float = 70.0,
        intercept_lateral_gain: float = 0.35,
        intercept_climb_gain: float = 0.25,
        velocity_feedforward_gain: float = 0.45,
        **kwargs: Any,
    ):
        if detector is None or not callable(detector):
            raise ValueError("detector must be callable and expose detect(rgb_image)")
        if not callable(getattr(detector, "detect", None)):
            raise ValueError("detector must expose detect(rgb_image) -> VisionDetection")

        kwargs.setdefault("k_lateral", intercept_lateral_gain)
        super().__init__(detector=detector, **kwargs)

        self.intercept_climb_gain = float(intercept_climb_gain)
        self.velocity_feedforward_gain = float(velocity_feedforward_gain)

        self._rich_detector: RichTargetDetector = detector
        self._predictor = MonocularInterceptPredictor(
            image_width=self.image_width,
            image_height=self.image_height,
            target_width_m=target_width_m,
            fov_deg=fov_deg,
            history_length=predictor_history_length,
            min_velocity_window_s=min_velocity_window_s,
            velocity_smoothing=velocity_smoothing,
            prediction_smoothing=prediction_smoothing,
            max_prediction_offset_px=max_prediction_offset_px,
            pursuer_speed_mps=self.climb_speed * self.speed_boost_factor,
        )

        self._steer_centroid: PixelCoord | None = None
        self._estimated_distance_m: float | None = None
        self._lead_time_s: float = 0.0
        self._predicted_centroid: PixelCoord | None = None
        self._pixel_velocity: tuple[float, float] | None = None

    def reset_episode(self) -> None:
        super().reset_episode()
        self._predictor.reset()
        self._steer_centroid = None
        self._estimated_distance_m = None
        self._lead_time_s = 0.0
        self._predicted_centroid = None
        self._pixel_velocity = None

    def _predicted_pixel_errors(self) -> tuple[float, float] | None:
        if self._steer_centroid is None:
            return None

        u, v = self._steer_centroid
        ex = (float(u) - 0.5 * self.image_width) / (0.5 * self.image_width)
        ey = (float(v) - 0.5 * self.image_height) / (0.5 * self.image_height)
        return ex, ey

    def update_graphical_sensor(self, sensor_type: str, data: dict) -> None:
        if sensor_type != "MonocularCamera" or not data:
            return

        camera = data.get("camera")
        if camera is None:
            return

        try:
            rgba = camera.get_rgba()
            if rgba is None or len(getattr(rgba, "shape", ())) < 3:
                return

            rgb = np.asarray(rgba[..., :3])
            if rgb.dtype != np.uint8:
                if rgb.max() <= 1.0:
                    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)

            h, w = rgb.shape[:2]
            if w != self.image_width or h != self.image_height:
                self.image_width, self.image_height = w, h
                self._predictor.image_width = w
                self._predictor.image_height = h
                half_fov = np.radians(self._predictor.fov_deg * 0.5)
                self._predictor.fx = 0.5 * w / np.tan(half_fov)

            detection = self._rich_detector.detect(rgb)
            self._centroid = detection.centroid
            self._has_detection = detection.centroid is not None

            estimate = self._predictor.update(
                self._sim_time,
                detection.centroid,
                detection.bbox_width_px,
                pursuer_velocity=self.v if self.reveived_first_state else None,
            )
            self._steer_centroid = estimate.predicted_centroid or detection.centroid
            self._estimated_distance_m = estimate.estimated_distance_m
            self._lead_time_s = estimate.lead_time_s
            self._predicted_centroid = estimate.predicted_centroid
            self._pixel_velocity = estimate.pixel_velocity

            if self.show_debug_feed:
                self._ensure_debug_preview()
                if self._camera_preview is not None:
                    debug_image = detection.debug_image
                    if debug_image is None:
                        debug_image = rgb
                    debug_image = _draw_intercept_overlay(
                        debug_image,
                        detection.centroid,
                        estimate.predicted_centroid,
                        estimate.estimated_distance_m,
                        estimate.lead_time_s,
                    )
                    self._camera_preview.update(debug_image)
        except Exception as exc:
            carb.log_warn(f"Intercept detector failed: {exc}")

    def _vision_references(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        if self._yaw_ref is None:
            self._init_yaw_reference()

        errors = self._predicted_pixel_errors()
        if not self._has_detection or errors is None:
            return self._patrol_references(dt)

        ex, ey = errors
        v_ref = self._bearing_velocity_from_image_errors(ex, ey)
        v_ref[2] += float(self.intercept_climb_gain * self._soften_error(-ey))

        if self._pixel_velocity is not None and self._estimated_distance_m is not None:
            du_dt, dv_dt = self._pixel_velocity
            meters_per_pixel = self._estimated_distance_m / self._predictor.fx
            cam_u, cam_v = self._camera_axes_in_world()
            feedforward = cam_u * (du_dt * meters_per_pixel) + cam_v * (dv_dt * meters_per_pixel)
            ff_norm = float(np.linalg.norm(feedforward))
            max_ff = self.max_horizontal_speed * 0.5
            if ff_norm > max_ff > 0.0:
                feedforward *= max_ff / ff_norm
            v_ref += self.velocity_feedforward_gain * feedforward
            v_ref = self._smooth_velocity_reference(v_ref)

        p_ref = self.p + v_ref * dt
        p_ref[2] = float(np.clip(p_ref[2], self.cruise_height, self.max_altitude))
        self._last_status = "intercepting"

        return p_ref, v_ref, np.zeros(3), np.zeros(3), float(self._yaw_ref), 0.0
