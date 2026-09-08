#!/usr/bin/env python
"""
| File: vision_intercept.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Monocular distance approximation and image-plane intercept prediction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

PixelCoord = tuple[int, int]

# 3DR Iris motor-to-motor span (approximate visible width when viewed from below/side).
# Typical PX4 Iris quad is ~0.55 m between rotor centers.
IRIS_DRONE_WIDTH_M = 0.55


@dataclass
class InterceptEstimate:
    centroid: PixelCoord | None
    predicted_centroid: PixelCoord | None
    estimated_distance_m: float | None
    lead_time_s: float
    pixel_velocity: tuple[float, float] | None


class MonocularInterceptPredictor:
    """
    Estimate range from apparent target width and predict future image position.

    Range model (pinhole): distance = (fx * real_width) / bbox_width_px

    Pixel velocity is heavily filtered and ego-motion from the pursuer camera is
    subtracted so the prediction reflects target motion rather than own movement.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        target_width_m: float = IRIS_DRONE_WIDTH_M,
        fov_deg: float = 80.0,
        history_length: int = 24,
        min_bbox_width_px: float = 8.0,
        min_velocity_window_s: float = 0.45,
        min_lead_time_s: float = 0.35,
        max_lead_time_s: float = 2.5,
        pursuer_speed_mps: float = 3.0,
        velocity_smoothing: float = 0.12,
        distance_smoothing: float = 0.15,
        lead_time_smoothing: float = 0.1,
        prediction_smoothing: float = 0.15,
        max_prediction_offset_px: float = 70.0,
        ego_motion_scale: float = 1.0,
    ):
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.target_width_m = float(target_width_m)
        self.fov_deg = float(fov_deg)
        self.history_length = int(history_length)
        self.min_bbox_width_px = float(min_bbox_width_px)
        self.min_velocity_window_s = float(min_velocity_window_s)
        self.min_lead_time_s = float(min_lead_time_s)
        self.max_lead_time_s = float(max_lead_time_s)
        self.pursuer_speed_mps = float(pursuer_speed_mps)
        self.velocity_smoothing = float(velocity_smoothing)
        self.distance_smoothing = float(distance_smoothing)
        self.lead_time_smoothing = float(lead_time_smoothing)
        self.prediction_smoothing = float(prediction_smoothing)
        self.max_prediction_offset_px = float(max_prediction_offset_px)
        self.ego_motion_scale = float(ego_motion_scale)

        half_fov = np.radians(self.fov_deg * 0.5)
        self.fx = 0.5 * self.image_width / np.tan(half_fov)

        self._history: deque[tuple[float, float, float, float | None]] = deque(
            maxlen=self.history_length
        )
        self._filtered_pixel_velocity: np.ndarray | None = None
        self._filtered_distance: float | None = None
        self._filtered_lead_time: float = 0.0
        self._filtered_prediction: np.ndarray | None = None

    def reset(self) -> None:
        self._history.clear()
        self._filtered_pixel_velocity = None
        self._filtered_distance = None
        self._filtered_lead_time = 0.0
        self._filtered_prediction = None

    def _smooth_scalar(self, previous: float | None, measurement: float, alpha: float) -> float:
        if previous is None:
            return float(measurement)
        return float((1.0 - alpha) * previous + alpha * measurement)

    def _smooth_vector(
        self, previous: np.ndarray | None, measurement: np.ndarray, alpha: float
    ) -> np.ndarray:
        if previous is None:
            return measurement.astype(float)
        return (1.0 - alpha) * previous + alpha * measurement

    def _estimate_distance(self, bbox_width_px: float | None) -> float | None:
        if bbox_width_px is None or bbox_width_px < self.min_bbox_width_px:
            return self._filtered_distance
        raw = float((self.fx * self.target_width_m) / bbox_width_px)
        self._filtered_distance = self._smooth_scalar(self._filtered_distance, raw, self.distance_smoothing)
        return self._filtered_distance

    def _pixel_velocity_raw(self) -> tuple[float, float] | None:
        if len(self._history) < 2:
            return None

        t1, u1, v1, _ = self._history[-1]
        t0, u0, v0 = t1, u1, v1
        for sample in reversed(self._history):
            if t1 - sample[0] >= self.min_velocity_window_s:
                t0, u0, v0, _ = sample
                break
            t0, u0, v0, _ = sample

        dt = t1 - t0
        if dt < 0.5 * self.min_velocity_window_s:
            return None
        return ((u1 - u0) / dt, (v1 - v0) / dt)

    def _compensate_ego_motion(
        self,
        pixel_velocity: tuple[float, float],
        distance_m: float,
        pursuer_velocity: np.ndarray | None,
    ) -> tuple[float, float]:
        if pursuer_velocity is None:
            return pixel_velocity

        du, dv = pixel_velocity
        d = max(distance_m, 1.0)
        scale = self.fx / d

        # Approximate world-frame ego motion projected into the upward camera image.
        vx, vy, _ = np.asarray(pursuer_velocity, dtype=float).reshape(3)
        du -= self.ego_motion_scale * scale * vx
        dv -= self.ego_motion_scale * scale * vy
        return (du, dv)

    def _filter_pixel_velocity(self, measurement: tuple[float, float] | None) -> tuple[float, float] | None:
        if measurement is None:
            return (
                (float(self._filtered_pixel_velocity[0]), float(self._filtered_pixel_velocity[1]))
                if self._filtered_pixel_velocity is not None
                else None
            )

        vec = np.array(measurement, dtype=float)
        self._filtered_pixel_velocity = self._smooth_vector(
            self._filtered_pixel_velocity,
            vec,
            self.velocity_smoothing,
        )
        return (float(self._filtered_pixel_velocity[0]), float(self._filtered_pixel_velocity[1]))

    def _limit_prediction_offset(
        self,
        centroid: PixelCoord,
        predicted_xy: np.ndarray,
    ) -> np.ndarray:
        offset = predicted_xy - np.array(centroid, dtype=float)
        norm = float(np.linalg.norm(offset))
        if norm > self.max_prediction_offset_px > 0.0:
            offset *= self.max_prediction_offset_px / norm
        return np.array(centroid, dtype=float) + offset

    def _smooth_prediction(self, centroid: PixelCoord, raw_prediction: np.ndarray) -> PixelCoord:
        limited = self._limit_prediction_offset(centroid, raw_prediction)
        self._filtered_prediction = self._smooth_vector(
            self._filtered_prediction,
            limited,
            self.prediction_smoothing,
        )
        u = int(np.clip(round(self._filtered_prediction[0]), 0, self.image_width - 1))
        v = int(np.clip(round(self._filtered_prediction[1]), 0, self.image_height - 1))
        return (u, v)

    def update(
        self,
        sim_time: float,
        centroid: PixelCoord | None,
        bbox_width_px: float | None,
        pursuer_velocity: np.ndarray | None = None,
    ) -> InterceptEstimate:
        if centroid is None:
            self.reset()
            return InterceptEstimate(
                centroid=None,
                predicted_centroid=None,
                estimated_distance_m=None,
                lead_time_s=0.0,
                pixel_velocity=None,
            )

        self._history.append((sim_time, float(centroid[0]), float(centroid[1]), bbox_width_px))

        distance = self._estimate_distance(bbox_width_px)
        raw_pixel_vel = self._pixel_velocity_raw()

        if distance is None:
            return InterceptEstimate(
                centroid=centroid,
                predicted_centroid=centroid,
                estimated_distance_m=None,
                lead_time_s=self._filtered_lead_time,
                pixel_velocity=self._filter_pixel_velocity(raw_pixel_vel),
            )

        if raw_pixel_vel is not None:
            compensated = self._compensate_ego_motion(raw_pixel_vel, distance, pursuer_velocity)
        else:
            compensated = None

        pixel_vel = self._filter_pixel_velocity(compensated)

        raw_lead = float(np.clip(distance / self.pursuer_speed_mps, self.min_lead_time_s, self.max_lead_time_s))
        self._filtered_lead_time = self._smooth_scalar(
            self._filtered_lead_time if self._filtered_lead_time > 0.0 else None,
            raw_lead,
            self.lead_time_smoothing,
        )
        lead_time = self._filtered_lead_time

        if pixel_vel is None:
            predicted = self._smooth_prediction(centroid, np.array(centroid, dtype=float))
        else:
            du_dt, dv_dt = pixel_vel
            raw_xy = np.array(
                [
                    centroid[0] + du_dt * lead_time,
                    centroid[1] + dv_dt * lead_time,
                ],
                dtype=float,
            )
            predicted = self._smooth_prediction(centroid, raw_xy)

        return InterceptEstimate(
            centroid=centroid,
            predicted_centroid=predicted,
            estimated_distance_m=distance,
            lead_time_s=lead_time,
            pixel_velocity=pixel_vel,
        )
