#!/usr/bin/env python
"""
| File: world_intercept.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Vision-only world-frame target state estimation and constant-velocity
| intercept solving for a forward-facing interceptor drone.
|
| The pursuer only observes the target through its monocular camera. From a single
| bounding box we recover:
|   - range   : from apparent width via the pinhole model  d = fx * real_width / bbox_px
|   - bearing : from the pixel offset of the centroid combined with the pursuer heading
| These give a full 3D target position in the world frame. Differencing filtered
| positions over time yields the target velocity, which feeds a constant-velocity
| intercept solver so the pursuer can aim at where the target *will be* instead of
| flying straight at where it *is* (which overshoots against a crossing target).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 3DR Iris motor-to-motor span (approximate visible width). Kept consistent with
# vision_intercept.IRIS_DRONE_WIDTH_M so both predictors use the same target model.
IRIS_DRONE_WIDTH_M = 0.55

PixelCoord = tuple[int, int]


@dataclass
class TargetEstimate:
    """World-frame estimate of the target produced from a single camera frame."""

    position: np.ndarray            # filtered target position in world frame [m]
    velocity: np.ndarray            # filtered target velocity in world frame [m/s]
    distance_m: float               # straight-line range from pursuer to target [m]
    bearing_unit: np.ndarray        # unit vector (world) from pursuer toward target
    velocity_valid: bool            # True once enough samples exist for a velocity estimate


def solve_intercept(
    pursuer_position: np.ndarray,
    pursuer_speed: float,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    max_time_s: float = 12.0,
) -> tuple[np.ndarray | None, float | None]:
    """
    Constant-velocity intercept solution.

    Find the earliest time ``t >= 0`` at which a pursuer travelling at constant speed
    ``pursuer_speed`` from ``pursuer_position`` can reach a target that moves along a
    straight line ``target_position + target_velocity * t``.

    Solving ``|d + v_t * t| = s * t`` gives a quadratic ``a t^2 + b t + c = 0`` with
        a = |v_t|^2 - s^2
        b = 2 (d . v_t)
        c = |d|^2
    where ``d = target_position - pursuer_position``.

    Returns:
        (aim_point, time_to_intercept) or (None, None) if no feasible solution exists
        (e.g. the pursuer is slower than a receding target).
    """
    d = np.asarray(target_position, dtype=float) - np.asarray(pursuer_position, dtype=float)
    v_t = np.asarray(target_velocity, dtype=float)
    speed = float(pursuer_speed)

    a = float(v_t @ v_t) - speed * speed
    b = 2.0 * float(d @ v_t)
    c = float(d @ d)

    t: float | None = None

    if abs(a) < 1e-6:
        # Target speed ~= pursuer speed: falls back to a linear equation b t + c = 0.
        if abs(b) > 1e-9:
            candidate = -c / b
            if candidate > 1e-3:
                t = candidate
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            sqrt_disc = float(np.sqrt(discriminant))
            roots = ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a))
            positive_roots = [r for r in roots if r > 1e-3]
            if positive_roots:
                t = min(positive_roots)

    if t is None or t > max_time_s:
        return None, None

    aim_point = np.asarray(target_position, dtype=float) + v_t * t
    return aim_point, float(t)


class TargetWorldEstimator:
    """
    Recover a world-frame target position/velocity from monocular detections.

    Assumptions:
      - The camera boresight is aligned with the pursuer body forward axis (+X body),
        matching the forward-camera convention used by the torpedo examples.
      - Square pixels (fy == fx), so the vertical field of view follows from the image
        aspect ratio.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        fov_deg: float = 70.0,
        target_width_m: float = IRIS_DRONE_WIDTH_M,
        position_smoothing: float = 0.35,
        velocity_smoothing: float = 0.2,
        min_bbox_width_px: float = 6.0,
        max_velocity_mps: float = 12.0,
    ):
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.fov_deg = float(fov_deg)
        self.target_width_m = float(target_width_m)
        self.position_smoothing = float(position_smoothing)
        self.velocity_smoothing = float(velocity_smoothing)
        self.min_bbox_width_px = float(min_bbox_width_px)
        self.max_velocity_mps = float(max_velocity_mps)

        self.fx = 0.5 * self.image_width / np.tan(np.radians(self.fov_deg * 0.5))

        self._last_time: float | None = None
        self._last_raw_position: np.ndarray | None = None
        self._filtered_position: np.ndarray | None = None
        self._filtered_velocity: np.ndarray | None = None

    def set_image_size(self, width: int, height: int) -> None:
        if width == self.image_width and height == self.image_height:
            return
        self.image_width = int(width)
        self.image_height = int(height)
        self.fx = 0.5 * self.image_width / np.tan(np.radians(self.fov_deg * 0.5))

    def reset(self) -> None:
        self._last_time = None
        self._last_raw_position = None
        self._filtered_position = None
        self._filtered_velocity = None

    def _bearing_unit(self, pursuer_yaw: float, centroid: PixelCoord) -> np.ndarray:
        """
        Unit direction (world frame) from the pursuer toward the pixel ``centroid``.

        Sign conventions are chosen to match the working forward-camera image servo:
          - +horizontal pixel offset (target to the right of centre) -> world "right"
          - +vertical pixel offset   (target below centre)           -> downward (-Z)
        """
        u, v = centroid
        heading = np.array([np.cos(pursuer_yaw), np.sin(pursuer_yaw), 0.0])
        right = np.array([np.sin(pursuer_yaw), -np.cos(pursuer_yaw), 0.0])
        up = np.array([0.0, 0.0, 1.0])

        x_offset = (float(u) - 0.5 * self.image_width) / self.fx
        y_offset = (float(v) - 0.5 * self.image_height) / self.fx

        direction = heading + x_offset * right - y_offset * up
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return heading
        return direction / norm

    def _smooth_position(self, measurement: np.ndarray) -> np.ndarray:
        if self._filtered_position is None:
            self._filtered_position = measurement.astype(float)
        else:
            alpha = self.position_smoothing
            self._filtered_position = (1.0 - alpha) * self._filtered_position + alpha * measurement
        return self._filtered_position

    def _smooth_velocity(self, measurement: np.ndarray) -> np.ndarray:
        if self._filtered_velocity is None:
            self._filtered_velocity = measurement.astype(float)
        else:
            alpha = self.velocity_smoothing
            self._filtered_velocity = (1.0 - alpha) * self._filtered_velocity + alpha * measurement
        return self._filtered_velocity

    def update(
        self,
        sim_time: float,
        pursuer_position: np.ndarray,
        pursuer_yaw: float,
        centroid: PixelCoord | None,
        bbox_width_px: float | None,
    ) -> TargetEstimate | None:
        """
        Fold a new detection into the estimate. Returns None (and resets the internal
        filters) when there is no usable detection this frame.
        """
        if (
            centroid is None
            or bbox_width_px is None
            or bbox_width_px < self.min_bbox_width_px
        ):
            self.reset()
            return None

        distance = float((self.fx * self.target_width_m) / bbox_width_px)
        bearing = self._bearing_unit(pursuer_yaw, centroid)
        raw_position = np.asarray(pursuer_position, dtype=float) + distance * bearing

        filtered_position = self._smooth_position(raw_position).copy()

        velocity_valid = False
        if self._last_time is not None and self._last_raw_position is not None:
            dt = sim_time - self._last_time
            if dt > 1e-4:
                raw_velocity = (raw_position - self._last_raw_position) / dt
                speed = float(np.linalg.norm(raw_velocity))
                if speed > self.max_velocity_mps > 0.0:
                    raw_velocity *= self.max_velocity_mps / speed
                self._smooth_velocity(raw_velocity)
                velocity_valid = True

        self._last_time = sim_time
        self._last_raw_position = raw_position

        velocity = (
            self._filtered_velocity.copy()
            if self._filtered_velocity is not None
            else np.zeros(3)
        )

        return TargetEstimate(
            position=filtered_position,
            velocity=velocity,
            distance_m=distance,
            bearing_unit=bearing,
            velocity_valid=velocity_valid,
        )
