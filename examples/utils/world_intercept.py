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

# Camera-to-body rotation (columns = camera axes expressed in the FLU body frame),
# matching the forward-camera mounting used by the torpedo examples:
#   camera +X (image right)   = body -Y
#   camera +Y (image down)    = body -Z
#   camera +Z (boresight/fwd) = body +X
# A pixel ray [x_off, y_off, 1] (camera frame) maps to body via CAMERA_TO_BODY, then to
# world via the pursuer body-to-world rotation. This properly accounts for the pursuer's
# pitch/roll (ego-motion), instead of assuming a level, yaw-only camera.
CAMERA_TO_BODY = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)

# Same convention after pitching the forward camera up 90 deg (boresight = body +Z):
#   camera +X (image right)   = body -Y
#   camera +Y (image down)    = body +X
#   camera +Z (boresight/up)  = body +Z
CAMERA_TO_BODY_UPWARD = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


@dataclass
class TargetEstimate:
    """World-frame estimate of the target produced from a single camera frame."""

    position: np.ndarray            # filtered target position in world frame [m]
    velocity: np.ndarray            # filtered target velocity in world frame [m/s]
    acceleration: np.ndarray        # filtered target acceleration in world frame [m/s^2]
    distance_m: float               # straight-line range from pursuer to target [m]
    bearing_unit: np.ndarray        # unit vector (world) from pursuer toward target
    velocity_valid: bool            # True once enough samples exist for a velocity estimate
    acceleration_valid: bool        # True once enough samples exist for an acceleration estimate


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


def _predict_position(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    t: float,
    accel_time_cap_s: float,
) -> np.ndarray:
    """Constant-acceleration prediction, with the acceleration term limited to a short
    horizon so noisy/large accelerations can't blow up long-range predictions."""
    te = min(max(t, 0.0), accel_time_cap_s)
    return position + velocity * t + 0.5 * acceleration * te * te


def solve_intercept_iterative(
    pursuer_position: np.ndarray,
    pursuer_speed: float,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    target_acceleration: np.ndarray | None = None,
    max_time_s: float = 12.0,
    iterations: int = 4,
    accel_time_cap_s: float = 2.5,
) -> tuple[np.ndarray | None, float | None]:
    """
    Iterative (fixed-point) intercept solver supporting a constant-acceleration target.

    Unlike the closed-form constant-velocity ``solve_intercept``, this predicts the target
    with ``p + v t + 1/2 a t^2`` (acceleration term horizon-limited), which lets the pursuer
    anticipate a decelerating/reversing target instead of overshooting the turnaround.

    Starting from ``t = range / speed`` it repeatedly (a) predicts the future target
    position at ``t`` and (b) recomputes ``t = |future - pursuer| / speed`` until it
    converges. Returns (aim_point, time_to_intercept) or (None, None) if infeasible.
    """
    P = np.asarray(pursuer_position, dtype=float)
    pos = np.asarray(target_position, dtype=float)
    vel = np.asarray(target_velocity, dtype=float)
    acc = np.zeros(3) if target_acceleration is None else np.asarray(target_acceleration, dtype=float)
    speed = float(pursuer_speed)
    if speed <= 1e-6:
        return None, None

    t = float(np.linalg.norm(pos - P) / speed)
    for _ in range(max(1, iterations)):
        future = _predict_position(pos, vel, acc, t, accel_time_cap_s)
        t_new = float(np.linalg.norm(future - P) / speed)
        if not np.isfinite(t_new):
            return None, None
        t_new = float(np.clip(t_new, 0.0, max_time_s))
        converged = abs(t_new - t) < 1e-3
        t = t_new
        if converged:
            break

    aim_point = _predict_position(pos, vel, acc, t, accel_time_cap_s)
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
        acceleration_smoothing: float = 0.15,
        min_bbox_width_px: float = 6.0,
        max_velocity_mps: float = 12.0,
        max_acceleration_mps2: float = 15.0,
        camera_offset_body: tuple[float, float, float] = (0.15, 0.0, 0.0),
        camera_to_body: np.ndarray | None = None,
    ):
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.fov_deg = float(fov_deg)
        self.target_width_m = float(target_width_m)
        self.camera_offset_body = np.asarray(camera_offset_body, dtype=float)
        self.camera_to_body = (
            np.asarray(CAMERA_TO_BODY, dtype=float)
            if camera_to_body is None
            else np.asarray(camera_to_body, dtype=float)
        )
        self.position_smoothing = float(position_smoothing)
        self.velocity_smoothing = float(velocity_smoothing)
        self.acceleration_smoothing = float(acceleration_smoothing)
        self.min_bbox_width_px = float(min_bbox_width_px)
        self.max_velocity_mps = float(max_velocity_mps)
        self.max_acceleration_mps2 = float(max_acceleration_mps2)

        self.fx = 0.5 * self.image_width / np.tan(np.radians(self.fov_deg * 0.5))

        self._last_time: float | None = None
        self._last_raw_position: np.ndarray | None = None
        self._filtered_position: np.ndarray | None = None
        self._filtered_velocity: np.ndarray | None = None
        self._prev_filtered_velocity: np.ndarray | None = None
        self._filtered_acceleration: np.ndarray | None = None

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
        self._prev_filtered_velocity = None
        self._filtered_acceleration = None

    def _bearing_unit(self, pursuer_rotation: np.ndarray, centroid: PixelCoord) -> np.ndarray:
        """
        Unit direction (world frame) from the camera toward the pixel ``centroid``,
        using the *full* pursuer attitude so pitch/roll (ego-motion) are accounted for.

        The pixel ray in the camera frame is [x_off, y_off, 1] (x right, y down, z fwd);
        it is rotated to the body frame via CAMERA_TO_BODY and then to the world frame via
        the pursuer body-to-world rotation.
        """
        u, v = centroid
        x_offset = (float(u) - 0.5 * self.image_width) / self.fx
        y_offset = (float(v) - 0.5 * self.image_height) / self.fx

        camera_ray = np.array([x_offset, y_offset, 1.0])
        direction = pursuer_rotation @ (self.camera_to_body @ camera_ray)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return pursuer_rotation @ self.camera_to_body[:, 2]
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

    def _smooth_acceleration(self, measurement: np.ndarray) -> np.ndarray:
        if self._filtered_acceleration is None:
            self._filtered_acceleration = measurement.astype(float)
        else:
            alpha = self.acceleration_smoothing
            self._filtered_acceleration = (
                (1.0 - alpha) * self._filtered_acceleration + alpha * measurement
            )
        return self._filtered_acceleration

    def update(
        self,
        sim_time: float,
        pursuer_position: np.ndarray,
        pursuer_rotation: np.ndarray,
        centroid: PixelCoord | None,
        bbox_width_px: float | None,
    ) -> TargetEstimate | None:
        """
        Fold a new detection into the estimate. Returns None (and resets the internal
        filters) when there is no usable detection this frame.

        ``pursuer_rotation`` is the 3x3 body-to-world rotation matrix (e.g. from the
        vehicle attitude), used to account for the camera's true orientation.
        """
        if (
            centroid is None
            or bbox_width_px is None
            or bbox_width_px < self.min_bbox_width_px
        ):
            self.reset()
            return None

        pursuer_rotation = np.asarray(pursuer_rotation, dtype=float)
        distance = float((self.fx * self.target_width_m) / bbox_width_px)
        bearing = self._bearing_unit(pursuer_rotation, centroid)
        # World position of the camera (body origin + mounted offset rotated to world).
        camera_position = np.asarray(pursuer_position, dtype=float) + pursuer_rotation @ self.camera_offset_body
        raw_position = camera_position + distance * bearing

        filtered_position = self._smooth_position(raw_position).copy()

        velocity_valid = False
        acceleration_valid = False
        if self._last_time is not None and self._last_raw_position is not None:
            dt = sim_time - self._last_time
            if dt > 1e-4:
                raw_velocity = (raw_position - self._last_raw_position) / dt
                speed = float(np.linalg.norm(raw_velocity))
                if speed > self.max_velocity_mps > 0.0:
                    raw_velocity *= self.max_velocity_mps / speed
                self._smooth_velocity(raw_velocity)
                velocity_valid = True

                # Acceleration from the change in the (filtered) velocity estimate.
                if self._prev_filtered_velocity is not None:
                    raw_acceleration = (self._filtered_velocity - self._prev_filtered_velocity) / dt
                    accel_mag = float(np.linalg.norm(raw_acceleration))
                    if accel_mag > self.max_acceleration_mps2 > 0.0:
                        raw_acceleration *= self.max_acceleration_mps2 / accel_mag
                    self._smooth_acceleration(raw_acceleration)
                    acceleration_valid = True
                self._prev_filtered_velocity = self._filtered_velocity.copy()

        self._last_time = sim_time
        self._last_raw_position = raw_position

        velocity = (
            self._filtered_velocity.copy()
            if self._filtered_velocity is not None
            else np.zeros(3)
        )
        acceleration = (
            self._filtered_acceleration.copy()
            if self._filtered_acceleration is not None
            else np.zeros(3)
        )

        return TargetEstimate(
            position=filtered_position,
            velocity=velocity,
            acceleration=acceleration,
            distance_m=distance,
            bearing_unit=bearing,
            velocity_valid=velocity_valid,
            acceleration_valid=acceleration_valid,
        )
