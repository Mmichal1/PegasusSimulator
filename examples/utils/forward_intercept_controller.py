#!/usr/bin/env python
"""
| File: forward_intercept_controller.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Forward-facing vision-only interceptor with constant-velocity lead pursuit.
|
| This is the forward-camera sibling of upward_intercept_torpedo_controller_pluggable.
| Unlike pure pursuit (which flies straight at the target's current image position and
| overshoots crossing targets), this controller:
|   1. Reconstructs the target's world position/velocity from monocular detections
|      (bearing from the pixel offset + range from bounding-box width).
|   2. Solves a constant-velocity intercept to obtain a lead / aim point.
|   3. Steers its *translation* toward the lead point, while *yawing* to keep the
|      current target centred so it stays inside the (narrow) forward field of view.
|
| The detector must expose ``detect(rgb) -> VisionDetection`` (bounding box width is
| required for range), e.g. utils.yolo_detector_adapter.YoloDroneDetectorAdapter.
"""

from __future__ import annotations

from typing import Any, Protocol

import carb
import numpy as np

from torpedo_controller_pluggable import PluggableTorpedoVisionController
from world_intercept import (
    CAMERA_TO_BODY,
    IRIS_DRONE_WIDTH_M,
    TargetWorldEstimator,
    solve_intercept_iterative,
)

try:
    from camera_preview import import_cv2
except ImportError:  # pragma: no cover (Isaac Sim environment only)
    import_cv2 = None

RgbImage = np.ndarray
PixelCoord = tuple[int, int]


class RichTargetDetector(Protocol):
    def detect(self, rgb_image: RgbImage) -> Any: ...


def _draw_intercept_overlay(
    rgb: RgbImage,
    centroid: PixelCoord | None,
    distance_m: float | None,
    target_speed_mps: float | None,
    time_to_intercept_s: float | None,
    lead_enabled: bool,
    status: str,
    trajectory_points: list[PixelCoord] | None = None,
    aim_point_image: PixelCoord | None = None,
    coasting: bool = False,
) -> RgbImage:
    """
    Annotate a *copy* of the camera frame for the preview window only.

    NOTE: This is drawn on the preview image, never on the frame fed to the detector,
    so the overlay can never leak back into detection. The predicted trajectory and
    intercept point are the controller's world-frame estimates projected back into the
    image (see ForwardInterceptController._project_world_to_image).
    """
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

    # Predicted target trajectory (amber while coasting, green while detected). BGR colours.
    traj_color = (0, 165, 255) if coasting else (0, 255, 0)
    if trajectory_points and len(trajectory_points) >= 2:
        pts = np.array(trajectory_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(bgr, [pts], False, traj_color, 2, cv2.LINE_AA)

    # Intercept / aim point (magenta).
    if aim_point_image is not None:
        au, av = int(aim_point_image[0]), int(aim_point_image[1])
        cv2.drawMarker(bgr, (au, av), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
        cv2.circle(bgr, (au, av), 9, (255, 0, 255), 2)

    if centroid is not None:
        u, v = int(centroid[0]), int(centroid[1])
        cv2.circle(bgr, (u, v), 10, (0, 255, 0), 2)
        cv2.line(bgr, (cx, cy), (u, v), (255, 200, 0), 2)

    lines = [f"mode: {status}{' (lead)' if lead_enabled else ' (pure)'}"]
    if distance_m is not None:
        lines.append(f"range ~{distance_m:.1f} m")
    if target_speed_mps is not None:
        lines.append(f"tgt speed ~{target_speed_mps:.1f} m/s")
    if time_to_intercept_s is not None:
        lines.append(f"t-intercept ~{time_to_intercept_s:.2f} s")

    y = 26
    for text in lines:
        cv2.putText(bgr, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class ForwardInterceptController(PluggableTorpedoVisionController):
    """Vision-only forward interceptor that leads a constant-velocity target."""

    def __init__(
        self,
        detector: RichTargetDetector,
        pursuer_speed: float = 3.5,
        target_width_m: float = IRIS_DRONE_WIDTH_M,
        fov_deg: float = 70.0,
        position_smoothing: float = 0.35,
        velocity_smoothing: float = 0.2,
        acceleration_smoothing: float = 0.15,
        min_bbox_width_px: float = 6.0,
        max_intercept_time_s: float = 12.0,
        min_altitude: float = 0.6,
        max_altitude: float = 10.0,
        lead_enabled: bool = True,
        coast_time_s: float = 2.5,               # keep pursuing last trajectory this long after a detection drop
        # --- smoothness / camera-stability shaping ---
        max_yaw_rate: float = 1.2,               # rad/s cap on how fast the camera pans
        speed_ramp_time_s: float = 2.5,          # ramp 0 -> pursuer_speed after (re)acquiring
        velocity_time_constant_s: float = 0.5,   # low-pass on the commanded velocity vector
        alignment_min: float = 0.25,             # min speed fraction while still turning to face target
        # --- vertical channel (decoupled from the horizontal chase) ---
        climb_rate_gain: float = 1.0,            # P gain (1/s) mapping altitude error -> vertical speed
        max_climb_rate: float = 1.5,             # m/s cap on climb (kept low to avoid ballooning up)
        max_descent_rate: float = 2.5,           # m/s cap on descent (higher so it can shed altitude)
        **kwargs: Any,
    ):
        if not callable(getattr(detector, "detect", None)):
            raise ValueError("detector must expose detect(rgb_image) -> VisionDetection")

        # Drive the geometric controller at the pursuer speed and start from the spawn height.
        kwargs.setdefault("forward_speed", pursuer_speed)
        super().__init__(detector=detector, **kwargs)

        self._rich_detector: RichTargetDetector = detector
        self.pursuer_speed = float(pursuer_speed)
        self.max_intercept_time_s = float(max_intercept_time_s)
        self.min_altitude = float(min_altitude)
        self.max_altitude = float(max_altitude)
        self.lead_enabled = bool(lead_enabled)
        self.coast_time_s = float(coast_time_s)

        self.max_yaw_rate = float(max_yaw_rate)
        self.speed_ramp_time_s = float(speed_ramp_time_s)
        self.velocity_time_constant_s = float(velocity_time_constant_s)
        self.alignment_min = float(alignment_min)
        self.climb_rate_gain = float(climb_rate_gain)
        self.max_climb_rate = float(max_climb_rate)
        self.max_descent_rate = float(max_descent_rate)

        self._estimator = TargetWorldEstimator(
            image_width=self.image_width,
            image_height=self.image_height,
            fov_deg=fov_deg,
            target_width_m=target_width_m,
            position_smoothing=position_smoothing,
            velocity_smoothing=velocity_smoothing,
            acceleration_smoothing=acceleration_smoothing,
            min_bbox_width_px=min_bbox_width_px,
            max_acceleration_mps2=max_target_accel_mps2,
        )

        # Latest world-frame estimate / intercept solution (consumed by _vision_references).
        self._target_position: np.ndarray | None = None
        self._target_velocity: np.ndarray | None = None
        self._target_acceleration: np.ndarray | None = None
        self._distance_m: float | None = None
        self._aim_point: np.ndarray | None = None
        self._time_to_intercept: float | None = None

        # Command-shaping state.
        self._filtered_velocity_ref: np.ndarray | None = None
        self._pursuit_start_time: float | None = None

        # Last-known target memory (for coasting through detection dropouts).
        self._last_target_position: np.ndarray | None = None
        self._last_target_velocity: np.ndarray | None = None
        self._last_seen_time: float | None = None

    def reset_episode(self) -> None:
        super().reset_episode()
        self._estimator.reset()
        self._target_position = None
        self._target_velocity = None
        self._distance_m = None
        self._aim_point = None
        self._time_to_intercept = None
        self._target_acceleration = None
        self._filtered_velocity_ref = None
        self._pursuit_start_time = None
        self._last_target_position = None
        self._last_target_velocity = None
        self._last_seen_time = None

    def _pursuer_yaw(self) -> float:
        forward = self.R.as_matrix()[:, 0]
        return float(np.arctan2(forward[1], forward[0]))

    def _project_world_to_image(self, world_point: np.ndarray) -> PixelCoord | None:
        """
        Project a world point into the pursuer camera image using the full attitude.

        Exact inverse of TargetWorldEstimator's bearing model, so a projected target
        position lands back on its detection regardless of the drone's pitch/roll.
        Returns None if the point is behind the camera or projects far outside the frame.
        """
        if not self.reveived_first_state:
            return None

        fx = self._estimator.fx
        R = self.R.as_matrix()
        camera_position = self.p + R @ self._estimator.camera_offset_body

        rel = np.asarray(world_point, dtype=float) - camera_position
        # World -> body -> camera frame (CAMERA_TO_BODY is orthonormal, so inverse = transpose).
        dir_camera = CAMERA_TO_BODY.T @ (R.T @ rel)
        forward_comp = float(dir_camera[2])
        if forward_comp <= 1e-3:
            return None

        x_off = float(dir_camera[0]) / forward_comp
        y_off = float(dir_camera[1]) / forward_comp
        u = 0.5 * self.image_width + fx * x_off
        v = 0.5 * self.image_height + fx * y_off

        # Reject points that project absurdly far off-screen (keeps polylines sane).
        if not (-4 * self.image_width < u < 4 * self.image_width):
            return None
        if not (-4 * self.image_height < v < 4 * self.image_height):
            return None
        return (int(round(u)), int(round(v)))

    def _projected_trajectory_and_aim(self) -> tuple[list[PixelCoord], PixelCoord | None]:
        """Project the current predicted target trajectory + aim point into image space."""
        if not self._viz_active or self._viz_target_position is None:
            return [], None

        velocity = (
            self._viz_target_velocity
            if self._viz_target_velocity is not None
            else np.zeros(3)
        )
        acceleration = (
            self._viz_target_acceleration
            if self._viz_target_acceleration is not None
            else np.zeros(3)
        )
        horizon = max(3.0, float(self._viz_time_to_intercept) if self._viz_time_to_intercept else 0.0)

        trajectory: list[PixelCoord] = []
        for t in np.linspace(0.0, horizon, 16):
            te = min(t, self.accel_time_cap_s)
            future = self._viz_target_position + velocity * t + 0.5 * acceleration * te * te
            projected = self._project_world_to_image(future)
            if projected is not None:
                trajectory.append(projected)

        aim_image = (
            self._project_world_to_image(self._viz_aim_point)
            if self._viz_aim_point is not None
            else None
        )
        return trajectory, aim_image

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _tc_alpha(self, dt: float, time_constant_s: float) -> float:
        """First-order low-pass coefficient for the given time constant (dt-aware)."""
        if time_constant_s <= 1e-6:
            return 1.0
        return float(np.clip(dt / time_constant_s, 0.0, 1.0))

    def _step_yaw(self, dt: float, desired_yaw: float) -> tuple[float, float]:
        """Move _yaw_ref toward desired_yaw under a slew-rate limit; return (yaw_ref, yaw_rate_ref)."""
        yaw_error = self._wrap_to_pi(desired_yaw - self._yaw_ref)
        max_step = self.max_yaw_rate * dt
        yaw_step = float(np.clip(yaw_error, -max_step, max_step))
        self._yaw_ref = self._wrap_to_pi(self._yaw_ref + yaw_step)
        yaw_rate_ref = yaw_step / dt if dt > 1e-9 else 0.0
        return self._yaw_ref, yaw_rate_ref

    def _smooth_velocity_ref(self, dt: float, v_target: np.ndarray) -> np.ndarray:
        alpha = self._tc_alpha(dt, self.velocity_time_constant_s)
        if self._filtered_velocity_ref is None:
            self._filtered_velocity_ref = v_target.astype(float)
        else:
            self._filtered_velocity_ref = (
                (1.0 - alpha) * self._filtered_velocity_ref + alpha * v_target
            )
        return self._filtered_velocity_ref.copy()

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
            self._estimator.set_image_size(self.image_width, self.image_height)

            detection = self._rich_detector.detect(rgb)
            self._centroid = detection.centroid
            self._has_detection = detection.centroid is not None

            estimate = None
            if self.reveived_first_state and not self._in_startup_hold():
                estimate = self._estimator.update(
                    self._sim_time,
                    self.p,
                    self.R.as_matrix(),
                    detection.centroid,
                    detection.bbox_width_px,
                )

            if estimate is not None:
                self._target_position = estimate.position
                self._target_velocity = estimate.velocity
                self._target_acceleration = estimate.acceleration
                self._distance_m = estimate.distance_m

                aim_point, tti = (None, None)
                if self.lead_enabled and estimate.velocity_valid:
                    aim_point, tti = self._solve_intercept(
                        estimate.position, estimate.velocity, estimate.acceleration
                    )
                self._aim_point = aim_point
                self._time_to_intercept = tti

                # Remember the last good fix so we can coast through detection dropouts.
                self._last_target_position = estimate.position.copy()
                self._last_target_velocity = estimate.velocity.copy()
                self._last_seen_time = self._sim_time
            else:
                # No usable detection this frame: keep the last-known target/trajectory so
                # _vision_references can dead-reckon it (coast) instead of stopping.
                self._target_position = None
                self._target_velocity = None
                self._target_acceleration = None
                self._aim_point = None
                self._time_to_intercept = None

            if self.show_debug_feed:
                self._ensure_debug_preview()
                if self._camera_preview is not None:
                    debug_image = detection.debug_image
                    if debug_image is None:
                        debug_image = rgb
                    target_speed = (
                        float(np.linalg.norm(self._viz_target_velocity))
                        if self._viz_target_velocity is not None
                        else None
                    )
                    trajectory_points, aim_image = self._projected_trajectory_and_aim()
                    debug_image = _draw_intercept_overlay(
                        debug_image,
                        detection.centroid,
                        self._distance_m,
                        target_speed,
                        self._viz_time_to_intercept,
                        self.lead_enabled,
                        self._last_status,
                        trajectory_points=trajectory_points,
                        aim_point_image=aim_image,
                        coasting=self._viz_coasting,
                    )
                    self._camera_preview.update(debug_image)
        except Exception as exc:
            carb.log_warn(f"Forward intercept detector failed: {exc}")

    def _search_references(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Slowly rotate in place at cruise height to reacquire a lost target."""
        if self._yaw_ref is None:
            self._init_yaw_reference()

        # Losing the target arms a fresh speed ramp so re-acquisition is gentle again.
        self._pursuit_start_time = None

        yaw_rate = float(self.search_yaw_rate)
        self._yaw_ref = self._wrap_to_pi(self._yaw_ref + yaw_rate * dt)

        z_velocity = float(0.35 * (self.cruise_height - self.p[2]))
        v_ref = self._smooth_velocity_ref(dt, np.array([0.0, 0.0, z_velocity]))
        p_ref = np.array([self.p[0], self.p[1], self.p[2] + v_ref[2] * dt])
        p_ref[2] = float(np.clip(p_ref[2], self.min_altitude, self.max_altitude))
        self._last_status = "searching"

        return p_ref, v_ref, np.zeros(3), np.zeros(3), float(self._yaw_ref), yaw_rate

    def _speed_ramp(self) -> float:
        """Smoothstep 0->1 over speed_ramp_time_s since the target was (re)acquired."""
        if self._pursuit_start_time is None:
            self._pursuit_start_time = self._sim_time
        if self.speed_ramp_time_s <= 1e-6:
            return 1.0
        fraction = np.clip((self._sim_time - self._pursuit_start_time) / self.speed_ramp_time_s, 0.0, 1.0)
        return float(fraction * fraction * (3.0 - 2.0 * fraction))

    def _coast_target(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Dead-reckon the last-known target for up to coast_time_s after a detection drop.

        Returns (predicted_position, velocity) using the last good fix propagated along its
        last measured velocity, or None if there is no recent enough fix to coast on.
        """
        if (
            self._last_target_position is not None
            and self._last_seen_time is not None
            and (self._sim_time - self._last_seen_time) <= self.coast_time_s
        ):
            velocity = (
                self._last_target_velocity
                if self._last_target_velocity is not None
                else np.zeros(3)
            )
            elapsed = self._sim_time - self._last_seen_time
            return self._last_target_position + velocity * elapsed, velocity
        return None

    def _vision_references(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        if self._yaw_ref is None:
            self._init_yaw_reference()

        if self._has_detection and self._target_position is not None:
            # Fresh detection: use it and the lead point solved in update_graphical_sensor.
            target_position = self._target_position
            if self.lead_enabled and self._aim_point is not None:
                aim_point = self._aim_point
                self._last_status = "intercepting"
            else:
                aim_point = target_position
                self._last_status = "pursuing"
        else:
            # No fresh detection: coast on the remembered trajectory if it is recent enough,
            # recomputing the intercept from the dead-reckoned position. Only a real fix (in
            # update_graphical_sensor) ever refreshes the remembered target.
            coasted = self._coast_target()
            if coasted is None:
                return self._search_references(dt)

            target_position, target_velocity = coasted
            aim_point, tti = (None, None)
            if self.lead_enabled:
                aim_point, tti = solve_intercept(
                    self.p,
                    self.pursuer_speed,
                    target_position,
                    target_velocity,
                    max_time_s=self.max_intercept_time_s,
                )
            self._time_to_intercept = tti
            if aim_point is None:
                aim_point = target_position
            self._last_status = "coasting"

        to_aim = aim_point - self.p
        aim_distance = float(np.linalg.norm(to_aim))
        if aim_distance < 1e-6:
            direction = np.array([np.cos(self._yaw_ref), np.sin(self._yaw_ref), 0.0])
        else:
            direction = to_aim / aim_distance

        # Yaw to face the *current* (predicted) target so it stays centred in the forward
        # camera. Slew-rate limited so the camera pans smoothly.
        to_target = target_position - self.p
        desired_yaw = float(np.arctan2(to_target[1], to_target[0]))
        yaw_ref, yaw_rate_ref = self._step_yaw(dt, desired_yaw)

        # Ease into motion: ramp speed after (re)acquisition, and hold speed down while
        # still turning to face the target so it doesn't slew out of the narrow FOV.
        yaw_error = self._wrap_to_pi(desired_yaw - yaw_ref)
        alignment = float(np.clip(np.cos(yaw_error), self.alignment_min, 1.0))
        speed = self.pursuer_speed * self._speed_ramp() * alignment

        # Decouple the vertical channel from the horizontal chase. Spending the full speed
        # budget along the 3D line-of-sight makes the pursuer balloon upward toward a higher
        # target and then overshoot, because a multirotor climbs far more readily than it can
        # descend. Instead chase in the horizontal plane at `speed` (low-pass filtered for a
        # smooth camera), and close the altitude gap with a rate-limited proportional term
        # (prompt, no momentum) that allows a faster descent than climb.
        horizontal = np.array([direction[0], direction[1], 0.0])
        h_norm = float(np.linalg.norm(horizontal))
        v_xy = (horizontal / h_norm) * speed if h_norm > 1e-6 else np.zeros(3)
        v_ref = self._smooth_velocity_ref(dt, v_xy)

        alt_error = float(aim_point[2] - self.p[2])
        v_ref[2] = float(
            np.clip(self.climb_rate_gain * alt_error, -self.max_descent_rate, self.max_climb_rate)
        )

        p_ref = self.p + v_ref * dt
        p_ref[2] = float(np.clip(p_ref[2], self.min_altitude, self.max_altitude))

        # Publish what we are steering toward for the viewport visualization.
        self._viz_target_position = np.asarray(target_position, dtype=float)
        self._viz_target_velocity = np.asarray(target_velocity, dtype=float)
        self._viz_target_acceleration = np.asarray(target_acceleration, dtype=float)
        self._viz_aim_point = np.asarray(aim_point, dtype=float)
        self._viz_time_to_intercept = tti
        self._viz_coasting = coasting
        self._viz_active = True

        return p_ref, v_ref, np.zeros(3), np.zeros(3), yaw_ref, yaw_rate_ref
