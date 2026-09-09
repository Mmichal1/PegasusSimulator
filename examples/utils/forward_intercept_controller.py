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
from world_intercept import IRIS_DRONE_WIDTH_M, TargetWorldEstimator, solve_intercept

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

        self._estimator = TargetWorldEstimator(
            image_width=self.image_width,
            image_height=self.image_height,
            fov_deg=fov_deg,
            target_width_m=target_width_m,
            position_smoothing=position_smoothing,
            velocity_smoothing=velocity_smoothing,
            min_bbox_width_px=min_bbox_width_px,
        )

        # Latest world-frame estimate / intercept solution (consumed by _vision_references).
        self._target_position: np.ndarray | None = None
        self._target_velocity: np.ndarray | None = None
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
        self._filtered_velocity_ref = None
        self._pursuit_start_time = None
        self._last_target_position = None
        self._last_target_velocity = None
        self._last_seen_time = None

    def _pursuer_yaw(self) -> float:
        forward = self.R.as_matrix()[:, 0]
        return float(np.arctan2(forward[1], forward[0]))

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
                    self._pursuer_yaw(),
                    detection.centroid,
                    detection.bbox_width_px,
                )

            if estimate is not None:
                self._target_position = estimate.position
                self._target_velocity = estimate.velocity
                self._distance_m = estimate.distance_m

                aim_point, tti = (None, None)
                if self.lead_enabled and estimate.velocity_valid:
                    aim_point, tti = solve_intercept(
                        self.p,
                        self.pursuer_speed,
                        estimate.position,
                        estimate.velocity,
                        max_time_s=self.max_intercept_time_s,
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
                self._distance_m = None
                self._aim_point = None
                self._time_to_intercept = None

            if self.show_debug_feed:
                self._ensure_debug_preview()
                if self._camera_preview is not None:
                    debug_image = detection.debug_image
                    if debug_image is None:
                        debug_image = rgb
                    target_speed = (
                        float(np.linalg.norm(self._target_velocity))
                        if self._target_velocity is not None
                        else None
                    )
                    debug_image = _draw_intercept_overlay(
                        debug_image,
                        detection.centroid,
                        self._distance_m,
                        target_speed,
                        self._time_to_intercept,
                        self.lead_enabled,
                        self._last_status,
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

        # Yaw to face the *current* target (not the lead point) so it stays centred in
        # the forward camera. Slew-rate limited so the camera pans smoothly.
        to_target = target_position - self.p
        desired_yaw = float(np.arctan2(to_target[1], to_target[0]))
        yaw_ref, yaw_rate_ref = self._step_yaw(dt, desired_yaw)

        # Ease into motion: ramp speed after (re)acquisition, and hold speed down while
        # still turning to face the target so it doesn't slew out of the narrow FOV.
        yaw_error = self._wrap_to_pi(desired_yaw - yaw_ref)
        alignment = float(np.clip(np.cos(yaw_error), self.alignment_min, 1.0))
        speed = self.pursuer_speed * self._speed_ramp() * alignment

        v_target = direction * speed
        v_ref = self._smooth_velocity_ref(dt, v_target)

        p_ref = self.p + v_ref * dt
        p_ref[2] = float(np.clip(p_ref[2], self.min_altitude, self.max_altitude))

        return p_ref, v_ref, np.zeros(3), np.zeros(3), yaw_ref, yaw_rate_ref
