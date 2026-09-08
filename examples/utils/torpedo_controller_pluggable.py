#!/usr/bin/env python
"""
| Provide a callable:
|
|     detector(rgb_image: np.ndarray) -> (centroid, debug_image)
|
| where:
| - rgb_image: HxWx3 uint8 (RGB). (If you output float images in [0,1], convert inside detector.)
| - centroid: (u, v) pixel tuple, or None if no target detected.
| - debug_image: optional HxWx3 uint8 RGB image to show in the preview (can be same as input),
|   or None to let the controller render a simple overlay itself.
"""

from __future__ import annotations

from typing import Protocol,  Any

import carb
import numpy as np

from nonlinear_controller import NonlinearController

try:
    from camera_preview import create_camera_preview, import_cv2
except ImportError:  # pragma: no cover (Isaac Sim environment only)
    create_camera_preview = None
    import_cv2 = None


RgbImage = np.ndarray
PixelCoord = tuple[int, int]


class TargetDetector(Protocol):
    def __call__(self, rgb_image: RgbImage) -> tuple[PixelCoord | None, RgbImage | None]: ...


def _draw_basic_overlay(rgb: RgbImage, centroid: PixelCoord | None) -> RgbImage:
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

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class PluggableTorpedoVisionController(NonlinearController):
    def __init__(
        self,
        detector: TargetDetector,
        forward_speed: float = 2.0,
        speed_boost_factor: float = 2.0,
        k_yaw: float = 1.4,
        k_alt: float = 1.0,
        cruise_height: float = 1.5,
        search_yaw_rate: float = 0.2,
        startup_delay: float = 5.0,
        image_size: tuple[int, int] = (640, 480),
        show_debug_feed: bool = True,
        **kwargs: Any,
    ):
        # Do not load a CSV trajectory – this controller is fully vision-driven.
        super().__init__(trajectory_file=None, **kwargs)

        if detector is None or not callable(detector):
            raise ValueError("detector must be a callable(rgb_image) -> (centroid, debug_image)")

        self.detector: TargetDetector = detector

        self.forward_speed = float(forward_speed)
        self.speed_boost_factor = float(speed_boost_factor)
        self.k_yaw = float(k_yaw)
        self.k_alt = float(k_alt)
        self.cruise_height = float(cruise_height)
        self.search_yaw_rate = float(search_yaw_rate)
        self.startup_delay = float(startup_delay)

        self.image_width, self.image_height = int(image_size[0]), int(image_size[1])
        self.show_debug_feed = bool(show_debug_feed)

        self._sim_time: float = 0.0
        self._yaw_ref: float | None = None
        self._hover_position: np.ndarray | None = None

        self._centroid: PixelCoord | None = None
        self._has_detection: bool = False
        self._last_status: str = "standby"

        self._camera_preview: Any = None
        self._camera_backend: str | None = None
        self._skip_inprocess_opencv: bool = False

    def _in_startup_hold(self) -> bool:
        return self._sim_time < self.startup_delay

    def _ensure_debug_preview(self) -> None:
        if not self.show_debug_feed or create_camera_preview is None:
            return
        if self._camera_preview is not None:
            return
        try:
            self._camera_preview, self._camera_backend = create_camera_preview(
                "Drone camera",
                width=self.image_width,
                height=self.image_height,
                prefer_opencv=True,
                skip_inprocess_opencv=self._skip_inprocess_opencv,
            )
            carb.log_info(f"Pluggable torpedo debug camera backend: {self._camera_backend}")
        except Exception as exc:
            carb.log_warn(f"Debug camera preview unavailable: {exc}")

    def _close_debug_preview(self) -> None:
        if self._camera_preview is not None:
            self._camera_preview.close()
            self._camera_preview = None

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

            centroid, debug_image = self.detector(rgb)
            self._centroid = centroid
            self._has_detection = centroid is not None

            if self.show_debug_feed:
                self._ensure_debug_preview()
                if self._camera_preview is not None:
                    if debug_image is None:
                        debug_image = _draw_basic_overlay(rgb, centroid)
                    self._camera_preview.update(debug_image)
        except Exception as exc:
            carb.log_warn(f"Detector failed: {exc}")

    def _init_yaw_reference(self) -> None:
        forward = self.R.as_matrix()[:, 0]
        self._yaw_ref = float(np.arctan2(forward[1], forward[0]))

    def _hover_references(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        if self._hover_position is None:
            self._hover_position = self.p.copy()
        p_ref = self._hover_position.copy()
        v_ref = np.zeros(3)
        if self._yaw_ref is None:
            self._init_yaw_reference()
        # _yaw_ref is guaranteed set above
        return p_ref, v_ref, np.zeros(3), np.zeros(3), float(self._yaw_ref), 0.0

    def _vision_references(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        if self._yaw_ref is None:
            self._init_yaw_reference()

        detected = self._has_detection and self._centroid is not None
        if detected:
            u, v = self._centroid
            ex = (u - 0.5 * self.image_width) / (0.5 * self.image_width)
            ey = (v - 0.5 * self.image_height) / (0.5 * self.image_height)
            yaw_rate = float(-self.k_yaw * ex)
            z_velocity = float(-self.k_alt * ey)
            self._last_status = "pursuing"
        else:
            yaw_rate = float(self.search_yaw_rate)
            z_velocity = float(0.35 * (self.cruise_height - self.p[2]))
            self._last_status = "searching"

        self._yaw_ref = float(self._yaw_ref + yaw_rate * dt)
        yaw_ref = float(self._yaw_ref)
        yaw_rate_ref = yaw_rate

        heading = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
        speed = self.forward_speed * (self.speed_boost_factor if detected else 1.0)
        v_ref = heading * speed + np.array([0.0, 0.0, z_velocity])
        p_ref = self.p + v_ref * dt
        p_ref[2] = np.clip(p_ref[2], 0.5, 8.0)

        return p_ref, v_ref, np.zeros(3), np.zeros(3), yaw_ref, yaw_rate_ref

    def _apply_control(
        self,
        dt: float,
        p_ref: np.ndarray,
        v_ref: np.ndarray,
        a_ref: np.ndarray,
        j_ref: np.ndarray,
        yaw_ref: float,
        yaw_rate_ref: float,
    ) -> None:
        ep = self.p - p_ref
        ev = self.v - v_ref
        self.int = self.int + ep * dt
        ei = self.int

        F_des = (
            -(self.Kp @ ep)
            - (self.Kd @ ev)
            - (self.Ki @ ei)
            + np.array([0.0, 0.0, self.m * self.g])
            + (self.m * a_ref)
        )

        Z_B = self.R.as_matrix()[:, 2]
        u_1 = F_des @ Z_B
        u_1 = max(u_1, 0.05 * self.m * self.g)

        Z_b_des = F_des / np.linalg.norm(F_des)
        X_c_des = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
        Y_b_des = np.cross(Z_b_des, X_c_des)
        Y_b_des = Y_b_des / (np.linalg.norm(Y_b_des) + 1e-9)
        X_b_des = np.cross(Y_b_des, Z_b_des)

        R_des = np.c_[X_b_des, Y_b_des, Z_b_des]
        R = self.R.as_matrix()
        e_R = 0.5 * self.vee((R_des.T @ R) - (R.T @ R_des))

        self.a = (u_1 * Z_B) / self.m - np.array([0.0, 0.0, self.g])
        hw = (self.m / u_1) * (j_ref - np.dot(Z_b_des, j_ref) * Z_b_des)
        w_des = np.array(
            [
                -np.dot(hw, Y_b_des),
                np.dot(hw, X_b_des),
                yaw_rate_ref * Z_b_des[2],
            ]
        )
        e_w = self.w - w_des
        tau = -(self.Kr @ e_R) - (self.Kw @ e_w)

        if self.vehicle:
            self.input_ref = self.vehicle.force_and_torques_to_velocities(u_1, tau)

    def update(self, dt: float) -> None:
        if not self.reveived_first_state:
            return

        self._sim_time += dt
        self.total_time += dt

        if self._in_startup_hold():
            refs = self._hover_references()
            self._apply_control(dt, *refs)
            return

        refs = self._vision_references(dt)
        self._apply_control(dt, *refs)

    def stop(self) -> None:
        self._close_debug_preview()
        super().stop()

