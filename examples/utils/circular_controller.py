#!/usr/bin/env python
"""
| File: circular_controller.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Nonlinear controller that tracks a horizontal circular trajectory indefinitely.
"""

import carb
import numpy as np

from nonlinear_controller import NonlinearController


class CircularController(NonlinearController):
    """Tracks a constant-altitude circle using the nonlinear controller from NonlinearController."""

    def __init__(
        self,
        radius=3.0,
        height=1.5,
        center=(0.0, 0.0),
        angular_velocity=0.4,
        results_file=None,
        startup_delay=0.0,
        show_camera_feed=True,
        camera_window_name="Drone camera",
        **kwargs,
    ):
        self.radius = radius
        self.height = height
        self.center = np.array(center, dtype=float)
        self.angular_velocity = angular_velocity
        self.startup_delay = float(startup_delay)
        self._sim_time = 0.0
        self._hover_position = None
        self._hold_completed = False
        self._show_camera_feed = show_camera_feed
        self._camera_window_name = camera_window_name
        self._camera_preview = None
        self._camera_preview_error_logged = False
        self._skip_inprocess_opencv = False

        super().__init__(trajectory_file=None, results_file=results_file, **kwargs)

    def _gamma(self, t: float) -> float:
        return self.angular_velocity * t

    def pd(self, t, s=0.6, reverse=False):
        gamma = self._gamma(t)
        return np.array(
            [
                self.center[0] + self.radius * np.cos(gamma),
                self.center[1] + self.radius * np.sin(gamma),
                self.height,
            ]
        )

    def d_pd(self, t, s=0.6, reverse=False):
        gamma = self._gamma(t)
        omega = self.angular_velocity
        return np.array(
            [
                -self.radius * omega * np.sin(gamma),
                self.radius * omega * np.cos(gamma),
                0.0,
            ]
        )

    def dd_pd(self, t, s=0.6, reverse=False):
        gamma = self._gamma(t)
        omega = self.angular_velocity
        return np.array(
            [
                -self.radius * omega**2 * np.cos(gamma),
                -self.radius * omega**2 * np.sin(gamma),
                0.0,
            ]
        )

    def ddd_pd(self, t, s=0.6, reverse=False):
        gamma = self._gamma(t)
        omega = self.angular_velocity
        return np.array(
            [
                self.radius * omega**3 * np.sin(gamma),
                -self.radius * omega**3 * np.cos(gamma),
                0.0,
            ]
        )

    def yaw_d(self, t, s=0.6):
        return self._gamma(t) + np.pi / 2.0

    def d_yaw_d(self, t, s=0.6):
        return self.angular_velocity

    def reset_statistics(self):
        super().reset_statistics()
        self._sim_time = 0.0
        self._hover_position = None
        self._hold_completed = False
        self.attitude_rate_error_over_time = []
        self.int = np.array([0.0, 0.0, 0.0])

    def reset_episode(self) -> None:
        super().reset_episode()

    def _in_startup_hold(self) -> bool:
        return (not self._hold_completed) and (self._sim_time < self.startup_delay)

    def _hover_references(self):
        if self._hover_position is None:
            self._hover_position = self.p.copy()
        p_ref = self._hover_position.copy()
        v_ref = np.zeros(3)
        a_ref = np.zeros(3)
        j_ref = np.zeros(3)
        yaw_ref = 0.0
        yaw_rate_ref = 0.0
        return p_ref, v_ref, a_ref, j_ref, yaw_ref, yaw_rate_ref

    def update(self, dt: float):
        # Override to support startup hover hold before beginning circular motion.
        if not self.reveived_first_state:
            return

        self._sim_time += dt
        if self._in_startup_hold():
            p_ref, v_ref, a_ref, j_ref, yaw_ref, yaw_rate_ref = self._hover_references()

            ep = self.p - p_ref
            ev = self.v - v_ref
            self.int = self.int + (ep * dt)
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
            Z_b_cross_X_c = np.cross(Z_b_des, X_c_des)
            Y_b_des = Z_b_cross_X_c / (np.linalg.norm(Z_b_cross_X_c) + 1e-9)
            X_b_des = np.cross(Y_b_des, Z_b_des)

            R_des = np.c_[X_b_des, Y_b_des, Z_b_des]
            R = self.R.as_matrix()
            e_R = 0.5 * self.vee((R_des.T @ R) - (R.T @ R_des))

            self.a = (u_1 * Z_B) / self.m - np.array([0.0, 0.0, self.g])
            hw = (self.m / u_1) * (j_ref - np.dot(Z_b_des, j_ref) * Z_b_des)
            w_des = np.array(
                [-np.dot(hw, Y_b_des), np.dot(hw, X_b_des), yaw_rate_ref * Z_b_des[2]]
            )

            e_w = self.w - w_des
            tau = -(self.Kr @ e_R) - (self.Kw @ e_w)

            if self.vehicle:
                self.input_ref = self.vehicle.force_and_torques_to_velocities(u_1, tau)
            return

        # First frame after hold: reset trajectory time so the circle starts cleanly.
        if not self._hold_completed:
            self._hold_completed = True
            self.total_time = 0.0

        super().update(dt)

    @staticmethod
    def _rgb_from_rgba(rgba):
        if rgba is None or len(getattr(rgba, "shape", ())) < 3:
            return None

        rgb = np.asarray(rgba[..., :3])
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

        return rgb

    def _ensure_camera_preview(self, width, height):
        from camera_preview import OpenCvCameraPreview, create_camera_preview

        if self._camera_preview is not None:
            if isinstance(self._camera_preview, OpenCvCameraPreview) and self._camera_preview.init_error:
                carb.log_warn(
                    "In-process OpenCV window failed; switching to subprocess or fallback preview."
                )
                self._skip_inprocess_opencv = True
                self._close_camera_window()
            else:
                return True

        try:
            skip_inprocess = getattr(self, "_skip_inprocess_opencv", False)
            self._camera_preview, backend = create_camera_preview(
                self._camera_window_name,
                width=width,
                height=height,
                prefer_opencv=True,
                skip_inprocess_opencv=skip_inprocess,
            )
            self._skip_inprocess_opencv = backend != "opencv"

            if backend.startswith("opencv"):
                carb.log_info(f"Camera preview backend: {backend}")
            elif backend == "omni.ui" and not self._camera_preview_error_logged:
                carb.log_warn(
                    "Using Omniverse UI for camera preview. For a real OpenCV window, install "
                    "python3-opencv on the host: sudo apt install python3-opencv"
                )
            return True
        except Exception as exc:
            if not self._camera_preview_error_logged:
                carb.log_error(f"Camera preview failed to start:\n{exc}")
                self._camera_preview_error_logged = True
            self._show_camera_feed = False
            return False

    def update_graphical_sensor(self, sensor_type, data):
        """Show onboard camera RGB feed in a separate OpenCV window."""
        if not self._show_camera_feed or sensor_type != "MonocularCamera" or not data:
            return

        camera = data.get("camera")
        if camera is None:
            return

        rgba = camera.get_rgba()
        rgb = self._rgb_from_rgba(rgba)
        if rgb is None:
            return

        h, w = rgb.shape[:2]
        if not self._ensure_camera_preview(w, h):
            return

        self._camera_preview.update(rgb)

    def stop(self):
        super().stop()
        self._close_camera_window()

    def _close_camera_window(self):
        if self._camera_preview is not None:
            self._camera_preview.close()
            self._camera_preview = None
