#!/usr/bin/env python
"""
| File: torpedo_controller.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Endless vision-guided pursuit of red targets (pure image servo, no waypoints).
"""

import carb
import numpy as np

from nonlinear_controller import NonlinearController
from red_ball_tracker import detect_red_balls, draw_detection, select_blob_for_pursuit

try:
    from camera_preview import create_camera_preview
except ImportError:
    create_camera_preview = None


class TorpedoVisionController(NonlinearController):
    """
    Endless torpedo-style pursuit: steer toward whichever red ball is best centred
    in the camera. No waypoint list or target counter — suited for demo flights.
    """

    def __init__(
        self,
        forward_speed=2.0,
        k_yaw=1.4,
        k_alt=1.0,
        cruise_height=1.5,
        search_yaw_rate=0.2,
        startup_delay=5.0,
        image_size=(640, 480),
        show_debug_feed=True,
        **kwargs,
    ):
        super().__init__(trajectory_file=None, **kwargs)

        self.forward_speed = forward_speed
        self.k_yaw = k_yaw
        self.k_alt = k_alt
        self.cruise_height = cruise_height
        self.search_yaw_rate = search_yaw_rate
        self.startup_delay = float(startup_delay)
        self.image_width, self.image_height = image_size
        self.show_debug_feed = show_debug_feed

        self.ball_detected = False
        self.ball_centroid = None
        self._yaw_ref = None
        self._hover_position = None
        self._sim_time = 0.0
        self._camera_preview = None
        self._camera_backend = None
        self._skip_inprocess_opencv = False
        self._last_status = "standby"

    def _in_startup_hold(self):
        return self._sim_time < self.startup_delay

    def _ensure_debug_preview(self):
        if not self.show_debug_feed or create_camera_preview is None:
            return
        if self._camera_preview is not None:
            return
        try:
            self._camera_preview, self._camera_backend = create_camera_preview(
                "TORPEDA",
                width=self.image_width,
                height=self.image_height,
                prefer_opencv=True,
                skip_inprocess_opencv=self._skip_inprocess_opencv,
            )
            carb.log_info(f"Torpedo debug camera backend: {self._camera_backend}")
        except Exception as exc:
            carb.log_warn(f"Debug camera preview unavailable: {exc}")

    def _close_debug_preview(self):
        if self._camera_preview is not None:
            self._camera_preview.close()
            self._camera_preview = None

    def update_graphical_sensor(self, sensor_type, data):
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

            multi = detect_red_balls(rgb)
            detection = {
                "detected": False,
                "centroid": None,
                "blobs": multi["blobs"],
                "mask": multi["mask"],
                "status": self._last_status,
                "startup_hold": self._in_startup_hold(),
            }

            if self.reveived_first_state and not self._in_startup_hold():
                blob = select_blob_for_pursuit(
                    multi["blobs"], self.image_width, self.image_height
                )
                if blob is not None:
                    detection["detected"] = True
                    detection["centroid"] = blob["centroid"]
                    self.ball_detected = True
                    self.ball_centroid = blob["centroid"]
                else:
                    self.ball_detected = False
                    self.ball_centroid = None
            else:
                self.ball_detected = False
                self.ball_centroid = None

            if self.show_debug_feed:
                self._ensure_debug_preview()
                if self._camera_preview is not None:
                    self._camera_preview.update(draw_detection(rgb, detection))
        except Exception as exc:
            carb.log_warn(f"Vision update failed: {exc}")

    def _init_yaw_reference(self):
        forward = self.R.as_matrix()[:, 0]
        self._yaw_ref = float(np.arctan2(forward[1], forward[0]))

    def _hover_references(self):
        if self._hover_position is None:
            self._hover_position = self.p.copy()
        p_ref = self._hover_position.copy()
        v_ref = np.zeros(3)
        if self._yaw_ref is None:
            self._init_yaw_reference()
        return p_ref, v_ref, np.zeros(3), np.zeros(3), self._yaw_ref, 0.0

    def _vision_references(self, dt):
        if self._yaw_ref is None:
            self._init_yaw_reference()

        yaw_rate = 0.0
        z_velocity = 0.0

        if self.ball_detected and self.ball_centroid is not None:
            u, v = self.ball_centroid
            ex = (u - 0.5 * self.image_width) / (0.5 * self.image_width)
            ey = (v - 0.5 * self.image_height) / (0.5 * self.image_height)
            yaw_rate = float(-self.k_yaw * ex)
            z_velocity = float(-self.k_alt * ey)
            self._last_status = "pursuing"
        else:
            yaw_rate = self.search_yaw_rate
            z_velocity = float(0.35 * (self.cruise_height - self.p[2]))
            self._last_status = "searching"

        self._yaw_ref += yaw_rate * dt
        yaw_ref = self._yaw_ref
        yaw_rate_ref = yaw_rate

        heading = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
        v_ref = heading * self.forward_speed + np.array([0.0, 0.0, z_velocity])
        p_ref = self.p + v_ref * dt
        p_ref[2] = np.clip(p_ref[2], 0.5, 8.0)

        return p_ref, v_ref, np.zeros(3), np.zeros(3), yaw_ref, yaw_rate_ref

    def _apply_control(self, dt, p_ref, v_ref, a_ref, j_ref, yaw_ref, yaw_rate_ref):
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

    def update(self, dt):
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

    def stop(self):
        self._close_debug_preview()
        super().stop()
