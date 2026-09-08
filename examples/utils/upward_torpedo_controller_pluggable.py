#!/usr/bin/env python
"""
| File: upward_torpedo_controller_pluggable.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Vision-guided upward pursuit with an upward-pointing camera.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from torpedo_controller_pluggable import PluggableTorpedoVisionController


class UpwardTorpedoVisionController(PluggableTorpedoVisionController):
    """
    Pursue a target above the drone using an upward-facing camera.

    The upward camera axis is treated as the vehicle "front": velocity commands
    point the drone top/thrust vector toward the target instead of flying nose-first.
    """

    def __init__(
        self,
        climb_speed: float = 2.0,
        k_lateral: float = 0.35,
        k_climb: float = 0.25,
        angular_gain: float | None = None,
        max_horizontal_speed: float = 1.0,
        command_velocity_smoothing: float = 0.12,
        error_deadband: float = 0.05,
        error_soft_limit: float = 0.45,
        centroid_smoothing: float = 0.18,
        patrol_xy_gain: float = 0.25,
        patrol_z_gain: float = 0.35,
        max_descent_speed: float = 0.6,
        max_patrol_speed: float = 0.8,
        patrol_altitude_priority_margin: float = 1.5,
        max_patrol_tilt_deg: float = 12.0,
        max_patrol_descent_accel: float = 0.8,
        max_altitude: float = 25.0,
        cruise_height: float = 1.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("forward_speed", climb_speed)
        kwargs.setdefault("cruise_height", cruise_height)
        kwargs.setdefault("search_yaw_rate", 0.0)
        super().__init__(**kwargs)

        self.climb_speed = float(climb_speed)
        self.k_lateral = float(angular_gain if angular_gain is not None else k_lateral)
        self.k_climb = float(k_climb)
        self.max_horizontal_speed = float(max_horizontal_speed)
        self.command_velocity_smoothing = float(command_velocity_smoothing)
        self.error_deadband = float(error_deadband)
        self.error_soft_limit = float(error_soft_limit)
        self.centroid_smoothing = float(centroid_smoothing)
        self.patrol_xy_gain = float(patrol_xy_gain)
        self.patrol_z_gain = float(patrol_z_gain)
        self.max_descent_speed = float(max_descent_speed)
        self.max_patrol_speed = float(max_patrol_speed)
        self.patrol_altitude_priority_margin = float(patrol_altitude_priority_margin)
        self.max_patrol_tilt_deg = float(max_patrol_tilt_deg)
        self.max_patrol_descent_accel = float(max_patrol_descent_accel)
        self.max_altitude = float(max_altitude)

        self._home_position: np.ndarray | None = None
        self._filtered_centroid: np.ndarray | None = None
        self._filtered_velocity_ref: np.ndarray | None = None
        self._filtered_patrol_velocity_ref: np.ndarray | None = None

    def _ensure_home_position(self) -> None:
        if self._home_position is None:
            self._home_position = np.array([self.p[0], self.p[1], self.cruise_height], dtype=float)

    def _reset_tracking_filters(self) -> None:
        self._filtered_centroid = None
        self._filtered_velocity_ref = None
        self._filtered_patrol_velocity_ref = None

    def reset_episode(self) -> None:
        super().reset_episode()
        self._sim_time = 0.0
        self._home_position = None
        self._yaw_ref = None
        self._hover_position = None
        self._reset_tracking_filters()

    def _camera_axes_in_world(self) -> tuple[np.ndarray, np.ndarray]:
        """Image horizontal/vertical axes for the upward camera, expressed in world frame."""
        cam_u = np.array([-np.sin(self._yaw_ref), np.cos(self._yaw_ref), 0.0])
        cam_v = np.array([np.cos(self._yaw_ref), np.sin(self._yaw_ref), 0.0])
        return cam_u, cam_v

    def _soften_error(self, error: float) -> float:
        magnitude = abs(error)
        if magnitude < self.error_deadband:
            return 0.0
        limited = float(np.tanh(magnitude / self.error_soft_limit) * self.error_soft_limit)
        return float(np.sign(error) * limited)

    def _bearing_direction(self, ex: float, ey: float) -> np.ndarray:
        """
        Unit direction toward the target using the upward camera frame.

        World +Z is the camera boresight (drone top), and image offsets tilt that axis.
        """
        cam_u, cam_v = self._camera_axes_in_world()
        ex_soft = self._soften_error(ex) * self.k_lateral
        ey_soft = self._soften_error(ey) * self.k_lateral
        bearing = cam_u * ex_soft + cam_v * ey_soft + np.array([0.0, 0.0, 1.0])
        norm = float(np.linalg.norm(bearing))
        if norm < 1e-9:
            return np.array([0.0, 0.0, 1.0])
        return bearing / norm

    def _smooth_velocity_reference(self, velocity: np.ndarray) -> np.ndarray:
        if self._filtered_velocity_ref is None:
            self._filtered_velocity_ref = velocity.copy()
        else:
            alpha = self.command_velocity_smoothing
            self._filtered_velocity_ref = (1.0 - alpha) * self._filtered_velocity_ref + alpha * velocity
        return self._filtered_velocity_ref.copy()

    def _bearing_velocity_from_image_errors(
        self,
        ex: float,
        ey: float,
        speed_scale: float = 1.0,
    ) -> np.ndarray:
        bearing = self._bearing_direction(ex, ey)
        speed = self.climb_speed * self.speed_boost_factor * speed_scale
        v_ref = bearing * speed

        horizontal_norm = float(np.linalg.norm(v_ref[:2]))
        if horizontal_norm > self.max_horizontal_speed > 0.0:
            v_ref[:2] *= self.max_horizontal_speed / horizontal_norm

        return self._smooth_velocity_reference(v_ref)

    def _filtered_pixel_errors(self) -> tuple[float, float] | None:
        if not self._has_detection or self._centroid is None:
            self._reset_tracking_filters()
            return None

        u, v = self._centroid
        measurement = np.array([float(u), float(v)], dtype=float)
        if self._filtered_centroid is None:
            self._filtered_centroid = measurement
        else:
            alpha = self.centroid_smoothing
            self._filtered_centroid = (1.0 - alpha) * self._filtered_centroid + alpha * measurement

        u_f, v_f = self._filtered_centroid
        ex = (u_f - 0.5 * self.image_width) / (0.5 * self.image_width)
        ey = (v_f - 0.5 * self.image_height) / (0.5 * self.image_height)
        return float(ex), float(ey)

    def _smooth_patrol_velocity_reference(self, velocity: np.ndarray) -> np.ndarray:
        if self._filtered_patrol_velocity_ref is None:
            self._filtered_patrol_velocity_ref = velocity.copy()
        else:
            alpha = max(self.command_velocity_smoothing, 0.08)
            self._filtered_patrol_velocity_ref = (
                (1.0 - alpha) * self._filtered_patrol_velocity_ref + alpha * velocity
            )
        return self._filtered_patrol_velocity_ref.copy()

    def _patrol_references(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        entering_patrol = self._last_status != "patrolling"
        if entering_patrol:
            self.int = np.zeros(3)
            self._filtered_patrol_velocity_ref = None

        self._ensure_home_position()
        if self._yaw_ref is None:
            self._init_yaw_reference()

        home = self._home_position
        assert home is not None

        xy_error = home[:2] - self.p[:2]
        altitude_error = home[2] - self.p[2]
        above_home = self.p[2] > home[2] + self.patrol_altitude_priority_margin

        if above_home:
            # Descend first while staying mostly level above the home point.
            v_ref = np.array(
                [
                    0.15 * self.patrol_xy_gain * xy_error[0],
                    0.15 * self.patrol_xy_gain * xy_error[1],
                    float(np.clip(altitude_error * self.patrol_z_gain, -self.max_descent_speed, 0.2)),
                ],
                dtype=float,
            )
        else:
            v_ref = np.array(
                [
                    self.patrol_xy_gain * xy_error[0],
                    self.patrol_xy_gain * xy_error[1],
                    float(np.clip(altitude_error * self.patrol_z_gain, -self.max_descent_speed, 0.5)),
                ],
                dtype=float,
            )

        speed = float(np.linalg.norm(v_ref))
        if speed > self.max_patrol_speed > 0.0:
            v_ref *= self.max_patrol_speed / speed

        v_ref = self._smooth_patrol_velocity_reference(v_ref)
        p_ref = home.copy()
        p_ref[0] = float(self.p[0] + v_ref[0] * dt)
        p_ref[1] = float(self.p[1] + v_ref[1] * dt)
        z_step = float(
            np.clip(
                altitude_error * 0.08 * max(dt, 1e-3),
                -self.max_descent_speed * dt,
                self.max_descent_speed * dt,
            )
        )
        p_ref[2] = float(np.clip(self.p[2] + z_step, self.cruise_height, self.max_altitude))
        self._last_status = "patrolling"

        return p_ref, v_ref, np.zeros(3), np.zeros(3), float(self._yaw_ref), 0.0

    def _apply_level_patrol_control(
        self,
        dt: float,
        p_ref: np.ndarray,
        v_ref: np.ndarray,
        yaw_ref: float,
    ) -> None:
        ep = self.p - p_ref
        ev = self.v - v_ref
        self.int[:2] = self.int[:2] + ep[:2] * dt
        self.int[2] = 0.0
        ei = self.int

        a_des = -(self.Kp @ ep) - (self.Kd @ ev) - (self.Ki @ ei)
        a_des[2] = float(np.clip(a_des[2], -self.max_patrol_descent_accel, 1.2))

        max_horiz_accel = 1.0
        horiz_norm = float(np.linalg.norm(a_des[:2]))
        if horiz_norm > max_horiz_accel > 0.0:
            a_des[:2] *= max_horiz_accel / horiz_norm

        F_des = self.m * (a_des + np.array([0.0, 0.0, self.g]))

        max_tilt = np.radians(self.max_patrol_tilt_deg)
        horiz_force = F_des[:2]
        hf_norm = float(np.linalg.norm(horiz_force))
        max_horiz_force = self.m * self.g * np.tan(max_tilt)
        if hf_norm > max_horiz_force > 0.0:
            horiz_force = horiz_force * (max_horiz_force / hf_norm)

        Z_b_des = np.array([horiz_force[0], horiz_force[1], self.m * self.g], dtype=float)
        Z_b_des = Z_b_des / (np.linalg.norm(Z_b_des) + 1e-9)

        u_1 = float(F_des @ Z_b_des)
        u_1 = float(np.clip(u_1, 0.25 * self.m * self.g, 1.35 * self.m * self.g))

        X_c_des = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
        Y_b_des = np.cross(Z_b_des, X_c_des)
        Y_b_des = Y_b_des / (np.linalg.norm(Y_b_des) + 1e-9)
        X_b_des = np.cross(Y_b_des, Z_b_des)

        R_des = np.c_[X_b_des, Y_b_des, Z_b_des]
        R = self.R.as_matrix()
        e_R = 0.5 * self.vee((R_des.T @ R) - (R.T @ R_des))

        Z_B = R[:, 2]
        self.a = (u_1 * Z_B) / self.m - np.array([0.0, 0.0, self.g])
        w_des = np.zeros(3)
        e_w = self.w - w_des
        tau = -(self.Kr @ e_R) - (self.Kw @ e_w)

        if self.vehicle:
            self.input_ref = self.vehicle.force_and_torques_to_velocities(u_1, tau)

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
        if self._last_status == "patrolling":
            self._apply_level_patrol_control(dt, p_ref, v_ref, yaw_ref)
            return
        super()._apply_control(dt, p_ref, v_ref, a_ref, j_ref, yaw_ref, yaw_rate_ref)

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

    def _vision_references(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        if self._yaw_ref is None:
            self._init_yaw_reference()

        errors = self._filtered_pixel_errors()
        if errors is None:
            return self._patrol_references(dt)

        ex, ey = errors
        v_ref = self._bearing_velocity_from_image_errors(ex, ey)
        v_ref[2] += float(self.k_climb * self._soften_error(-ey))

        p_ref = self.p + v_ref * dt
        p_ref[2] = float(np.clip(p_ref[2], self.cruise_height, self.max_altitude))
        self._last_status = "climbing"

        return p_ref, v_ref, np.zeros(3), np.zeros(3), float(self._yaw_ref), 0.0
