#!/usr/bin/env python
"""
| File: pn_intercept_controller.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Forward-facing vision-only interceptor guided by Proportional Navigation (PN).
|
| Where ForwardInterceptController flies to a solved lead point, this controller uses the
| classic PN guidance law: it commands a lateral acceleration proportional to the
| line-of-sight (LOS) rotation rate and closing speed, driving the LOS rate to zero (a
| constant-bearing, decreasing-range collision course):
|
|     R      = target_pos - pursuer_pos                 (LOS vector)
|     V_rel  = target_vel - pursuer_vel                 (relative velocity)
|     Omega  = (R x V_rel) / (R . R)                     (LOS rotation-rate vector)
|     V_c    = -(R . V_rel) / |R|                        (closing speed, +ve when closing)
|     a_cmd  = N * V_c * (Omega x R_hat)                 (True PN lateral acceleration)
|            + (N/2) * a_target_perp                     (Augmented PN term, optional)
|
| The commanded acceleration steers the pursuer's velocity direction; yaw still tracks the
| current target so the forward camera keeps it in view. Reuses the full-attitude vision
| estimator, coasting, command shaping and camera overlay from ForwardInterceptController.
"""

from __future__ import annotations

from typing import Any

import carb
import numpy as np

from forward_intercept_controller import ForwardInterceptController, RichTargetDetector


class PNInterceptController(ForwardInterceptController):
    """Vision-only forward interceptor guided by (augmented) proportional navigation."""

    def __init__(
        self,
        detector: RichTargetDetector,
        nav_constant: float = 4.0,               # PN gain N (typically 3-5)
        use_augmented_pn: bool = True,           # add the target-acceleration term (APN)
        max_lateral_accel_mps2: float = 20.0,    # clamp on the PN acceleration command
        max_turn_rate_rad_s: float = 1.5,        # direction slew limit, especially during startup
        steering_speed_floor_mps: float = 2.0,   # prevents a/speed exploding near zero speed
        min_los_direction_cos: float = 0.25,     # never command into the target's rear hemisphere
        telemetry_interval_s: float = 0.0,       # 0 disables periodic PN diagnostics
        **kwargs: Any,
    ):
        # lead_enabled has no meaning for PN (it doesn't use the closed-form lead point),
        # but the parent's update_graphical_sensor tolerates it; force it off to skip the
        # unused intercept solve.
        kwargs.setdefault("lead_enabled", False)
        super().__init__(detector=detector, **kwargs)

        self.nav_constant = float(nav_constant)
        self.use_augmented_pn = bool(use_augmented_pn)
        self.max_lateral_accel_mps2 = float(max_lateral_accel_mps2)
        self.max_turn_rate_rad_s = float(max_turn_rate_rad_s)
        self.steering_speed_floor_mps = float(steering_speed_floor_mps)
        self.min_los_direction_cos = float(
            np.clip(min_los_direction_cos, 0.0, 1.0)
        )
        self.telemetry_interval_s = float(telemetry_interval_s)

        # Commanded flight-direction unit vector, steered by the PN acceleration.
        self._pn_direction: np.ndarray | None = None
        self._last_telemetry_time = -np.inf

    def reset_episode(self) -> None:
        super().reset_episode()
        self._pn_direction = None
        self._last_telemetry_time = -np.inf

    def _pn_acceleration(
        self,
        los: np.ndarray,
        rng: float,
        relative_velocity: np.ndarray,
        target_acceleration: np.ndarray | None,
    ) -> tuple[np.ndarray, float]:
        """Return (commanded lateral acceleration, closing_speed)."""
        u_los = los / rng
        omega = np.cross(los, relative_velocity) / (rng * rng)
        closing_speed = -float(np.dot(los, relative_velocity)) / rng

        # True PN assumes a closing engagement. A negative/noisy closing estimate would
        # reverse the steering sign and can turn the pursuer away from the target.
        effective_closing_speed = max(closing_speed, 0.0)
        a_cmd = self.nav_constant * effective_closing_speed * np.cross(omega, u_los)

        # Augmented PN: feed-forward the component of target acceleration across the LOS.
        if (
            self.use_augmented_pn
            and target_acceleration is not None
            and effective_closing_speed > 0.0
        ):
            a_perp = target_acceleration - np.dot(target_acceleration, u_los) * u_los
            a_cmd = a_cmd + 0.5 * self.nav_constant * a_perp

        magnitude = float(np.linalg.norm(a_cmd))
        if magnitude > self.max_lateral_accel_mps2 > 0.0:
            a_cmd *= self.max_lateral_accel_mps2 / magnitude

        return a_cmd, closing_speed

    def _vision_references(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        if self._yaw_ref is None:
            self._init_yaw_reference()

        target_position, target_velocity, target_acceleration, coasting = self._resolve_target()
        if target_position is None:
            self._viz_active = False
            self._viz_coasting = False
            self._pn_direction = None
            return self._search_references(dt)

        los = target_position - self.p
        rng = float(np.linalg.norm(los))
        if rng < 1e-6:
            u_los = np.array([np.cos(self._yaw_ref), np.sin(self._yaw_ref), 0.0])
            a_cmd = np.zeros(3)
            closing_speed = 0.0
        else:
            u_los = los / rng
            relative_velocity = target_velocity - self.v
            a_cmd, closing_speed = self._pn_acceleration(
                los, rng, relative_velocity, target_acceleration
            )

        # Yaw to keep the current target centred in the forward camera (rate-limited).
        desired_yaw = float(np.arctan2(los[1], los[0]))
        yaw_ref, yaw_rate_ref = self._step_yaw(dt, desired_yaw)
        yaw_error = self._wrap_to_pi(desired_yaw - yaw_ref)
        alignment = float(np.clip(np.cos(yaw_error), self.alignment_min, 1.0))
        speed = self.pursuer_speed * self._speed_ramp() * alignment

        # Steer the commanded velocity direction by the PN acceleration. The speed floor
        # prevents a/speed from exploding while the startup speed ramp is near zero.
        if self._pn_direction is None:
            self._pn_direction = u_los.copy()
        turn_rate = a_cmd / max(speed, self.steering_speed_floor_mps)
        turn_rate_norm = float(np.linalg.norm(turn_rate))
        if turn_rate_norm > self.max_turn_rate_rad_s > 0.0:
            turn_rate *= self.max_turn_rate_rad_s / turn_rate_norm
        direction = self._pn_direction + turn_rate * dt
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-9 else u_los

        # PN is a lateral guidance law, not permission to fly away from the target. Vision
        # velocity/range transients can otherwise accumulate until the persistent direction
        # crosses 90 degrees. Limit it to a forward cone around the current LOS.
        los_cos = float(np.dot(direction, u_los))
        if los_cos < self.min_los_direction_cos:
            lateral = direction - los_cos * u_los
            lateral_norm = float(np.linalg.norm(lateral))
            if lateral_norm > 1e-9:
                lateral /= lateral_norm
                lateral_scale = np.sqrt(
                    max(0.0, 1.0 - self.min_los_direction_cos**2)
                )
                direction = (
                    self.min_los_direction_cos * u_los
                    + lateral_scale * lateral
                )
            else:
                direction = u_los.copy()

        self._pn_direction = direction
        direction = self._pn_direction

        v_ref = self._smooth_velocity_ref(dt, direction * speed)
        p_ref = self.p + v_ref * dt
        p_ref[2] = float(np.clip(p_ref[2], self.min_altitude, self.max_altitude))

        if (
            self.telemetry_interval_s > 0.0
            and self._sim_time - self._last_telemetry_time >= self.telemetry_interval_s
        ):
            self._last_telemetry_time = self._sim_time
            direction_los_cos = float(np.dot(direction, u_los))
            target_radial_speed = float(np.dot(target_velocity, u_los))
            actual_radial_speed = float(np.dot(self.v, u_los))
            carb.log_warn(
                "[PN_TELEMETRY] "
                f"t={self._sim_time:.2f} range={rng:.2f} closing={closing_speed:.2f} "
                f"p={np.round(self.p, 2).tolist()} tgt={np.round(target_position, 2).tolist()} "
                f"target_radial={target_radial_speed:.2f} pursuer_radial={actual_radial_speed:.2f} "
                f"dir_los_cos={direction_los_cos:.3f} speed_cmd={speed:.2f} "
                f"a_cmd={np.linalg.norm(a_cmd):.2f} coasting={coasting}"
            )

        self._last_status = "coasting" if coasting else "PN-tracking"

        # Visualization: predicted target path + implied intercept point at time-to-go.
        t_go = rng / closing_speed if closing_speed > 0.1 else None
        aim = target_position + target_velocity * t_go if t_go is not None else target_position
        self._viz_target_position = np.asarray(target_position, dtype=float)
        self._viz_target_velocity = np.asarray(target_velocity, dtype=float)
        self._viz_target_acceleration = np.asarray(
            target_acceleration if target_acceleration is not None else np.zeros(3), dtype=float
        )
        self._viz_aim_point = np.asarray(aim, dtype=float)
        self._viz_time_to_intercept = t_go
        self._viz_coasting = coasting
        self._viz_active = True

        return p_ref, v_ref, np.zeros(3), np.zeros(3), yaw_ref, yaw_rate_ref
