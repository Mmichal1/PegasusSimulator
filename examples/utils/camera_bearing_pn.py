#!/usr/bin/env python
"""Camera-bearing proportional-navigation primitives.

This module is deliberately independent of Isaac Sim.  Guidance consumes only a
world-frame camera bearing plus the interceptor's own velocity.  Simulator target
state is not part of the camera-PN command path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray | None:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm < 1e-9 or not np.all(np.isfinite(value)):
        return None
    return value / norm


def _low_pass_alpha(dt: float, time_constant_s: float) -> float:
    if time_constant_s <= 1e-9:
        return 1.0
    return float(np.clip(dt / (time_constant_s + dt), 0.0, 1.0))


@dataclass(frozen=True)
class BearingPNCommand:
    bearing: np.ndarray
    los_rate: np.ndarray
    acceleration: np.ndarray
    closing_speed: float
    rate_valid: bool
    coasting: bool


class CameraBearingPN:
    """Filter camera bearings and produce true-PN lateral acceleration."""

    def __init__(
        self,
        nav_constant: float = 4.5,
        bearing_time_constant_s: float = 0.08,
        los_rate_time_constant_s: float = 0.15,
        max_lateral_accel_mps2: float = 6.0,
        closing_speed_floor_fraction: float = 0.35,
        jump_reject_cos: float = 0.5,
        coast_time_s: float = 0.35,
        max_sample_dt_s: float = 0.25,
    ):
        self.nav_constant = float(nav_constant)
        self.bearing_time_constant_s = float(bearing_time_constant_s)
        self.los_rate_time_constant_s = float(los_rate_time_constant_s)
        self.max_lateral_accel_mps2 = float(max_lateral_accel_mps2)
        self.closing_speed_floor_fraction = float(closing_speed_floor_fraction)
        self.jump_reject_cos = float(jump_reject_cos)
        self.coast_time_s = float(coast_time_s)
        self.max_sample_dt_s = float(max_sample_dt_s)
        self.reset()

    def reset(self) -> None:
        self.bearing: np.ndarray | None = None
        self.los_rate = np.zeros(3)
        self.last_sample_time: float | None = None
        self.rate_valid = False
        self.rejected_samples = 0

    def update(self, timestamp_s: float, measured_bearing: np.ndarray) -> bool:
        """Ingest one camera bearing. Returns False when the sample is rejected."""
        measured = _unit(measured_bearing)
        if measured is None:
            return False

        now = float(timestamp_s)
        if self.bearing is None or self.last_sample_time is None:
            self.bearing = measured
            self.last_sample_time = now
            self.rate_valid = False
            return True

        if float(np.dot(self.bearing, measured)) < self.jump_reject_cos:
            self.rejected_samples += 1
            return False

        dt = now - self.last_sample_time
        if dt <= 1e-5:
            return False

        previous = self.bearing.copy()
        bearing_alpha = _low_pass_alpha(dt, self.bearing_time_constant_s)
        filtered = _unit((1.0 - bearing_alpha) * previous + bearing_alpha * measured)
        if filtered is None:
            return False
        self.bearing = filtered
        self.last_sample_time = now

        if dt > self.max_sample_dt_s:
            self.rate_valid = False
            self.los_rate[:] = 0.0
            return True

        bearing_dot = (filtered - previous) / dt
        # Differentiation and numerical normalization can introduce a radial
        # component. LOS rotation must be tangent to the unit sphere.
        bearing_dot -= float(np.dot(bearing_dot, filtered)) * filtered
        measured_los_rate = np.cross(filtered, bearing_dot)
        rate_alpha = _low_pass_alpha(dt, self.los_rate_time_constant_s)
        self.los_rate = (
            (1.0 - rate_alpha) * self.los_rate
            + rate_alpha * measured_los_rate
        )
        self.rate_valid = True
        return True

    def is_fresh(self, timestamp_s: float) -> bool:
        return (
            self.bearing is not None
            and self.last_sample_time is not None
            and float(timestamp_s) - self.last_sample_time <= self.coast_time_s
        )

    def command(
        self,
        timestamp_s: float,
        pursuer_velocity: np.ndarray,
        pursuer_speed_mps: float,
    ) -> BearingPNCommand | None:
        if not self.is_fresh(timestamp_s) or self.bearing is None:
            return None

        u_los = self.bearing.copy()
        pursuer_velocity = np.asarray(pursuer_velocity, dtype=float)
        measured_closing = float(np.dot(pursuer_velocity, u_los))
        closing_floor = self.closing_speed_floor_fraction * float(pursuer_speed_mps)
        closing_speed = max(measured_closing, closing_floor, 0.0)

        acceleration = np.zeros(3)
        if self.rate_valid:
            acceleration = (
                self.nav_constant
                * closing_speed
                * np.cross(self.los_rate, u_los)
            )
            magnitude = float(np.linalg.norm(acceleration))
            if magnitude > self.max_lateral_accel_mps2 > 0.0:
                acceleration *= self.max_lateral_accel_mps2 / magnitude

        coasting = (
            self.last_sample_time is not None
            and float(timestamp_s) > self.last_sample_time + 1e-5
        )
        return BearingPNCommand(
            bearing=u_los,
            los_rate=self.los_rate.copy(),
            acceleration=acceleration,
            closing_speed=closing_speed,
            rate_valid=self.rate_valid,
            coasting=coasting,
        )


def true_pn_acceleration(
    los: np.ndarray,
    relative_velocity: np.ndarray,
    nav_constant: float,
    max_lateral_accel_mps2: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Ground-truth PN for diagnostic oracle mode only."""
    los = np.asarray(los, dtype=float)
    relative_velocity = np.asarray(relative_velocity, dtype=float)
    distance = float(np.linalg.norm(los))
    if distance < 1e-9:
        return np.zeros(3), 0.0, np.zeros(3)
    u_los = los / distance
    los_rate = np.cross(los, relative_velocity) / (distance * distance)
    closing_speed = max(-float(np.dot(los, relative_velocity)) / distance, 0.0)
    acceleration = (
        float(nav_constant)
        * closing_speed
        * np.cross(los_rate, u_los)
    )
    magnitude = float(np.linalg.norm(acceleration))
    if magnitude > max_lateral_accel_mps2 > 0.0:
        acceleration *= max_lateral_accel_mps2 / magnitude
    return acceleration, closing_speed, los_rate


def step_flight_direction(
    current_direction: np.ndarray | None,
    los_bearing: np.ndarray,
    lateral_acceleration: np.ndarray,
    dt: float,
    speed_mps: float,
    max_turn_rate_rad_s: float,
    min_los_direction_cos: float,
    los_anchor_time_constant_s: float = 0.0,
) -> np.ndarray:
    """Integrate PN acceleration into a unit flight direction with safety limits."""
    u_los = _unit(los_bearing)
    if u_los is None:
        raise ValueError("los_bearing must be non-zero and finite")
    direction = _unit(current_direction) if current_direction is not None else u_los.copy()
    if direction is None:
        direction = u_los.copy()

    turn_rate = np.asarray(lateral_acceleration, dtype=float) / max(
        float(speed_mps), 0.25
    )
    magnitude = float(np.linalg.norm(turn_rate))
    if magnitude > max_turn_rate_rad_s > 0.0:
        turn_rate *= max_turn_rate_rad_s / magnitude
    stepped = _unit(direction + turn_rate * max(float(dt), 0.0))
    direction = u_los.copy() if stepped is None else stepped

    if los_anchor_time_constant_s > 0.0:
        anchor_alpha = _low_pass_alpha(
            max(float(dt), 0.0), float(los_anchor_time_constant_s)
        )
        anchored = _unit(
            (1.0 - anchor_alpha) * direction + anchor_alpha * u_los
        )
        if anchored is not None:
            direction = anchored

    minimum_cos = float(np.clip(min_los_direction_cos, 0.0, 1.0))
    los_cos = float(np.dot(direction, u_los))
    if los_cos < minimum_cos:
        lateral = direction - los_cos * u_los
        lateral_unit = _unit(lateral)
        if lateral_unit is None:
            return u_los.copy()
        direction = (
            minimum_cos * u_los
            + np.sqrt(max(0.0, 1.0 - minimum_cos**2)) * lateral_unit
        )
    return direction
