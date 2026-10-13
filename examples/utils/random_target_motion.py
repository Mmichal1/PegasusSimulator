#!/usr/bin/env python
"""
| File: random_target_motion.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Sample a slow straight-line or circular target trajectory for detector data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nonlinear_controller import NonlinearController


@dataclass
class TargetMotion:
    """One randomized target trajectory, starting at ``spawn`` after ``startup_delay``."""

    kind: str  # "stationary" | "straight" | "circle"
    spawn: np.ndarray
    speed: float
    startup_delay: float
    # straight
    direction: np.ndarray | None = None
    # circle
    center_xy: np.ndarray | None = None
    radius: float | None = None
    omega: float | None = None
    phase0: float | None = None

    def describe(self) -> str:
        spawn = np.round(self.spawn, 2).tolist()
        if self.kind == "stationary":
            return f"stationary spawn={spawn}"
        if self.kind == "straight":
            heading = float(np.degrees(np.arctan2(self.direction[1], self.direction[0])))
            return (
                f"straight spawn={spawn} speed={self.speed:.2f} m/s "
                f"heading={heading:.0f} deg"
            )
        diameter = 2.0 * float(self.radius)
        return (
            f"circle spawn={spawn} speed={self.speed:.2f} m/s "
            f"diameter={diameter:.1f} m omega={self.omega:.3f} rad/s"
        )


def sample_target_motion(
    nominal_spawn: np.ndarray,
    spawn_jitter_m: tuple[float, float, float] = (1.2, 1.2, 0.6),
    speed_mps: tuple[float, float] = (0.35, 0.55),
    circle_diameter_m: tuple[float, float] = (3.0, 8.0),
    startup_delay: float = 1.0,
    seed: int | None = None,
    kind: str | None = None,
    heading_deg: float | None = None,
) -> TargetMotion:
    """
    Draw a spawn near ``nominal_spawn`` and either a constant-velocity straight
    path or a horizontal circle. ``kind`` can force a deterministic scenario;
    ``None`` or ``"random"`` preserves randomized selection.
    """
    rng = np.random.default_rng(seed)
    requested_kind = "random" if kind is None else str(kind).strip().lower()
    valid_kinds = {"random", "stationary", "straight", "circle"}
    if requested_kind not in valid_kinds:
        raise ValueError(
            f"kind must be one of {sorted(valid_kinds)}, got {kind!r}"
        )
    jitter = np.asarray(spawn_jitter_m, dtype=float)
    spawn = np.asarray(nominal_spawn, dtype=float) + rng.uniform(-jitter, jitter)
    speed = float(rng.uniform(*speed_mps))

    if requested_kind == "stationary":
        return TargetMotion(
            kind="stationary",
            spawn=spawn,
            speed=0.0,
            startup_delay=float(startup_delay),
        )

    use_straight = (
        requested_kind == "straight"
        or (requested_kind == "random" and rng.random() < 0.5)
    )
    if use_straight:
        yaw = (
            float(rng.uniform(0.0, 2.0 * np.pi))
            if heading_deg is None
            else float(np.radians(heading_deg))
        )
        direction = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        return TargetMotion(
            kind="straight",
            spawn=spawn,
            speed=speed,
            startup_delay=float(startup_delay),
            direction=direction,
        )

    radius = 0.5 * float(rng.uniform(*circle_diameter_m))
    phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
    sign = 1.0 if rng.random() < 0.5 else -1.0
    omega = sign * speed / max(radius, 0.3)
    center_xy = spawn[:2] - radius * np.array([np.cos(phase0), np.sin(phase0)])
    return TargetMotion(
        kind="circle",
        spawn=spawn,
        speed=speed,
        startup_delay=float(startup_delay),
        center_xy=center_xy,
        radius=radius,
        omega=omega,
        phase0=phase0,
    )


class RandomTargetController(NonlinearController):
    """Track a ``TargetMotion``: hover at spawn, then straight or circle at low speed."""

    def __init__(self, motion: TargetMotion, **kwargs):
        super().__init__(**kwargs)
        self.motion = motion
        # NonlinearController starts built-in trajectories at t=-5 s; we want t=0.
        self.total_time = 0.0

    def reset_statistics(self):
        super().reset_statistics()
        self.total_time = 0.0

    def _tau(self, t: float) -> float:
        return max(0.0, float(t) - self.motion.startup_delay)

    def pd(self, t, s=0.6, reverse=False):
        tau = self._tau(t)
        m = self.motion
        if m.kind == "stationary":
            return m.spawn.copy()
        if m.kind == "straight":
            return m.spawn + m.direction * (m.speed * tau)
        phase = m.phase0 + m.omega * tau
        return np.array(
            [
                m.center_xy[0] + m.radius * np.cos(phase),
                m.center_xy[1] + m.radius * np.sin(phase),
                m.spawn[2],
            ]
        )

    def d_pd(self, t, s=0.6, reverse=False):
        if self._tau(t) <= 0.0:
            return np.zeros(3)
        m = self.motion
        if m.kind == "stationary":
            return np.zeros(3)
        if m.kind == "straight":
            return m.direction * m.speed
        phase = m.phase0 + m.omega * self._tau(t)
        return np.array(
            [
                -m.radius * m.omega * np.sin(phase),
                m.radius * m.omega * np.cos(phase),
                0.0,
            ]
        )

    def dd_pd(self, t, s=0.6, reverse=False):
        if self._tau(t) <= 0.0 or self.motion.kind in ("stationary", "straight"):
            return np.zeros(3)
        m = self.motion
        phase = m.phase0 + m.omega * self._tau(t)
        return np.array(
            [
                -m.radius * m.omega**2 * np.cos(phase),
                -m.radius * m.omega**2 * np.sin(phase),
                0.0,
            ]
        )

    def ddd_pd(self, t, s=0.6, reverse=False):
        if self._tau(t) <= 0.0 or self.motion.kind in ("stationary", "straight"):
            return np.zeros(3)
        m = self.motion
        phase = m.phase0 + m.omega * self._tau(t)
        return np.array(
            [
                m.radius * m.omega**3 * np.sin(phase),
                -m.radius * m.omega**3 * np.cos(phase),
                0.0,
            ]
        )

    def yaw_d(self, t, s=0.6):
        m = self.motion
        if m.kind == "stationary":
            return 0.0
        if m.kind == "straight":
            return float(np.arctan2(m.direction[1], m.direction[0]))
        return float(m.phase0 + m.omega * self._tau(t) + np.pi / 2.0)

    def d_yaw_d(self, t, s=0.6):
        if self._tau(t) <= 0.0:
            return 0.0
        if self.motion.kind in ("stationary", "straight"):
            return 0.0
        return float(self.motion.omega)
