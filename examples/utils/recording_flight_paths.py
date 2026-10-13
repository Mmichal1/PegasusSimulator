#!/usr/bin/env python
"""
| File: recording_flight_paths.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Endless 3D trajectories for detector-recording scenes.
|
| Paths are built for an upward-looking ground camera at the origin. Every
| trajectory is periodic (shuttle or orbit) so a recording can run indefinitely.
| Startup delays, XY phases, Z phases and periods are randomized / staggered so
| drones do not enter the frame together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nonlinear_controller import NonlinearController


@dataclass
class RecordingPath:
    """One endless drone trajectory for a stationary-camera recording scene."""

    kind: str  # "crossing" | "sweep" | "helix"
    spawn: np.ndarray
    speed: float
    startup_delay: float
    z_min: float
    z_max: float
    # shuttle (crossing / sweep): center + direction * amplitude * sin(omega t + phase)
    # helix: circle about origin with radius, omega, phase0
    origin: np.ndarray | None = None
    direction: np.ndarray | None = None
    amplitude: float | None = None
    radius: float | None = None
    omega: float | None = None
    phase0: float | None = None
    z_omega: float | None = None
    z_phase: float | None = None

    def describe(self) -> str:
        spawn = np.round(self.spawn, 2).tolist()
        period = (
            abs(2.0 * np.pi / self.omega)
            if self.omega is not None and abs(self.omega) > 1e-9
            else 0.0
        )
        return (
            f"{self.kind} spawn={spawn} speed={self.speed:.2f} m/s "
            f"period={period:.1f}s delay={self.startup_delay:.1f}s "
            f"z=[{self.z_min:.1f},{self.z_max:.1f}]"
        )


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def _z_of(path: RecordingPath, tau: float) -> float:
    mid = 0.5 * (path.z_min + path.z_max)
    amp = 0.5 * (path.z_max - path.z_min)
    return mid + amp * np.sin(path.z_omega * tau + path.z_phase)


def _dz_of(path: RecordingPath, tau: float) -> float:
    amp = 0.5 * (path.z_max - path.z_min)
    return amp * path.z_omega * np.cos(path.z_omega * tau + path.z_phase)


def _staggered_delays(
    rng: np.random.Generator,
    num_drones: int,
    span_s: tuple[float, float],
) -> np.ndarray:
    """Spread startup delays across ``span_s`` so entries are not simultaneous."""
    low, high = float(span_s[0]), float(span_s[1])
    if num_drones <= 1:
        return np.array([float(rng.uniform(low, high))])
    # Even slots, then shuffle and add small jitter (clamped to the span).
    slots = np.linspace(low, high, num_drones)
    rng.shuffle(slots)
    width = max(0.2, 0.15 * (high - low) / max(num_drones - 1, 1))
    delays = slots + rng.uniform(-width, width, size=num_drones)
    return np.clip(delays, low, high)


def sample_recording_paths(
    num_drones: int = 5,
    z_min: float = 5.0,
    z_max: float = 10.0,
    xy_period_s: tuple[float, float] = (5.0, 10.0),
    max_peak_speed_mps: float = 5.0,
    fov_radius_m: float = 8.0,
    stay_in_view: bool = False,
    startup_delay_span_s: tuple[float, float] | None = None,
    seed: int | None = None,
) -> list[RecordingPath]:
    """
    Draw ``num_drones`` endless trajectories over an upward-looking ground camera.

    Crossing / sweep paths shuttle back and forth; helix paths orbit. When
    ``stay_in_view`` is True, horizontal motion is kept inside a margin of the
    FOV so the drone never leaves the frame (Z still varies in ``[z_min, z_max]``).
    Otherwise amplitudes extend past the FOV so drones enter and leave the shot.
    Delays and phases are staggered when there are multiple drones.
    """
    rng = np.random.default_rng(seed)
    z_min = float(z_min)
    z_max = float(z_max)
    if startup_delay_span_s is None:
        startup_delay_span_s = (0.0, 0.5) if stay_in_view else (0.5, 14.0)
    delays = _staggered_delays(rng, num_drones, startup_delay_span_s)

    # Cycle kinds so a run is mixed, but start index is random (avoids sync patterns).
    kind_cycle = np.array(["crossing", "sweep", "helix"])
    kind_order = np.roll(kind_cycle, int(rng.integers(0, 3)))

    # Keep a margin from the FOV edge so the silhouette stays on camera.
    max_r = 0.70 * float(fov_radius_m) if stay_in_view else None

    paths: list[RecordingPath] = []
    for i in range(num_drones):
        delay = float(delays[i])
        xy_period = float(rng.uniform(*xy_period_s))
        omega_mag = 2.0 * np.pi / max(xy_period, 1.0)
        # Independent Z period / phase so altitude motion is also unsynchronized.
        z_period = float(rng.uniform(7.0, 18.0))
        z_omega = 2.0 * np.pi / z_period
        z_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        kind = str(kind_order[i % len(kind_order)])
        # Random XY phase so even same-kind drones reverse at different times.
        phase0 = float(rng.uniform(0.0, 2.0 * np.pi))

        if kind in ("crossing", "sweep"):
            heading = float(rng.uniform(0.0, 2.0 * np.pi))
            direction = np.array([np.cos(heading), np.sin(heading), 0.0])
            lateral = np.array([-direction[1], direction[0], 0.0])
            if stay_in_view:
                # Shuttle entirely inside the FOV disk (margin max_r).
                amplitude = float(rng.uniform(0.20 * max_r, 0.55 * max_r))
                if kind == "crossing":
                    offset = float(rng.uniform(-0.25, 0.25)) * (max_r - amplitude)
                else:
                    offset = float(rng.uniform(0.20, 0.55)) * (max_r - amplitude)
                    if rng.random() < 0.5:
                        offset = -offset
            else:
                # Amplitude past the FOV so each shuttle leaves and re-enters.
                amplitude = float(rng.uniform(fov_radius_m + 2.0, fov_radius_m + 5.0))
                if kind == "crossing":
                    offset = float(rng.uniform(-0.35, 0.35)) * fov_radius_m
                else:
                    offset = float(rng.uniform(0.35, 0.9)) * fov_radius_m
                    if rng.random() < 0.5:
                        offset = -offset
            center = lateral * offset
            omega = omega_mag
            peak = amplitude * omega
            if peak > max_peak_speed_mps:
                omega = max_peak_speed_mps / amplitude
            speed = float(amplitude * omega)
            spawn = center + direction * (amplitude * np.sin(phase0))
            spawn = np.array([spawn[0], spawn[1], 0.0])
            paths.append(
                RecordingPath(
                    kind=kind,
                    spawn=spawn,
                    speed=speed,
                    startup_delay=delay,
                    z_min=z_min,
                    z_max=z_max,
                    origin=center.copy(),
                    direction=direction,
                    amplitude=amplitude,
                    omega=omega,
                    phase0=phase0,
                    z_omega=z_omega,
                    z_phase=z_phase,
                )
            )
        else:
            if stay_in_view:
                # Orbit fully inside the FOV disk: |center| + radius <= max_r.
                center_r = float(rng.uniform(0.0, 0.25 * max_r))
                radius = float(rng.uniform(0.35 * max_r, max_r - center_r))
            else:
                center_r = float(rng.uniform(0.0, 0.55 * fov_radius_m))
                radius = float(rng.uniform(0.75 * fov_radius_m, 1.25 * fov_radius_m))
            center_ang = float(rng.uniform(0.0, 2.0 * np.pi))
            center = np.array(
                [center_r * np.cos(center_ang), center_r * np.sin(center_ang), 0.0]
            )
            sign = 1.0 if rng.random() < 0.5 else -1.0
            omega = sign * omega_mag
            speed = float(radius * abs(omega))
            if speed > max_peak_speed_mps:
                omega = sign * max_peak_speed_mps / radius
                speed = float(max_peak_speed_mps)
            spawn = np.array(
                [
                    center[0] + radius * np.cos(phase0),
                    center[1] + radius * np.sin(phase0),
                    0.0,
                ]
            )
            paths.append(
                RecordingPath(
                    kind="helix",
                    spawn=spawn,
                    speed=speed,
                    startup_delay=delay,
                    z_min=z_min,
                    z_max=z_max,
                    origin=center,
                    radius=radius,
                    omega=omega,
                    phase0=phase0,
                    z_omega=z_omega,
                    z_phase=z_phase,
                )
            )

        p = paths[-1]
        p.spawn = p.spawn.copy()
        p.spawn[2] = _z_of(p, 0.0)

    return paths


class RecordingFlightController(NonlinearController):
    """Track a periodic ``RecordingPath`` (endless shuttle / orbit + oscillating Z)."""

    def __init__(self, path: RecordingPath, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.total_time = 0.0

    def reset_statistics(self):
        super().reset_statistics()
        self.total_time = 0.0

    def _tau(self, t: float) -> float:
        return max(0.0, float(t) - self.path.startup_delay)

    def pd(self, t, s=0.6, reverse=False):
        tau = self._tau(t)
        p = self.path
        z = _z_of(p, tau)
        phase = p.phase0 + p.omega * tau
        if p.kind in ("crossing", "sweep"):
            xy = p.origin[:2] + p.direction[:2] * (p.amplitude * np.sin(phase))
            return np.array([xy[0], xy[1], z])
        return np.array(
            [
                p.origin[0] + p.radius * np.cos(phase),
                p.origin[1] + p.radius * np.sin(phase),
                z,
            ]
        )

    def d_pd(self, t, s=0.6, reverse=False):
        tau = self._tau(t)
        if tau <= 0.0:
            return np.zeros(3)
        p = self.path
        vz = _dz_of(p, tau)
        phase = p.phase0 + p.omega * tau
        if p.kind in ("crossing", "sweep"):
            scale = p.amplitude * p.omega * np.cos(phase)
            return np.array(
                [p.direction[0] * scale, p.direction[1] * scale, vz]
            )
        return np.array(
            [
                -p.radius * p.omega * np.sin(phase),
                p.radius * p.omega * np.cos(phase),
                vz,
            ]
        )

    def dd_pd(self, t, s=0.6, reverse=False):
        tau = self._tau(t)
        if tau <= 0.0:
            return np.zeros(3)
        p = self.path
        amp_z = 0.5 * (p.z_max - p.z_min)
        az = -amp_z * p.z_omega**2 * np.sin(p.z_omega * tau + p.z_phase)
        phase = p.phase0 + p.omega * tau
        if p.kind in ("crossing", "sweep"):
            scale = -p.amplitude * p.omega**2 * np.sin(phase)
            return np.array(
                [p.direction[0] * scale, p.direction[1] * scale, az]
            )
        return np.array(
            [
                -p.radius * p.omega**2 * np.cos(phase),
                -p.radius * p.omega**2 * np.sin(phase),
                az,
            ]
        )

    def ddd_pd(self, t, s=0.6, reverse=False):
        tau = self._tau(t)
        if tau <= 0.0:
            return np.zeros(3)
        p = self.path
        amp_z = 0.5 * (p.z_max - p.z_min)
        jz = -amp_z * p.z_omega**3 * np.cos(p.z_omega * tau + p.z_phase)
        phase = p.phase0 + p.omega * tau
        if p.kind in ("crossing", "sweep"):
            scale = -p.amplitude * p.omega**3 * np.cos(phase)
            return np.array(
                [p.direction[0] * scale, p.direction[1] * scale, jz]
            )
        return np.array(
            [
                p.radius * p.omega**3 * np.sin(phase),
                -p.radius * p.omega**3 * np.cos(phase),
                jz,
            ]
        )

    def yaw_d(self, t, s=0.6):
        v = self.d_pd(t)
        if float(np.linalg.norm(v[:2])) < 1e-3:
            # Near turnaround, keep last heading from position along the path.
            p = self.path
            if p.kind in ("crossing", "sweep") and p.direction is not None:
                return float(np.arctan2(p.direction[1], p.direction[0]))
            return 0.0
        return float(np.arctan2(v[1], v[0]))

    def d_yaw_d(self, t, s=0.6):
        return 0.0
