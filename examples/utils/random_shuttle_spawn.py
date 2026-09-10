#!/usr/bin/env python
"""
| File: random_shuttle_spawn.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Randomized spawn point + flight direction for the LinearShuttleController
| target drone, shared by the interceptor examples (15 and 16).
|
| The spawn point equals the shuttle centre (the trajectory is
| ``center + axis * amplitude * sin(omega t)``, so at t=0 the drone sits at ``center``).
| It is drawn uniformly from an axis-aligned box. The flight direction is the world +Y axis
| rotated about the vertical Z axis by a random angle within +-max_yaw_deg, so the target
| still crosses roughly along Y but from a slightly different heading each run. Because the
| rotation is purely horizontal, the shuttle stays at a constant altitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ShuttleSpawn:
    """Randomized target spawn/geometry for a LinearShuttleController."""

    center: np.ndarray   # spawn position == shuttle centre, world frame [m]
    axis: np.ndarray     # unit flight direction (horizontal)
    yaw_deg: float       # heading offset of `axis` from world +Y [deg]


def sample_shuttle_spawn(
    x_range: tuple[float, float] = (4.0, 16.0),
    y_range: tuple[float, float] = (-5.0, 5.0),
    z_range: tuple[float, float] = (0.5, 5.0),
    max_yaw_deg: float = 30.0,
    seed: int | None = None,
) -> ShuttleSpawn:
    """
    Draw a random target spawn point and (slightly randomized) flight direction.

    Args:
        x_range/y_range/z_range: inclusive uniform sampling bounds for the spawn point [m].
        max_yaw_deg: the flight direction is +Y rotated about Z by a uniform angle in
            [-max_yaw_deg, +max_yaw_deg] degrees, so it stays "somewhat aligned" with +Y.
        seed: optional seed for a reproducible spawn (None -> different every run).

    Returns:
        ShuttleSpawn with ``center`` (spawn) and unit ``axis`` (flight direction).
    """
    rng = np.random.default_rng(seed)

    center = np.array(
        [
            rng.uniform(*x_range),
            rng.uniform(*y_range),
            rng.uniform(*z_range),
        ]
    )

    yaw_deg = float(rng.uniform(-max_yaw_deg, max_yaw_deg))
    yaw = np.radians(yaw_deg)
    # +Y rotated about +Z by `yaw`: [-sin, cos, 0]. Horizontal -> constant altitude.
    axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])

    return ShuttleSpawn(center=center, axis=axis, yaw_deg=yaw_deg)
