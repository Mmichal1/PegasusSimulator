#!/usr/bin/env python
"""
| File: linear_shuttle_controller.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Target drone that flies straight-line back-and-forth passes at a fixed altitude.
|
| The motion is a smooth 1-D shuttle along a horizontal axis:
|       p(t) = center + axis * amplitude * sin(omega * t)
| so the target repeatedly crosses in front of a pursuer. Near mid-pass the velocity is
| roughly constant, which is exactly the regime the constant-velocity intercept solver in
| forward_intercept_controller assumes.
|
| It reuses CircularController's startup hover-hold and geometric tracking; only the
| trajectory definition is replaced.
"""

import numpy as np

from circular_controller import CircularController


class LinearShuttleController(CircularController):
    """Fly back and forth along a straight horizontal line at constant altitude."""

    def __init__(
        self,
        center=(0.0, 0.0, 2.0),
        axis=(0.0, 1.0, 0.0),
        amplitude=6.0,
        angular_velocity=0.35,
        startup_delay=0.0,
        **kwargs,
    ):
        self.line_center = np.array(center, dtype=float)

        axis_vec = np.array(axis, dtype=float)
        axis_norm = float(np.linalg.norm(axis_vec))
        self.axis = axis_vec / axis_norm if axis_norm > 1e-9 else np.array([0.0, 1.0, 0.0])
        self.amplitude = float(amplitude)

        # CircularController needs radius/height/center, but pd()/derivatives are fully
        # overridden below so those circle parameters are never used for the trajectory.
        super().__init__(
            radius=0.0,
            height=float(self.line_center[2]),
            center=(float(self.line_center[0]), float(self.line_center[1])),
            angular_velocity=angular_velocity,
            startup_delay=startup_delay,
            show_camera_feed=False,
            **kwargs,
        )

    def spawn_position(self) -> np.ndarray:
        """World position at t=0 (sin(0)=0), i.e. the centre of the shuttle line."""
        return self.line_center.copy()

    def pd(self, t, s=0.6, reverse=False):
        phase = self.angular_velocity * t
        return self.line_center + self.axis * (self.amplitude * np.sin(phase))

    def d_pd(self, t, s=0.6, reverse=False):
        omega = self.angular_velocity
        phase = omega * t
        return self.axis * (self.amplitude * omega * np.cos(phase))

    def dd_pd(self, t, s=0.6, reverse=False):
        omega = self.angular_velocity
        phase = omega * t
        return self.axis * (-self.amplitude * omega**2 * np.sin(phase))

    def ddd_pd(self, t, s=0.6, reverse=False):
        omega = self.angular_velocity
        phase = omega * t
        return self.axis * (-self.amplitude * omega**3 * np.cos(phase))

    def yaw_d(self, t, s=0.6):
        # Keep a constant heading so direction reversals do not induce a yaw flip.
        return 0.0

    def d_yaw_d(self, t, s=0.6):
        return 0.0
