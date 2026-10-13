"""Fast, Isaac-independent regression tests for camera-bearing PN."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


UTILS_DIR = Path(__file__).resolve().parents[1] / "examples" / "utils"
sys.path.insert(0, str(UTILS_DIR))

from camera_bearing_pn import (  # noqa: E402
    CameraBearingPN,
    step_flight_direction,
    true_pn_acceleration,
)


class CameraBearingPNTests(unittest.TestCase):
    def test_rotating_bearing_generates_correct_lateral_acceleration(self):
        guidance = CameraBearingPN(nav_constant=4.5)
        for frame in range(60):
            timestamp = frame / 30.0
            angle = 0.02 * timestamp
            bearing = np.array([np.sin(angle), 0.0, np.cos(angle)])
            self.assertTrue(guidance.update(timestamp, bearing))

        command = guidance.command(
            2.0, pursuer_velocity=np.array([0.0, 0.0, 2.0]), pursuer_speed_mps=2.1
        )
        self.assertIsNotNone(command)
        self.assertTrue(command.rate_valid)
        self.assertGreater(command.acceleration[0], 0.0)
        self.assertAlmostEqual(command.los_rate[1], 0.02, delta=0.003)

    def test_rejects_track_jump_and_expires_after_coast(self):
        guidance = CameraBearingPN(jump_reject_cos=0.5, coast_time_s=0.35)
        self.assertTrue(guidance.update(0.0, np.array([0.0, 0.0, 1.0])))
        self.assertFalse(guidance.update(0.1, np.array([1.0, 0.0, 0.0])))
        self.assertEqual(guidance.rejected_samples, 1)
        self.assertIsNotNone(guidance.command(0.3, np.zeros(3), 2.1))
        self.assertIsNone(guidance.command(0.36, np.zeros(3), 2.1))

    def test_oracle_true_pn_is_lateral_to_los(self):
        los = np.array([0.0, 3.0, 15.0])
        acceleration, closing, _ = true_pn_acceleration(
            los,
            relative_velocity=np.array([0.2, 0.0, -2.1]),
            nav_constant=4.5,
            max_lateral_accel_mps2=6.0,
        )
        self.assertGreater(closing, 0.0)
        self.assertAlmostEqual(
            float(np.dot(acceleration, los / np.linalg.norm(los))),
            0.0,
            delta=1e-9,
        )

    def test_ideal_camera_pn_hits_stationary_and_moving_targets(self):
        for target_velocity in (
            np.zeros(3),
            np.array([0.2, 0.0, 0.0]),
        ):
            with self.subTest(target_velocity=target_velocity):
                miss = self._simulate_camera_pn(target_velocity)
                self.assertLess(miss, 0.05)

    @staticmethod
    def _simulate_camera_pn(target_velocity: np.ndarray) -> float:
        pursuer = np.array([0.0, 0.0, 1.5])
        target = np.array([0.0, 3.0, 16.0])
        pursuer_speed = 2.1
        direction = None
        guidance = CameraBearingPN(
            nav_constant=4.5, max_lateral_accel_mps2=6.0
        )
        minimum_distance = np.inf
        dt = 1.0 / 60.0

        for step in range(int(12.0 / dt)):
            timestamp = step * dt
            los = target - pursuer
            bearing = los / np.linalg.norm(los)
            if step % 2 == 0:  # 30 Hz camera, 60 Hz guidance
                guidance.update(timestamp, bearing)
            velocity = (
                np.zeros(3) if direction is None else direction * pursuer_speed
            )
            command = guidance.command(timestamp, velocity, pursuer_speed)
            if command is None or not command.rate_valid:
                direction = bearing
            else:
                direction = step_flight_direction(
                    direction,
                    command.bearing,
                    command.acceleration,
                    dt,
                    pursuer_speed,
                    max_turn_rate_rad_s=2.5,
                    min_los_direction_cos=0.25,
                )
            pursuer += direction * pursuer_speed * dt
            target += target_velocity * dt
            minimum_distance = min(
                minimum_distance, float(np.linalg.norm(target - pursuer))
            )
        return minimum_distance


if __name__ == "__main__":
    unittest.main()
