#!/usr/bin/env python
"""
| File: 12_python_endless_circles.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Single-vehicle example that flies an endless horizontal circle using a custom
| Python control backend (no PX4 or ROS). Includes a forward-facing onboard camera whose feed
| is shown in a separate OpenCV window (see examples/utils/camera_preview.py for Isaac Sim notes).
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import os
import sys

import numpy as np
import omni.timeline
from omni.isaac.core.world import World
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/utils")
from circular_controller import CircularController

# Circle parameters (meters and rad/s)
CIRCLE_RADIUS = 5.0
CIRCLE_HEIGHT = 5.0
CIRCLE_CENTER = (0.0, 0.0)
CIRCLE_ANGULAR_VELOCITY = 0.4

# Forward-facing onboard camera (lower resolution for a responsive preview window)
CAMERA_CONFIG = {
    "resolution": (640, 480),
    "frequency": 30,
    "depth": False,
    "position": np.array([0.15, 0.0, 0.0]),
    "orientation": np.array([0.0, 0.0, 180.0]),
}


class PegasusApp:
    """Standalone app: one Iris quadrotor tracking a horizontal circle forever."""

    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        spawn_xy = (
            CIRCLE_CENTER[0] + CIRCLE_RADIUS,
            CIRCLE_CENTER[1],
            CIRCLE_HEIGHT,
        )

        self.controller = CircularController(
            radius=CIRCLE_RADIUS,
            height=CIRCLE_HEIGHT,
            center=CIRCLE_CENTER,
            angular_velocity=CIRCLE_ANGULAR_VELOCITY,
            Ki=[0.5, 0.5, 0.5],
            Kr=[2.0, 2.0, 2.0],
            show_camera_feed=True,
        )

        config = MultirotorConfig()
        config.backends = [self.controller]
        config.graphical_sensors = [MonocularCamera("camera", config=CAMERA_CONFIG)]

        Multirotor(
            "/World/quadrotor1",
            ROBOTS["Iris"],
            0,
            list(spawn_xy),
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config,
        )

        self.world.reset()

    def run(self):
        self.timeline.play()

        while simulation_app.is_running():
            self.world.step(render=True)

        carb.log_warn("PegasusApp Simulation App is closing.")
        self.controller._close_camera_window()
        self.timeline.stop()
        simulation_app.close()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()
