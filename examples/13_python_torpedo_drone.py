#!/usr/bin/env python
"""
| File: 13_python_torpedo_drone.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Endless torpedo-style demo — a camera drone visually pursues red balls
| placed on a closed loop at varying heights (no waypoints or target counter).
"""

import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import os
import sys

import isaacsim.core.utils.prims as prim_utils
import numpy as np
import omni.timeline
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.world import World
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/utils")
from torpedo_controller import TorpedoVisionController

DRONE_SPAWN = np.array([0.0, 0.0, 1.5])
BALL_RADIUS = 0.4
NUM_BALLS = 12

FORWARD_SPEED = 2.5
STARTUP_DELAY_S = 5.0

CAMERA_CONFIG = {
    "resolution": (640, 480),
    "frequency": 30,
    "depth": False,
    "position": np.array([0.15, 0.0, 0.0]),
    "orientation": np.array([0.0, 0.0, 180.0]),
}


def generate_ball_loop(num_balls=NUM_BALLS):
    """
    Red balls on a horizontal ellipse with varying altitude so the drone can weave
    through them indefinitely (always another ball ahead to centre in the camera).
    """
    center_x = 22.0
    center_y = 0.0
    radius_x = 15.0
    radius_y = 10.0
    z_base = 1.5
    z_amplitude = 1.2

    positions = []
    for index in range(num_balls):
        angle = 2.0 * np.pi * index / num_balls
        x = center_x + radius_x * np.cos(angle)
        y = center_y + radius_y * np.sin(angle)
        z = z_base + z_amplitude * np.sin(1.9 * angle + 0.6 * index)
        z = float(np.clip(z, 0.8, 3.2))
        positions.append(np.array([x, y, z]))

    return positions


def setup_bright_lighting():
    """Dome + distant lights for an evenly lit scene."""
    prim_utils.create_prim(
        "/World/Light/DomeLight",
        "DomeLight",
        position=np.array([0.0, 0.0, 10.0]),
        attributes={
            "inputs:intensity": 2500.0,
            "inputs:color": (1.0, 1.0, 1.0),
        },
    )

    sun_orientation = Rotation.from_euler("XYZ", [-50.0, 0.0, 30.0], degrees=True).as_quat()
    prim_utils.create_prim(
        "/World/Light/Sun",
        "DistantLight",
        position=np.array([0.0, 0.0, 0.0]),
        orientation=sun_orientation,
        attributes={
            "inputs:intensity": 6000.0,
            "inputs:angle": 1.0,
            "inputs:color": (1.0, 0.98, 0.95),
        },
    )

    prim_utils.create_prim(
        "/World/Light/Fill",
        "DistantLight",
        position=np.array([0.0, 0.0, 0.0]),
        orientation=Rotation.from_euler("XYZ", [-20.0, 0.0, -140.0], degrees=True).as_quat(),
        attributes={
            "inputs:intensity": 2500.0,
            "inputs:angle": 1.2,
            "inputs:color": (0.95, 0.95, 1.0),
        },
    )


def spawn_ball_loop(world, positions):
    for index, position in enumerate(positions):
        ball = VisualSphere(
            prim_path=f"/World/ball_{index}",
            name=f"ball_{index}",
            radius=BALL_RADIUS,
            position=position,
            color=np.array([1.0, 0.05, 0.05]),
        )
        world.scene.add(ball)


class PegasusApp:
    """Endless vision-guided torpedo demo."""

    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world_settings = {
            "physics_dt": 1.0 / 500.0,
            "stage_units_in_meters": 1.0,
            "rendering_dt": 1.0 / 60.0,
        }
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])
        setup_bright_lighting()

        ball_positions = generate_ball_loop()
        spawn_ball_loop(self.world, ball_positions)
        carb.log_info(f"Spawned {len(ball_positions)} balls on a loop (varying height).")

        self.controller = TorpedoVisionController(
            forward_speed=FORWARD_SPEED,
            k_yaw=1.6,
            k_alt=1.2,
            startup_delay=STARTUP_DELAY_S,
            image_size=CAMERA_CONFIG["resolution"],
            show_debug_feed=True,
            Kp=[8.0, 8.0, 10.0],
            Kd=[6.0, 6.0, 8.0],
            Ki=[0.4, 0.4, 1.0],
            Kr=[3.0, 3.0, 3.0],
            Kw=[0.4, 0.4, 0.4],
        )

        config = MultirotorConfig()
        config.backends = [self.controller]
        config.graphical_sensors = [MonocularCamera("camera", config=CAMERA_CONFIG)]

        Multirotor(
            "/World/torpedo_drone",
            ROBOTS["Iris"],
            0,
            DRONE_SPAWN.tolist(),
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config,
        )

        loop_center = np.mean(ball_positions, axis=0)
        self.pg.set_viewport_camera(
            [DRONE_SPAWN[0] - 10.0, DRONE_SPAWN[1] - 10.0, DRONE_SPAWN[2] + 6.0],
            loop_center.tolist(),
        )

        self.world.reset()
        carb.log_info(
            f"Endless torpedo demo — {STARTUP_DELAY_S:.0f}s startup delay, then visual pursuit."
        )

    def run(self):
        self.timeline.play()

        while simulation_app.is_running():
            self.world.step(render=True)

        carb.log_warn("Torpedo simulation closing.")
        self.controller._close_debug_preview()
        self.timeline.stop()
        simulation_app.close()


def main():
    PegasusApp().run()


if __name__ == "__main__":
    main()
