#!/usr/bin/env python
"""
| File: 14_python_yolo_torpedo_drone.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Torpedo-style seeker using a pluggable detector (YOLO drone tracker).
|
| Scenario:
| - A pursuer drone has a forward camera and uses the pluggable torpedo controller.
| - A second drone flies an endless circular trajectory (moving target).
| - The pursuer uses `YoloDroneTracker` to detect the target drone in the camera feed
|   and steers toward it.
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

import isaacsim.core.utils.prims as prim_utils
import numpy as np
import omni.timeline
from omni.isaac.core.world import World
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS

# Make examples/utils importable
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(EXAMPLES_DIR, "utils")
sys.path.insert(0, UTILS_DIR)

from circular_controller import CircularController
from torpedo_controller_pluggable import PluggableTorpedoVisionController
from track import YoloDroneTracker


PURSER_SPAWN = np.array([0.0, -6.0, 2.0])
TARGET_CENTER = np.array([18.0, 0.0, 2.0])
STARTUP_DELAY_S = 5.0

CAMERA_CONFIG = {
    "resolution": (640, 480),
    "frequency": 30,
    "depth": False,
    "position": np.array([0.15, 0.0, 0.0]),
    "orientation": np.array([0.0, 0.0, 180.0]),
}


def setup_bright_lighting():
    """Dome + distant lights for an evenly lit scene."""
    prim_utils.create_prim(
        "/World/Light/DomeLight",
        "DomeLight",
        position=np.array([0.0, 0.0, 10.0]),
        attributes={
            "inputs:intensity": 3000.0,
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
            "inputs:intensity": 7000.0,
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
            "inputs:intensity": 2800.0,
            "inputs:angle": 1.2,
            "inputs:color": (0.95, 0.95, 1.0),
        },
    )


class PegasusApp:
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

        # Ensure the YOLO model path below resolves as requested by the user.
        # (The repository stores the .pt in examples/utils/)
        os.chdir(UTILS_DIR)

        tracker_instance = YoloDroneTracker(
            model_path="yolo_based_drone_detector.pt",
            tracker_config="botsort.yaml",
            conf_thresh=0.4,
        )

        # ---- Pursuer (vision-guided) ----
        pursuer_controller = PluggableTorpedoVisionController(
            detector=tracker_instance,
            forward_speed=3.0,
            speed_boost_factor=1.0,
            k_yaw=1.4,
            k_alt=1.0,
            cruise_height=float(PURSER_SPAWN[2]),
            search_yaw_rate=0.25,
            startup_delay=STARTUP_DELAY_S,
            image_size=CAMERA_CONFIG["resolution"],
            show_debug_feed=True,
            Kp=[8.0, 8.0, 10.0],
            Kd=[6.0, 6.0, 8.0],
            Ki=[0.4, 0.4, 1.0],
            Kr=[3.0, 3.0, 3.0],
            Kw=[0.4, 0.4, 0.4],
        )

        pursuer_config = MultirotorConfig()
        pursuer_config.backends = [pursuer_controller]
        pursuer_config.graphical_sensors = [MonocularCamera("camera", config=CAMERA_CONFIG)]

        Multirotor(
            "/World/pursuer_drone",
            ROBOTS["Iris"],
            0,
            PURSER_SPAWN.tolist(),
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=pursuer_config,
        )

        # ---- Target drone (endless circular trajectory) ----
        target_controller = CircularController(
            radius=6.0,
            height=float(TARGET_CENTER[2]),
            center=(float(TARGET_CENTER[0]), float(TARGET_CENTER[1])),
            angular_velocity=0.22,
            startup_delay=STARTUP_DELAY_S,
            show_camera_feed=False,
            Ki=[0.5, 0.5, 0.5],
            Kr=[2.0, 2.0, 2.0],
        )

        target_spawn = (TARGET_CENTER + np.array([6.0, 0.0, 0.0])).tolist()
        target_config = MultirotorConfig()
        target_config.backends = [target_controller]

        Multirotor(
            "/World/target_drone",
            ROBOTS["Iris"],
            1,
            target_spawn,
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=target_config,
        )

        self.pg.set_viewport_camera(
            [PURSER_SPAWN[0] - 10.0, PURSER_SPAWN[1] - 10.0, PURSER_SPAWN[2] + 6.0],
            TARGET_CENTER.tolist(),
        )

        self.world.reset()
        carb.log_info(
            "YOLO torpedo demo ready — pursuer uses YoloDroneTracker to chase target drone."
        )

    def run(self):
        self.timeline.play()

        while simulation_app.is_running():
            self.world.step(render=True)

        carb.log_warn("YOLO torpedo simulation closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    PegasusApp().run()


if __name__ == "__main__":
    main()

