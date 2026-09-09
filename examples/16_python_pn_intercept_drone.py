#!/usr/bin/env python
"""
| File: 16_python_pn_intercept_drone.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Vision-only interceptor guided by Proportional Navigation (PN).
|
| Scenario (identical to 15_python_intercept_drone so the two guidance laws can be
| compared directly):
| - A target drone flies straight-line back-and-forth passes (LinearShuttleController).
| - A pursuer drone has a forward camera and detects the target with YOLO
|   (YoloDroneDetectorAdapter, which also reports bounding-box width).
| - The pursuer reconstructs the target's world position/velocity/acceleration from the
|   monocular detection and steers with (augmented) Proportional Navigation: it commands a
|   lateral acceleration proportional to the line-of-sight rotation rate and closing speed,
|   driving the LOS rate to zero for a collision course.
|
| NAV_CONSTANT is the PN gain N (typically 3-5). USE_AUGMENTED_PN adds the estimated
| target-acceleration term, which helps against the target's turnarounds.
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

from linear_shuttle_controller import LinearShuttleController
from pn_intercept_controller import PNInterceptController
from track import YoloDroneTracker
from yolo_detector_adapter import YoloDroneDetectorAdapter

# ---- Scenario geometry (matches 15 for a like-for-like comparison) ----
PURSUER_SPAWN = np.array([0.0, 0.0, 6.5])         # faces +X toward the target line
TARGET_CENTER = np.array([10.0, 0.0, 8.5])         # centre of the shuttle line
TARGET_AXIS = np.array([0.0, 1.0, 0.0])            # target crosses left/right (along Y)
TARGET_AMPLITUDE = 50.0                            # metres each side of centre
TARGET_ANGULAR_VELOCITY = 0.18                     # rad/s; peak speed ~ A * omega

PURSUER_SPEED = 5.0                                # commanded flight speed [m/s]
NAV_CONSTANT = 4.0                                 # PN gain N (typically 3-5)
USE_AUGMENTED_PN = True                            # add target-acceleration term (APN)
STARTUP_DELAY_S = 5.0

CAMERA_CONFIG = {
    "resolution": (640, 480),
    "frequency": 30,
    "depth": False,
    "position": np.array([0.15, 0.0, 0.0]),
    "orientation": np.array([0.0, 0.0, 180.0]),
    "fov": 70.0,
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

        # The YOLO model + tracker config live in examples/utils/.
        os.chdir(UTILS_DIR)

        tracker = YoloDroneTracker(
            model_path="yolo_based_drone_detector.pt",
            tracker_config="botsort.yaml",
            conf_thresh=0.4,
        )
        # Adapter also reports bounding-box width, which the range estimator needs.
        detector = YoloDroneDetectorAdapter(tracker)

        # ---- Pursuer (vision-only, Proportional Navigation guidance) ----
        self.pursuer_controller = PNInterceptController(
            detector=detector,
            pursuer_speed=PURSUER_SPEED,
            nav_constant=NAV_CONSTANT,
            use_augmented_pn=USE_AUGMENTED_PN,
            fov_deg=float(CAMERA_CONFIG["fov"]),
            cruise_height=float(PURSUER_SPAWN[2]),
            search_yaw_rate=0.3,
            # Smooth start-up so the camera doesn't slew the target out of view:
            max_yaw_rate=1.0,
            speed_ramp_time_s=3.0,
            velocity_time_constant_s=0.4,   # a bit snappier than the lead-pursuit example
            alignment_min=0.2,
            coast_time_s=5.0,               # keep guiding on the last trajectory when detection drops
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
        pursuer_config.backends = [self.pursuer_controller]
        pursuer_config.graphical_sensors = [MonocularCamera("camera", config=CAMERA_CONFIG)]

        Multirotor(
            "/World/pursuer_drone",
            ROBOTS["Iris"],
            0,
            PURSUER_SPAWN.tolist(),
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=pursuer_config,
        )

        # ---- Target drone (straight-line back-and-forth passes) ----
        self.target_controller = LinearShuttleController(
            center=tuple(TARGET_CENTER.tolist()),
            axis=tuple(TARGET_AXIS.tolist()),
            amplitude=TARGET_AMPLITUDE,
            angular_velocity=TARGET_ANGULAR_VELOCITY,
            startup_delay=STARTUP_DELAY_S,
            Ki=[0.5, 0.5, 0.5],
            Kr=[2.0, 2.0, 2.0],
        )

        target_config = MultirotorConfig()
        target_config.backends = [self.target_controller]

        Multirotor(
            "/World/target_drone",
            ROBOTS["Iris"],
            1,
            self.target_controller.spawn_position().tolist(),
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=target_config,
        )

        self.pg.set_viewport_camera(
            [PURSUER_SPAWN[0] - 8.0, PURSUER_SPAWN[1] - 10.0, PURSUER_SPAWN[2] + 6.0],
            TARGET_CENTER.tolist(),
        )

        self.world.reset()
        carb.log_info(
            f"PN intercept demo ready — Proportional Navigation (N={NAV_CONSTANT}, "
            f"augmented={USE_AUGMENTED_PN}). LOS/target overlay is drawn on the onboard "
            "camera preview (never on the detector frame)."
        )

    def run(self):
        self.timeline.play()

        while simulation_app.is_running():
            self.world.step(render=True)

        carb.log_warn("PN intercept simulation closing.")
        self.pursuer_controller._close_debug_preview()
        self.timeline.stop()
        simulation_app.close()


def main():
    PegasusApp().run()


if __name__ == "__main__":
    main()
