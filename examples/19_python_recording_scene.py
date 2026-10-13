#!/usr/bin/env python
"""
| File: 19_python_recording_scene.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Stationary-camera recording scene with one in-FOV drone for detector data.
|
| Scenario:
| - A stationary platform on the ground under the flight volume holds a monocular
|   camera looking straight up (body +Z).
| - One Iris drone flies an endless 3D trajectory that stays inside the camera
|   FOV, with altitude varying in 5–10 m.
| - Outdoor lighting uses an HDR cloudy sky dome.
| - Raw camera frames (no overlays) are written to MP4 under examples/recordings/.
|
| Env vars:
|   PEGASUS_HEADLESS=1
|   PEGASUS_RUN_DURATION_S=40
|   PEGASUS_NUM_DRONES=1
|   PEGASUS_TARGET_SEED=<int>          # reproducible path draws
|   PEGASUS_RECORD=0                   # disable recording
|   PEGASUS_RECORD_PATH=/path/file.mp4
|   PEGASUS_RECORD_DIR=...
"""

import os
import sys
from datetime import datetime

import carb
from isaacsim import SimulationApp

HEADLESS = os.getenv("PEGASUS_HEADLESS", "0") == "1"
simulation_app = SimulationApp({"headless": HEADLESS})

import isaacsim.core.utils.prims as prim_utils
import numpy as np
import omni.timeline
from omni.isaac.core.world import World
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(EXAMPLES_DIR, "utils")
sys.path.insert(0, UTILS_DIR)

from camera_mp4_recorder import CameraMp4Recorder
from nonlinear_controller import NonlinearController
from recording_flight_paths import RecordingFlightController, sample_recording_paths

NUM_DRONES = int(os.getenv("PEGASUS_NUM_DRONES", "1"))
Z_MIN_M = 5.0
Z_MAX_M = 10.0
# Ground platform directly under the flight volume; camera looks along body +Z.
CAMERA_PLATFORM_SPAWN = np.array([0.0, 0.0, 0.25])
CAMERA_PLATFORM_YAW_DEG = 0.0
# Approximate horizontal FOV radius at mid altitude (70 deg FOV, z~7.5 m).
FOV_RADIUS_M = 8.0
# Keep the drone inside this fraction of FOV_RADIUS_M for the whole flight.
STAY_IN_VIEW = True
# Full XY loop period while remaining in frame.
XY_PERIOD_S = (5.0, 10.0)

# Outdoor HDR sky (clouds / real environment lighting) for the dome light.
CLOUDY_SKY_HDR = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.5/NVIDIA/Assets/Skies/Cloudy/abandoned_parking_4k.hdr"
)

TARGET_SEED = os.getenv("PEGASUS_TARGET_SEED")
TARGET_SEED = int(TARGET_SEED) if TARGET_SEED else None
RUN_DURATION_S = float(os.getenv("PEGASUS_RUN_DURATION_S", "40"))
RECORD_ENABLED = os.getenv("PEGASUS_RECORD", "1") != "0"
RECORD_DIR = os.getenv(
    "PEGASUS_RECORD_DIR", os.path.join(EXAMPLES_DIR, "recordings")
)
RECORD_PATH = os.getenv("PEGASUS_RECORD_PATH")

CAMERA_CONFIG = {
    "resolution": (640, 480),
    "frequency": 30,
    "depth": False,
    # Sit above the body mesh; +90 deg pitch aims the Isaac boresight up body +Z.
    "position": np.array([0.0, 0.0, 0.20]),
    "orientation": np.array([0.0, 90.0, 180.0]),
    "fov": 70.0,
    "clipping_range": (0.3, 80.0),
}


class HoverController(NonlinearController):
    """Hold a fixed world position / yaw (stationary camera platform)."""

    def __init__(self, hover_position, yaw_rad: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self._hover = np.asarray(hover_position, dtype=float)
        self._yaw = float(yaw_rad)
        self.total_time = 0.0
        self._recorder: CameraMp4Recorder | None = None

    def reset_statistics(self):
        super().reset_statistics()
        self.total_time = 0.0

    def set_recorder(self, recorder: CameraMp4Recorder | None) -> None:
        self._recorder = recorder

    def pd(self, t, s=0.6, reverse=False):
        return self._hover.copy()

    def d_pd(self, t, s=0.6, reverse=False):
        return np.zeros(3)

    def dd_pd(self, t, s=0.6, reverse=False):
        return np.zeros(3)

    def ddd_pd(self, t, s=0.6, reverse=False):
        return np.zeros(3)

    def yaw_d(self, t, s=0.6):
        return self._yaw

    def d_yaw_d(self, t, s=0.6):
        return 0.0

    def update_graphical_sensor(self, sensor_type: str, data: dict) -> None:
        if self._recorder is None or sensor_type != "MonocularCamera" or not data:
            return
        camera = data.get("camera")
        if camera is None:
            return
        try:
            self._recorder.write(camera.get_rgba())
        except Exception as exc:
            carb.log_warn(f"[RECORD SCENE] camera record failed: {exc}")


def setup_outdoor_sky():
    """Flat-plane ground plus an HDR dome sky (clouds) and a soft sun."""
    prim_utils.create_prim(
        "/World/Light/DomeLight",
        "DomeLight",
        position=np.array([0.0, 0.0, 0.0]),
        attributes={
            "inputs:intensity": 1200.0,
            "inputs:color": (1.0, 1.0, 1.0),
            "inputs:texture:file": CLOUDY_SKY_HDR,
        },
    )
    prim_utils.create_prim(
        "/World/Light/Sun",
        "DistantLight",
        orientation=Rotation.from_euler("XYZ", [-55.0, 0.0, 35.0], degrees=True).as_quat(),
        attributes={
            "inputs:intensity": 3500.0,
            "inputs:angle": 1.0,
            "inputs:color": (1.0, 0.98, 0.92),
        },
    )


def _default_record_path() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = "rand" if TARGET_SEED is None else str(TARGET_SEED)
    name = f"scene_{stamp}_n{NUM_DRONES}_seed{seed}.mp4"
    return os.path.join(RECORD_DIR, name)


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world_settings = {
            "physics_dt": 1.0 / 250.0,
            "stage_units_in_meters": 1.0,
            "rendering_dt": 1.0 / 60.0,
        }
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])
        setup_outdoor_sky()

        self.recorder = None
        if RECORD_ENABLED:
            record_path = RECORD_PATH or _default_record_path()
            self.recorder = CameraMp4Recorder(
                record_path, fps=float(CAMERA_CONFIG["frequency"])
            )
            carb.log_warn(f"[RECORD SCENE] recording raw camera to {self.recorder.path}")

        yaw_rad = np.radians(CAMERA_PLATFORM_YAW_DEG)
        self.camera_controller = HoverController(
            CAMERA_PLATFORM_SPAWN,
            yaw_rad=yaw_rad,
            Ki=[0.5, 0.5, 0.5],
            Kr=[2.0, 2.0, 2.0],
        )
        self.camera_controller.set_recorder(self.recorder)
        camera_config = MultirotorConfig()
        camera_config.backends = [self.camera_controller]
        camera_config.graphical_sensors = [
            MonocularCamera("camera", config=CAMERA_CONFIG)
        ]
        Multirotor(
            "/World/camera_platform",
            ROBOTS["Iris"],
            0,
            CAMERA_PLATFORM_SPAWN.tolist(),
            Rotation.from_euler(
                "XYZ", [0.0, 0.0, CAMERA_PLATFORM_YAW_DEG], degrees=True
            ).as_quat(),
            config=camera_config,
        )

        self.paths = sample_recording_paths(
            num_drones=NUM_DRONES,
            z_min=Z_MIN_M,
            z_max=Z_MAX_M,
            xy_period_s=XY_PERIOD_S,
            fov_radius_m=FOV_RADIUS_M,
            stay_in_view=STAY_IN_VIEW,
            seed=TARGET_SEED,
        )
        self.controllers: list[RecordingFlightController] = []
        for i, path in enumerate(self.paths):
            controller = RecordingFlightController(
                path,
                Ki=[0.5, 0.5, 0.5],
                Kr=[2.0, 2.0, 2.0],
            )
            self.controllers.append(controller)
            config = MultirotorConfig()
            config.backends = [controller]
            Multirotor(
                f"/World/drone_{i}",
                ROBOTS["Iris"],
                i + 1,
                path.spawn.tolist(),
                Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
                config=config,
            )
            carb.log_warn(f"[RECORD SCENE] drone_{i}: {path.describe()}")

        look_at = np.array([0.0, 0.0, 0.5 * (Z_MIN_M + Z_MAX_M)])
        self.pg.set_viewport_camera(
            [12.0, -12.0, 18.0],
            look_at.tolist(),
        )
        self.world.reset()
        carb.log_warn(
            f"[RECORD SCENE] ready — {NUM_DRONES} drones, Z∈[{Z_MIN_M},{Z_MAX_M}] m, "
            f"upward ground camera at {np.round(CAMERA_PLATFORM_SPAWN, 2).tolist()}"
        )

    def run(self):
        self.timeline.play()
        last_log = -1.0
        try:
            while simulation_app.is_running():
                self.world.step(render=True)
                sim_time = (
                    float(self.camera_controller.total_time)
                    if self.camera_controller.reveived_first_state
                    else 0.0
                )
                if sim_time - last_log >= 2.0:
                    last_log = sim_time
                    poses = []
                    for i, ctrl in enumerate(self.controllers):
                        if ctrl.reveived_first_state:
                            poses.append(f"d{i}={np.round(ctrl.p, 1).tolist()}")
                    carb.log_warn(
                        f"[RECORD SCENE] t={sim_time:.1f}s "
                        + (" ".join(poses) if poses else "waiting for state")
                    )
                if RUN_DURATION_S > 0.0 and sim_time >= RUN_DURATION_S:
                    break
        finally:
            if self.recorder is not None:
                self.recorder.close()
                carb.log_warn(
                    f"[RECORD SCENE] saved {self.recorder.frame_count} frames to "
                    f"{self.recorder.path}"
                )
            carb.log_warn("[RECORD SCENE] simulation closing.")
            self.timeline.stop()
            simulation_app.close()


def main():
    PegasusApp().run()


if __name__ == "__main__":
    main()
