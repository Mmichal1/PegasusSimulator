#!/usr/bin/env python
"""
| File: 18_python_cad_pn_intercept.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Camera-bearing PN for a CAD rocket intercepting one Iris target.
|
| Scenario:
| - The pursuer is the CAD quadrotor (examples/17), with the camera on the nose /
|   camera stand at the beak, looking up the rocket axis.
| - One Iris target is stationary or follows a slow straight/circular path.
| - Camera-bearing PN uses centroid bearing rate and interceptor state only.
|   Simulator target truth is restricted to scoring and explicit oracle diagnostics.
| - The onboard camera, interceptor and target third-person cameras, and a
|   stationary ground camera are recorded to MP4 (PEGASUS_RECORD=0 to disable).
|   Files go to
|   examples/recordings/ unless PEGASUS_RECORD_PATH is set.
|
| Useful environment variables:
|   PEGASUS_TARGET_SEED=0
|   PEGASUS_TARGET_MOTION=random|stationary|straight|circle
|   PEGASUS_GUIDANCE_MODE=camera_pn|pure_pursuit|oracle_pn
|   PEGASUS_ENGAGEMENT=vertical|horizontal
|   PEGASUS_TARGET_SPEED_MPS=0.5
|   PEGASUS_PURSUER_SPEED_MPS=3.0
|   PEGASUS_TARGET_X_AT_LAUNCH_M=5.0
|   PEGASUS_TARGET_INITIAL_X_M=<override automatic placement>
|   PEGASUS_TARGET_ALTITUDE_M=15.0
|   PEGASUS_PURSUER_START_ALTITUDE_M=1.5
|   PEGASUS_HORIZONTAL_LOCK_DELAY_S=3.5
|   PEGASUS_RUN_DURATION_S=22
|   PEGASUS_RECORD=0
|   PEGASUS_RECORD_PATH=/path/onboard.mp4  # third-person siblings are derived
|   PEGASUS_TELEMETRY=1
|   PEGASUS_TELEMETRY_PATH=/path/run.csv
|   PEGASUS_SUMMARY_PATH=/path/run.summary.json
|
| The raw STEP file must already be converted to USD once in the Isaac Sim GUI:
|     examples/models/dron_do_symulacji.usd
"""

import os
import sys
import csv
import json
from datetime import datetime

import carb
from isaacsim import SimulationApp

HEADLESS = os.getenv("PEGASUS_HEADLESS", "0") == "1"
simulation_app = SimulationApp({"headless": HEADLESS})

import isaacsim.core.utils.prims as prim_utils
import numpy as np
import omni.timeline
from isaacsim.sensors.camera.camera import Camera
from omni.isaac.core.world import World
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(EXAMPLES_DIR, "utils")
sys.path.insert(0, UTILS_DIR)

from cad_multirotor_usd import bake_cad_multirotor_usd
from camera_bearing_pn import (
    CameraBearingPN,
    step_flight_direction,
    true_pn_acceleration,
)
from camera_mp4_recorder import CameraMp4Recorder
from filtered_botsort_detector_adapter import FilteredBotSortDetectorAdapter
from filtered_detection.lib import BotSortFilteredTracker
from pn_intercept_controller import PNInterceptController
from random_target_motion import RandomTargetController, sample_target_motion
from world_intercept import CAMERA_TO_BODY_UPWARD

MODELS_DIR = os.path.join(EXAMPLES_DIR, "models")
CAD_USD = os.path.join(MODELS_DIR, "dron_do_symulacji.usd")
VEHICLE_USD = os.path.join(MODELS_DIR, "dron_do_symulacji_multirotor.usd")

BODY_MASS_KG = 1.3
ENGAGEMENT = os.getenv("PEGASUS_ENGAGEMENT", "vertical").strip().lower()
if ENGAGEMENT not in {"vertical", "horizontal"}:
    raise ValueError("PEGASUS_ENGAGEMENT must be 'vertical' or 'horizontal'")
HORIZONTAL_CHASE = ENGAGEMENT == "horizontal"

TARGET_SPEED = float(
    os.getenv("PEGASUS_TARGET_SPEED_MPS", "0.5" if HORIZONTAL_CHASE else "0.15")
)
HORIZONTAL_LOCK_DELAY_S = float(
    os.getenv("PEGASUS_HORIZONTAL_LOCK_DELAY_S", "3.5")
)
TARGET_MOTION_STARTUP_S = 1.0
TARGET_X_AT_LAUNCH_M = float(os.getenv("PEGASUS_TARGET_X_AT_LAUNCH_M", "5.0"))
target_initial_x_env = os.getenv("PEGASUS_TARGET_INITIAL_X_M")
TARGET_INITIAL_X_M = (
    float(target_initial_x_env)
    if target_initial_x_env is not None
    else TARGET_X_AT_LAUNCH_M
    - TARGET_SPEED * max(0.0, HORIZONTAL_LOCK_DELAY_S - TARGET_MOTION_STARTUP_S)
)

PURSUER_SPAWN = (
    np.array([0.0, 0.0, float(os.getenv("PEGASUS_PURSUER_START_ALTITUDE_M", "1.5"))])
    if HORIZONTAL_CHASE
    else np.array([0.0, 0.0, 1.5])
)
TARGET_SPAWN_NOMINAL = (
    np.array(
        [
            TARGET_INITIAL_X_M,
            0.0,
            float(os.getenv("PEGASUS_TARGET_ALTITUDE_M", "15.0")),
        ]
    )
    if HORIZONTAL_CHASE
    else np.array([0.0, 3.0, 16.0])
)
TARGET_SPAWN_JITTER_M = (
    (0.0, 0.0, 0.0) if HORIZONTAL_CHASE else (1.2, 1.2, 0.6)
)
TARGET_SPEED_MPS = (TARGET_SPEED, TARGET_SPEED)
TARGET_CIRCLE_DIAMETER_M = (3.0, 8.0)
TARGET_SEED = int(os.getenv("PEGASUS_TARGET_SEED", "0"))
TARGET_MOTION = os.getenv(
    "PEGASUS_TARGET_MOTION", "straight" if HORIZONTAL_CHASE else "random"
).strip().lower()

PURSUER_SPEED = float(
    os.getenv("PEGASUS_PURSUER_SPEED_MPS", "3.0" if HORIZONTAL_CHASE else "2.1")
)
NAV_CONSTANT = float(os.getenv("PEGASUS_PN_NAV_CONSTANT", "4.5"))
MAX_LATERAL_ACCEL_MPS2 = float(
    os.getenv(
        "PEGASUS_PN_MAX_LATERAL_ACCEL_MPS2",
        "1.5" if HORIZONTAL_CHASE else "6.0",
    )
)
MAX_TURN_RATE_RAD_S = float(
    os.getenv("PEGASUS_PN_MAX_TURN_RATE_RAD_S", "1.0" if HORIZONTAL_CHASE else "2.5")
)
BEARING_FILTER_TAU_S = float(
    os.getenv("PEGASUS_PN_BEARING_FILTER_TAU_S", "0.20" if HORIZONTAL_CHASE else "0.08")
)
LOS_RATE_FILTER_TAU_S = float(
    os.getenv("PEGASUS_PN_LOS_RATE_FILTER_TAU_S", "0.50" if HORIZONTAL_CHASE else "0.15")
)
BEARING_COAST_TIME_S = float(
    os.getenv("PEGASUS_PN_BEARING_COAST_TIME_S", "3.0" if HORIZONTAL_CHASE else "0.35")
)
CLOSING_FLOOR_FRACTION = float(
    os.getenv("PEGASUS_PN_CLOSING_FLOOR_FRACTION", "0.35")
)
CAMERA_CONE_DEG = float(
    os.getenv(
        "PEGASUS_PN_CAMERA_CONE_DEG",
        str(
            (
                23.5 + max(0.0, PURSUER_SPEED - 3.0)
                if PURSUER_SPEED <= 4.5
                else max(12.0, 25.0 - (19.0 / 3.0) * (PURSUER_SPEED - 4.5))
            )
            if HORIZONTAL_CHASE
            else 30.0
        ),
    )
)
GUIDANCE_MODE = os.getenv("PEGASUS_GUIDANCE_MODE", "camera_pn").strip().lower()
# In horizontal mode, no-target search already hovers upright. Launch immediately
# when the first camera bearing becomes available instead of imposing a second hold.
STARTUP_DELAY_S = 0.0 if HORIZONTAL_CHASE else 5.0
MAX_FORWARD_ACCEL_MPS2 = float(
    os.getenv(
        "PEGASUS_MAX_FORWARD_ACCEL_MPS2",
        "18.0" if HORIZONTAL_CHASE else "3.5",
    )
)

FILTERED_DETECTION_DIR = os.path.join(EXAMPLES_DIR, "filtered_detection")
DETECTOR_MODEL = os.path.join(
    FILTERED_DETECTION_DIR, "models", "yolo_based_drone_detector.pt"
)
DETECTOR_DEVICE = os.getenv("PEGASUS_DETECTOR_DEVICE", "cpu")
RUN_DURATION_S = float(os.getenv("PEGASUS_RUN_DURATION_S", "0"))
PN_TELEMETRY_INTERVAL_S = float(os.getenv("PEGASUS_PN_TELEMETRY_INTERVAL_S", "0.5"))
RECORD_ENABLED = os.getenv("PEGASUS_RECORD", "1") != "0"
RECORD_DIR = os.getenv(
    "PEGASUS_RECORD_DIR", os.path.join(EXAMPLES_DIR, "recordings")
)
RECORD_PATH = os.getenv("PEGASUS_RECORD_PATH")
TELEMETRY_ENABLED = os.getenv("PEGASUS_TELEMETRY", "1") != "0"
TELEMETRY_PATH = os.getenv("PEGASUS_TELEMETRY_PATH")
SUMMARY_PATH = os.getenv("PEGASUS_SUMMARY_PATH")
HIT_RADIUS_M = float(
    os.getenv("PEGASUS_HIT_RADIUS_M", "0.25" if HORIZONTAL_CHASE else "0.5")
)

# CAD camera stand is at ~[0, 0, 0.13] m; the nose cone (`gora`) extends to ~0.15 m.
# Sit the camera above the beak so it is not inside the mesh, looking up body +Z.
CAMERA_OFFSET_BODY = np.array([0.0, 0.0, 0.25])
CAMERA_CONFIG = {
    "resolution": (640, 480),
    "frequency": 30,
    "depth": False,
    "position": CAMERA_OFFSET_BODY,
    # Opposite pitch of the previous mount: [0, -90, 180] was looking down at
    # the CAD body. +90 deg pitches the boresight to body +Z (up the beak).
    "orientation": np.array([0.0, 90.0, 180.0]),
    # A wide lens made the side-on Iris too small for the detector at launch.
    "fov": 70.0,
    "clipping_range": (0.08, 100.0),
}
PURSUER_THIRD_PERSON_CAMERA_CONFIG = {
    "resolution": CAMERA_CONFIG["resolution"],
    "frequency": CAMERA_CONFIG["frequency"],
    "depth": False,
    # Follow behind the rocket's -Z tail and look forward along its +Z nose.
    "position": np.array([0.0, 0.0, -3.0]),
    "orientation": CAMERA_CONFIG["orientation"],
    "fov": 80.0,
    "clipping_range": (0.1, 150.0),
}
TARGET_THIRD_PERSON_CAMERA_CONFIG = {
    "resolution": CAMERA_CONFIG["resolution"],
    "frequency": CAMERA_CONFIG["frequency"],
    "depth": False,
    "fov": 80.0,
    "clipping_range": (0.1, 150.0),
}
GROUND_CAMERA_CONFIG = {
    "resolution": (1280, 720),
    "frequency": CAMERA_CONFIG["frequency"],
    "depth": False,
    "fov": 80.0,
    "clipping_range": (0.1, 200.0),
}

OUTDOOR_SKY_HDR = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.5/NVIDIA/Assets/Skies/Cloudy/abandoned_parking_4k.hdr"
)


class CadPNInterceptController(PNInterceptController):
    """Rocket interceptor with camera-bearing PN and diagnostic ablation modes."""

    _MAX_SLEW_RAD_S = 5.5

    def __init__(
        self,
        *args,
        recorder: CameraMp4Recorder | None = None,
        third_person_recorder: CameraMp4Recorder | None = None,
        guidance_mode: str = "camera_pn",
        bearing_filter_tau_s: float = 0.08,
        los_rate_filter_tau_s: float = 0.15,
        bearing_coast_time_s: float = 0.35,
        closing_floor_fraction: float = 0.35,
        camera_cone_deg: float = 30.0,
        max_forward_accel_mps2: float = 3.5,
        forward_speed_gain: float = 2.2,
        gravity_compensation_priority: bool = False,
        altitude_thrust_priority: bool = False,
        pn_los_anchor_tau_s: float = 0.0,
        pn_accel_feedforward: float = 1.0,
        launch_direction: np.ndarray | None = None,
        launch_acquisition_time_s: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        valid_modes = {"pure_pursuit", "camera_pn", "oracle_pn"}
        if guidance_mode not in valid_modes:
            raise ValueError(
                f"guidance_mode must be one of {sorted(valid_modes)}, "
                f"got {guidance_mode!r}"
            )
        self.guidance_mode = guidance_mode
        self._bearing_pn = CameraBearingPN(
            nav_constant=self.nav_constant,
            max_lateral_accel_mps2=self.max_lateral_accel_mps2,
            bearing_time_constant_s=bearing_filter_tau_s,
            los_rate_time_constant_s=los_rate_filter_tau_s,
            coast_time_s=bearing_coast_time_s,
            closing_speed_floor_fraction=closing_floor_fraction,
        )
        self._camera_cone_cos = float(
            np.cos(np.radians(np.clip(camera_cone_deg, 1.0, 89.0)))
        )
        self.max_forward_accel_mps2 = float(max_forward_accel_mps2)
        self.forward_speed_gain = float(forward_speed_gain)
        self.gravity_compensation_priority = bool(gravity_compensation_priority)
        self.altitude_thrust_priority = bool(altitude_thrust_priority)
        self.pn_los_anchor_tau_s = float(pn_los_anchor_tau_s)
        self.pn_accel_feedforward = float(pn_accel_feedforward)
        self.launch_direction = None
        if launch_direction is not None:
            launch = np.asarray(launch_direction, dtype=float)
            launch_norm = float(np.linalg.norm(launch))
            if launch_norm > 1e-9:
                self.launch_direction = launch / launch_norm
        self.launch_acquisition_time_s = float(launch_acquisition_time_s)
        self._hit_dir: np.ndarray | None = None
        self._cmd_dir: np.ndarray | None = None
        self._los_rate = np.zeros(3)
        self._guidance_accel = np.zeros(3)
        self._closing_speed = 0.0
        self._guidance_active = False
        self._oracle_target_position: np.ndarray | None = None
        self._oracle_target_velocity: np.ndarray | None = None
        self._recorder = recorder
        self._third_person_recorder = third_person_recorder

    def reset_episode(self) -> None:
        super().reset_episode()
        self._bearing_pn.reset()
        self._hit_dir = None
        self._cmd_dir = None
        self._los_rate = np.zeros(3)
        self._guidance_accel = np.zeros(3)
        self._closing_speed = 0.0
        self._guidance_active = False
        self._oracle_target_position = None
        self._oracle_target_velocity = None

    def _pixel_bearing_world(self) -> np.ndarray | None:
        if not self._has_detection or self._centroid is None:
            return None
        if not self.reveived_first_state:
            return None
        return self._estimator._bearing_unit(self.R.as_matrix(), self._centroid)

    def _set_hit_dir(self, direction: np.ndarray) -> None:
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return
        self._hit_dir = direction / norm

    def _ingest_bearing(self, bearing: np.ndarray | None) -> None:
        if bearing is None or float(bearing[2]) < 0.15:
            return
        if self._bearing_pn.update(self._sim_time, bearing):
            self._set_hit_dir(self._bearing_pn.bearing)

    def set_oracle_target_state(
        self, position: np.ndarray, velocity: np.ndarray
    ) -> None:
        """Inject simulator truth for the explicit diagnostic oracle mode only."""
        self._oracle_target_position = np.asarray(position, dtype=float).copy()
        self._oracle_target_velocity = np.asarray(velocity, dtype=float).copy()

    def update_graphical_sensor(self, sensor_type: str, data: dict) -> None:
        if sensor_type == "MonocularCamera" and data:
            camera = data.get("camera")
            camera_name = str(data.get("camera_name", ""))
            recorder = (
                self._third_person_recorder
                if camera_name.startswith("third_person_pursuer")
                else self._recorder
                if camera_name.startswith("camera")
                else None
            )
            if camera is not None and recorder is not None:
                try:
                    recorder.write(camera.get_rgba())
                except Exception as exc:
                    carb.log_warn(f"[CAD PN] camera record failed: {exc}")
        if (
            sensor_type != "MonocularCamera"
            or not data
            or not str(data.get("camera_name", "")).startswith("camera")
        ):
            return
        super().update_graphical_sensor(sensor_type, data)
        self._ingest_bearing(self._pixel_bearing_world())

    def _slew_toward(self, desired: np.ndarray, dt: float) -> np.ndarray:
        if self._cmd_dir is None or dt <= 0.0:
            self._cmd_dir = desired.copy()
            return self._cmd_dir
        cos = float(np.clip(np.dot(self._cmd_dir, desired), -1.0, 1.0))
        angle = float(np.arccos(cos))
        if angle < 1e-4:
            self._cmd_dir = desired.copy()
            return self._cmd_dir
        step = min(1.0, (self._MAX_SLEW_RAD_S * dt) / angle)
        blended = (1.0 - step) * self._cmd_dir + step * desired
        norm = float(np.linalg.norm(blended))
        self._cmd_dir = blended / norm if norm > 1e-9 else desired.copy()
        return self._cmd_dir

    def _vision_references(self, dt):
        self._guidance_active = False
        self._guidance_accel[:] = 0.0
        self._los_rate[:] = 0.0
        self._closing_speed = 0.0

        u_los: np.ndarray | None = None
        rate_valid = False
        coasting = False

        if self.guidance_mode == "oracle_pn":
            if self._oracle_target_position is not None:
                los = self._oracle_target_position - self.p
                rng = float(np.linalg.norm(los))
                if rng > 1e-6:
                    u_los = los / rng
                    target_velocity = (
                        self._oracle_target_velocity
                        if self._oracle_target_velocity is not None
                        else np.zeros(3)
                    )
                    (
                        self._guidance_accel,
                        self._closing_speed,
                        self._los_rate,
                    ) = true_pn_acceleration(
                        los,
                        target_velocity - self.v,
                        self.nav_constant,
                        self.max_lateral_accel_mps2,
                    )
                    rate_valid = True
        else:
            command = self._bearing_pn.command(
                self._sim_time, self.v, self.pursuer_speed
            )
            if command is not None:
                u_los = command.bearing
                self._closing_speed = command.closing_speed
                self._los_rate = command.los_rate
                rate_valid = command.rate_valid
                coasting = command.coasting
                if self.guidance_mode == "camera_pn" and rate_valid:
                    self._guidance_accel = command.acceleration

        if (
            u_los is None
            and self.launch_direction is not None
            and self._sim_time <= self.launch_acquisition_time_s
        ):
            u_los = self.launch_direction.copy()
            self._set_hit_dir(u_los)
            direction = self._slew_toward(u_los, dt)
            self._guidance_active = True
            self._last_status = "launch-acquisition"
            v_ref = direction * self.pursuer_speed
            return (
                self.p + v_ref * dt,
                v_ref,
                np.zeros(3),
                np.zeros(3),
                0.0,
                0.0,
            )

        if u_los is None:
            self._cmd_dir = None
            self._last_status = "searching"
            return self._search_references(dt)

        self._set_hit_dir(u_los)
        if self.guidance_mode == "pure_pursuit" or not rate_valid:
            direction = self._slew_toward(u_los, dt)
            status = "pure-pursuit" if self.guidance_mode == "pure_pursuit" else "acquiring-rate"
        else:
            direction = step_flight_direction(
                self._cmd_dir,
                u_los,
                self._guidance_accel,
                dt,
                self.pursuer_speed,
                max_turn_rate_rad_s=self.max_turn_rate_rad_s,
                min_los_direction_cos=self.min_los_direction_cos,
                los_anchor_time_constant_s=self.pn_los_anchor_tau_s,
            )
            self._cmd_dir = direction
            status = self.guidance_mode

        self._guidance_active = True
        v_ref = direction * self.pursuer_speed
        p_ref = self.p + v_ref * dt
        yaw_ref = float(np.arctan2(direction[1], direction[0]))
        self._last_status = f"{status}-coast" if coasting else status
        self._viz_active = True
        self._viz_coasting = coasting
        self._viz_aim_point = self.p + 20.0 * direction
        return p_ref, v_ref, np.zeros(3), np.zeros(3), yaw_ref, 0.0

    def _apply_control(
        self,
        dt: float,
        p_ref: np.ndarray,
        v_ref: np.ndarray,
        a_ref: np.ndarray,
        j_ref: np.ndarray,
        yaw_ref: float,
        yaw_rate_ref: float,
    ) -> None:
        # While hovering or without an active bearing, use the geometric controller.
        if self._in_startup_hold() or not self._guidance_active:
            super()._apply_control(
                dt, p_ref, v_ref, a_ref, j_ref, yaw_ref, yaw_rate_ref
            )
            return

        # Regulate speed along the PN flight direction, add PN lateral
        # feed-forward, and keep the camera boresight inside the target cone.
        flight_dir = self._cmd_dir if self._cmd_dir is not None else self._hit_dir
        u_los = self._hit_dir if self._hit_dir is not None else flight_dir
        v_along = float(np.dot(self.v, flight_dir))
        v_perp = self.v - v_along * flight_dir
        # Slow down while the velocity is still across the line of sight so the
        # turn can finish before the vehicle shoots past the target.
        speed = float(np.linalg.norm(self.v))
        if speed > 0.4:
            align = float(np.clip(v_along / speed, 0.0, 1.0))
        else:
            align = 1.0
        v_cmd = self.pursuer_speed * (0.4 + 0.6 * align)
        a_along = float(
            np.clip(
                self.forward_speed_gain * (v_cmd - v_along),
                -max(4.0, min(self.max_forward_accel_mps2, self.pursuer_speed)),
                self.max_forward_accel_mps2,
            )
        )
        a_perp = -3.5 * v_perp
        perp_norm = float(np.linalg.norm(a_perp))
        if perp_norm > 8.0:
            a_perp *= 8.0 / perp_norm
        specific = (
            a_along * flight_dir
            + a_perp
            + self.pn_accel_feedforward * self._guidance_accel
            + np.array([0.0, 0.0, self.g])
        )
        spec_norm = float(np.linalg.norm(specific))
        z_des = specific / spec_norm if spec_norm > 1e-6 else u_los.copy()
        # Project the thrust/nose direction exactly onto the edge of the camera
        # cone when gravity compensation would rotate the target out of view.
        los_cos = float(np.clip(np.dot(z_des, u_los), -1.0, 1.0))
        if not self.gravity_compensation_priority and los_cos < self._camera_cone_cos:
            lateral = z_des - los_cos * u_los
            lateral_norm = float(np.linalg.norm(lateral))
            if lateral_norm > 1e-9:
                lateral /= lateral_norm
                cone_sin = np.sqrt(max(0.0, 1.0 - self._camera_cone_cos**2))
                z_des = self._camera_cone_cos * u_los + cone_sin * lateral
            else:
                z_des = u_los.copy()
        if z_des[2] < 0.2:
            z_des = z_des.copy()
            z_des[2] = 0.2
            z_des = z_des / float(np.linalg.norm(z_des))

        u_1 = self.m * float(np.dot(specific, z_des))
        if self.altitude_thrust_priority and z_des[2] > 0.1:
            # With the nose constrained near a low-elevation target, increase
            # thrust enough for its vertical component to balance gravity.
            u_1 = max(u_1, self.m * self.g / float(z_des[2]))
        u_1 = float(np.clip(u_1, 0.45 * self.m * self.g, 2.8 * self.m * self.g))

        heading = float(np.arctan2(z_des[1], z_des[0]))
        x_c = np.array([np.cos(heading), np.sin(heading), 0.0])
        if abs(float(z_des[2])) > 0.98:
            x_c = np.array([1.0, 0.0, 0.0])
        y_b = np.cross(z_des, x_c)
        y_b = y_b / (float(np.linalg.norm(y_b)) + 1e-9)
        x_b = np.cross(y_b, z_des)
        r_des = np.c_[x_b, y_b, z_des]
        rotation = self.R.as_matrix()
        e_r = 0.5 * self.vee((r_des.T @ rotation) - (rotation.T @ r_des))
        tau = -(self.Kr @ e_r) - (self.Kw @ self.w)
        if self.vehicle:
            self.input_ref = self.vehicle.force_and_torques_to_velocities(u_1, tau)

    def diagnostic_snapshot(self) -> dict:
        """Return scalar/list diagnostics suitable for CSV serialization."""
        return {
            "guidance_mode": self.guidance_mode,
            "status": self._last_status,
            "detected": bool(self._has_detection),
            "centroid_u": None if self._centroid is None else int(self._centroid[0]),
            "centroid_v": None if self._centroid is None else int(self._centroid[1]),
            "bbox_width_px": (
                None
                if self._distance_m is None or self._distance_m <= 1e-9
                else float(
                    self._estimator.fx
                    * self._estimator.target_width_m
                    / self._distance_m
                )
            ),
            "bearing": None if self._hit_dir is None else self._hit_dir.tolist(),
            "los_rate": self._los_rate.tolist(),
            "closing_speed": self._closing_speed,
            "guidance_accel": self._guidance_accel.tolist(),
            "command_direction": (
                None if self._cmd_dir is None else self._cmd_dir.tolist()
            ),
            "rejected_bearings": self._bearing_pn.rejected_samples,
        }


def _record_paths(motion) -> dict[str, str]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = "rand" if TARGET_SEED is None else str(TARGET_SEED)
    if RECORD_PATH:
        onboard = os.path.abspath(RECORD_PATH)
        stem, extension = os.path.splitext(onboard)
        extension = extension or ".mp4"
        base = stem.removesuffix("_onboard")
    else:
        base = os.path.join(
            RECORD_DIR, f"chase_{stamp}_{motion.kind}_seed{seed}"
        )
        extension = ".mp4"
        onboard = f"{base}_onboard{extension}"
    return {
        "onboard": onboard,
        "pursuer_third_person": f"{base}_pursuer_third_person{extension}",
        "target_third_person": f"{base}_target_third_person{extension}",
        "ground": f"{base}_ground{extension}",
    }


def _look_at_world_orientation(
    camera_position: np.ndarray, look_at: np.ndarray
) -> np.ndarray:
    """Scalar-first quaternion for a +X-forward, +Z-up world camera."""
    forward = np.asarray(look_at, dtype=float) - np.asarray(camera_position, dtype=float)
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    forward /= forward_norm
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up_hint))) > 0.95:
        up_hint = np.array([0.0, 1.0, 0.0])
    left = np.cross(up_hint, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    xyzw = Rotation.from_matrix(np.column_stack((forward, left, up))).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _diagnostic_paths(motion) -> tuple[str, str]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"cad_pn_{stamp}_{GUIDANCE_MODE}_{motion.kind}_seed{TARGET_SEED}"
    diagnostics_dir = os.path.join(EXAMPLES_DIR, "logs", "cad_pn")
    telemetry_path = TELEMETRY_PATH or os.path.join(diagnostics_dir, stem + ".csv")
    summary_path = SUMMARY_PATH or os.path.splitext(telemetry_path)[0] + ".summary.json"
    return telemetry_path, summary_path


class InterceptTelemetryWriter:
    """Fixed-rate CSV diagnostics for camera, guidance, and simulator truth."""

    FIELDS = [
        "sim_time", "seed", "motion", "guidance_mode", "status",
        "detected", "detection_loss_s", "centroid_u", "centroid_v",
        "bbox_width_px", "miss_m", "true_range_m", "bearing_error_deg",
        "nose_los_cos", "closing_speed", "rejected_bearings",
        "p_x", "p_y", "p_z", "v_x", "v_y", "v_z",
        "target_x", "target_y", "target_z",
        "target_vx", "target_vy", "target_vz",
        "bearing_x", "bearing_y", "bearing_z",
        "los_rate_x", "los_rate_y", "los_rate_z",
        "accel_x", "accel_y", "accel_z",
        "cmd_x", "cmd_y", "cmd_z",
    ]

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self.rows = 0

    @staticmethod
    def _vector(row: dict, prefix: str, value, names=("x", "y", "z")) -> None:
        vector = [None, None, None] if value is None else np.asarray(value).tolist()
        for axis, component in zip(names, vector):
            row[f"{prefix}_{axis}"] = component

    def write(
        self,
        ctrl: CadPNInterceptController,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        detection_loss_s: float,
    ) -> None:
        diagnostic = ctrl.diagnostic_snapshot()
        true_los = np.asarray(target_position, dtype=float) - ctrl.p
        true_range = float(np.linalg.norm(true_los))
        true_bearing = true_los / true_range if true_range > 1e-9 else np.zeros(3)
        measured = diagnostic["bearing"]
        bearing_error_deg = None
        if measured is not None and true_range > 1e-9:
            bearing_error_deg = float(
                np.degrees(
                    np.arccos(np.clip(np.dot(measured, true_bearing), -1.0, 1.0))
                )
            )
        nose = ctrl.R.as_matrix()[:, 2]
        row = {
            "sim_time": ctrl._sim_time,
            "seed": TARGET_SEED,
            "motion": self_motion_kind(ctrl),
            "guidance_mode": diagnostic["guidance_mode"],
            "status": diagnostic["status"],
            "detected": int(diagnostic["detected"]),
            "detection_loss_s": detection_loss_s,
            "centroid_u": diagnostic["centroid_u"],
            "centroid_v": diagnostic["centroid_v"],
            "bbox_width_px": diagnostic["bbox_width_px"],
            "miss_m": true_range,
            "true_range_m": true_range,
            "bearing_error_deg": bearing_error_deg,
            "nose_los_cos": (
                float(np.dot(nose, true_bearing)) if true_range > 1e-9 else 0.0
            ),
            "closing_speed": diagnostic["closing_speed"],
            "rejected_bearings": diagnostic["rejected_bearings"],
        }
        self._vector(row, "p", ctrl.p)
        self._vector(row, "v", ctrl.v)
        self._vector(row, "target", target_position)
        self._vector(row, "target", target_velocity, names=("vx", "vy", "vz"))
        self._vector(row, "bearing", diagnostic["bearing"])
        self._vector(row, "los_rate", diagnostic["los_rate"])
        self._vector(row, "accel", diagnostic["guidance_accel"])
        self._vector(row, "cmd", diagnostic["command_direction"])
        self._writer.writerow(row)
        self.rows += 1
        if self.rows % 30 == 0:
            self._file.flush()

    def close(self) -> None:
        self._file.close()


def self_motion_kind(ctrl: CadPNInterceptController) -> str:
    """Keep the telemetry writer independent of the app object."""
    return str(getattr(ctrl, "_diagnostic_motion_kind", TARGET_MOTION))


def setup_outdoor_lighting():
    prim_utils.create_prim(
        "/World/Light/DomeLight",
        "DomeLight",
        position=np.array([0.0, 0.0, 0.0]),
        attributes={
            "inputs:intensity": 1200.0,
            "inputs:color": (1.0, 1.0, 1.0),
            "inputs:texture:file": OUTDOOR_SKY_HDR,
        },
    )
    prim_utils.create_prim(
        "/World/Light/Sun",
        "DistantLight",
        orientation=Rotation.from_euler("XYZ", [-50.0, 0.0, 30.0], degrees=True).as_quat(),
        attributes={
            "inputs:intensity": 3500.0,
            "inputs:angle": 1.0,
            "inputs:color": (1.0, 0.98, 0.95),
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
        setup_outdoor_lighting()

        if not os.path.exists(CAD_USD):
            raise FileNotFoundError(
                f"Converted CAD USD not found at {CAD_USD}. Import the STEP in Isaac Sim first."
            )
        info = bake_cad_multirotor_usd(CAD_USD, VEHICLE_USD, mass_kg=BODY_MASS_KG)
        carb.log_warn(
            "[CAD PN] baked vehicle USD: "
            f"rotors={np.round(info['rotor_positions_m'], 4).tolist()} "
            f"mass={info['mass_kg']} kg"
        )

        tracker = BotSortFilteredTracker(
            model_path=DETECTOR_MODEL,
            tracker_config="botsort.yaml",
            conf_thresh=0.35,
            device=DETECTOR_DEVICE,
            imgsz=640,
            input_is_rgb=True,
            draw=not HEADLESS,
        )
        detector = FilteredBotSortDetectorAdapter(tracker)

        self.target_motion = sample_target_motion(
            TARGET_SPAWN_NOMINAL,
            spawn_jitter_m=TARGET_SPAWN_JITTER_M,
            speed_mps=TARGET_SPEED_MPS,
            circle_diameter_m=TARGET_CIRCLE_DIAMETER_M,
            startup_delay=TARGET_MOTION_STARTUP_S,
            seed=TARGET_SEED,
            kind=TARGET_MOTION,
            heading_deg=0.0 if HORIZONTAL_CHASE else None,
        )
        self.telemetry_path, self.summary_path = _diagnostic_paths(self.target_motion)
        self.telemetry = (
            InterceptTelemetryWriter(self.telemetry_path)
            if TELEMETRY_ENABLED
            else None
        )
        if self.telemetry is not None:
            carb.log_warn(f"[CAD PN] telemetry CSV: {self.telemetry.path}")
        self.recorders: dict[str, CameraMp4Recorder] = {}
        if RECORD_ENABLED:
            for camera_name, record_path in _record_paths(self.target_motion).items():
                self.recorders[camera_name] = CameraMp4Recorder(
                    record_path, fps=float(CAMERA_CONFIG["frequency"])
                )
                carb.log_warn(
                    f"[CAD PN] recording {camera_name} camera to "
                    f"{self.recorders[camera_name].path}"
                )

        self.pursuer_controller = CadPNInterceptController(
            detector=detector,
            recorder=self.recorders.get("onboard"),
            third_person_recorder=self.recorders.get("pursuer_third_person"),
            guidance_mode=GUIDANCE_MODE,
            pursuer_speed=PURSUER_SPEED,
            nav_constant=NAV_CONSTANT,
            use_augmented_pn=False,
            max_lateral_accel_mps2=MAX_LATERAL_ACCEL_MPS2,
            max_turn_rate_rad_s=MAX_TURN_RATE_RAD_S,
            bearing_filter_tau_s=BEARING_FILTER_TAU_S,
            los_rate_filter_tau_s=LOS_RATE_FILTER_TAU_S,
            bearing_coast_time_s=BEARING_COAST_TIME_S,
            closing_floor_fraction=CLOSING_FLOOR_FRACTION,
            camera_cone_deg=CAMERA_CONE_DEG,
            max_forward_accel_mps2=MAX_FORWARD_ACCEL_MPS2,
            forward_speed_gain=1.0 if HORIZONTAL_CHASE else 2.2,
            gravity_compensation_priority=False,
            altitude_thrust_priority=HORIZONTAL_CHASE,
            pn_los_anchor_tau_s=0.35 if HORIZONTAL_CHASE else 0.0,
            pn_accel_feedforward=0.0 if HORIZONTAL_CHASE else 1.0,
            # No horizontal launch shortcut: acquire through the upward camera
            # while hovering, then transition into the chase.
            launch_direction=None,
            launch_acquisition_time_s=0.0,
            telemetry_interval_s=PN_TELEMETRY_INTERVAL_S,
            fov_deg=float(CAMERA_CONFIG["fov"]),
            cruise_height=float(PURSUER_SPAWN[2]),
            search_yaw_rate=0.3,
            max_yaw_rate=1.0,
            speed_ramp_time_s=3.0,
            velocity_time_constant_s=0.4,
            alignment_min=0.2,
            coast_time_s=5.0,
            startup_delay=STARTUP_DELAY_S,
            image_size=CAMERA_CONFIG["resolution"],
            show_debug_feed=not HEADLESS,
            draw_guidance_overlay=False,
            min_altitude=0.6,
            max_altitude=float(TARGET_SPAWN_NOMINAL[2] + 6.0),
            camera_offset_body=tuple(CAMERA_OFFSET_BODY.tolist()),
            camera_to_body=CAMERA_TO_BODY_UPWARD,
            Kp=[8.0, 8.0, 10.0],
            Kd=[6.0, 6.0, 8.0],
            Ki=[0.4, 0.4, 1.0],
            Kr=[2.0, 2.0, 2.0],
            Kw=[0.8, 0.8, 0.8],
        )
        self.pursuer_controller.m = BODY_MASS_KG
        self.pursuer_controller._diagnostic_motion_kind = self.target_motion.kind

        pursuer_config = MultirotorConfig()
        pursuer_config.backends = [self.pursuer_controller]
        pursuer_config.graphical_sensors = [
            MonocularCamera("camera", config=CAMERA_CONFIG),
            MonocularCamera(
                "third_person_pursuer",
                config=PURSUER_THIRD_PERSON_CAMERA_CONFIG,
            ),
        ]

        Multirotor(
            "/World/pursuer_drone",
            VEHICLE_USD,
            0,
            PURSUER_SPAWN.tolist(),
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=pursuer_config,
        )

        self.target_controller = RandomTargetController(
            self.target_motion,
            Ki=[0.5, 0.5, 0.5],
            Kr=[2.0, 2.0, 2.0],
        )
        target_config = MultirotorConfig()
        target_config.backends = [self.target_controller]
        Multirotor(
            "/World/target_drone",
            ROBOTS["Iris"],
            1,
            self.target_motion.spawn.tolist(),
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=target_config,
        )

        self.target_chase_camera = None
        self._last_target_chase_frame_time = -np.inf
        self._target_camera_direction = np.array([-1.0, 0.0, -0.2])
        self._target_camera_direction /= np.linalg.norm(self._target_camera_direction)
        if "target_third_person" in self.recorders:
            self.target_chase_camera = Camera(
                prim_path="/World/target_third_person_camera",
                frequency=float(TARGET_THIRD_PERSON_CAMERA_CONFIG["frequency"]),
                resolution=TARGET_THIRD_PERSON_CAMERA_CONFIG["resolution"],
            )
        self.ground_camera = None
        self._last_ground_frame_time = -np.inf
        self._ground_camera_position = np.array(
            [TARGET_X_AT_LAUNCH_M + 3.0, -14.0, 1.2]
        )
        self._ground_camera_look_at = np.array(
            [TARGET_X_AT_LAUNCH_M + 3.0, 0.0, 9.0]
        )
        if "ground" in self.recorders:
            self.ground_camera = Camera(
                prim_path="/World/ground_interception_camera",
                frequency=float(GROUND_CAMERA_CONFIG["frequency"]),
                resolution=GROUND_CAMERA_CONFIG["resolution"],
            )

        midpoint = 0.5 * (PURSUER_SPAWN + self.target_motion.spawn)
        self.pg.set_viewport_camera(
            [8.0, -8.0, float(midpoint[2])],
            midpoint.tolist(),
        )
        self.world.reset()
        if self.target_chase_camera is not None:
            self.target_chase_camera.initialize()
            self.target_chase_camera.set_resolution(
                TARGET_THIRD_PERSON_CAMERA_CONFIG["resolution"],
                maintain_square_pixels=True,
            )
            self.target_chase_camera.set_clipping_range(
                *TARGET_THIRD_PERSON_CAMERA_CONFIG["clipping_range"]
            )
            self.target_chase_camera.set_frequency(
                float(TARGET_THIRD_PERSON_CAMERA_CONFIG["frequency"])
            )
            self._position_target_chase_camera(
                PURSUER_SPAWN, self.target_motion.spawn
            )
        if self.ground_camera is not None:
            self.ground_camera.initialize()
            self.ground_camera.set_resolution(
                GROUND_CAMERA_CONFIG["resolution"],
                maintain_square_pixels=True,
            )
            self.ground_camera.set_clipping_range(
                *GROUND_CAMERA_CONFIG["clipping_range"]
            )
            self.ground_camera.set_frequency(float(GROUND_CAMERA_CONFIG["frequency"]))
            # 20.955 mm horizontal aperture with a 12.5 mm focal length gives
            # roughly an 80-degree horizontal FOV, wide enough for the launch
            # and intercept to remain in one stationary shot. Isaac uses cm.
            self.ground_camera.set_focal_length(1.25)
            self.ground_camera.set_world_pose(
                position=self._ground_camera_position,
                orientation=_look_at_world_orientation(
                    self._ground_camera_position, self._ground_camera_look_at
                ),
                camera_axes="world",
            )
        carb.log_warn(
            "[CAD PN] ready — single CAD rocket pursuer vs target "
            f"({self.target_motion.describe()}, guidance={GUIDANCE_MODE}, "
            f"N={NAV_CONSTANT})."
        )

    def _position_target_chase_camera(
        self, pursuer_position: np.ndarray, target_position: np.ndarray
    ) -> None:
        if self.target_chase_camera is None:
            return
        to_interceptor = np.asarray(pursuer_position, dtype=float) - np.asarray(
            target_position, dtype=float
        )
        separation = float(np.linalg.norm(to_interceptor))
        if separation > 0.5:
            direction = to_interceptor / separation
            self._target_camera_direction = direction
        else:
            direction = self._target_camera_direction

        # Stay on the far side of the target, with a lateral offset so the
        # target does not completely hide the approaching interceptor.
        up = np.array([0.0, 0.0, 1.0])
        side = np.cross(up, direction)
        side_norm = float(np.linalg.norm(side))
        if side_norm < 1e-6:
            side = np.array([0.0, 1.0, 0.0])
        else:
            side /= side_norm
        camera_position = (
            np.asarray(target_position, dtype=float)
            - 4.0 * direction
            + 1.5 * side
            + np.array([0.0, 0.0, 0.8])
        )
        look_at = np.asarray(target_position, dtype=float) + min(
            2.0, 0.2 * separation
        ) * direction
        self.target_chase_camera.set_world_pose(
            position=camera_position,
            orientation=_look_at_world_orientation(camera_position, look_at),
            camera_axes="world",
        )

    def run(self):
        self.timeline.play()
        last_miss_log = -1.0
        last_telemetry_time = -np.inf
        previous_sim_time = 0.0
        detection_loss_s = 0.0
        closest_miss = np.inf
        closest_pose = None
        hit_logged = False
        end_reason = "simulation_stopped"
        try:
            while simulation_app.is_running():
                self.world.step(render=True)
                ctrl = self.pursuer_controller
                if ctrl.reveived_first_state:
                    target_p = self.target_controller.p
                    target_v = self.target_controller.v
                    self._position_target_chase_camera(ctrl.p, target_p)
                    target_recorder = self.recorders.get("target_third_person")
                    if (
                        self.target_chase_camera is not None
                        and target_recorder is not None
                        and ctrl._sim_time - self._last_target_chase_frame_time
                        >= 1.0 / float(TARGET_THIRD_PERSON_CAMERA_CONFIG["frequency"])
                    ):
                        self._last_target_chase_frame_time = ctrl._sim_time
                        try:
                            target_recorder.write(self.target_chase_camera.get_rgba())
                        except Exception as exc:
                            carb.log_warn(
                                f"[CAD PN] target third-person record failed: {exc}"
                            )
                    ground_recorder = self.recorders.get("ground")
                    if (
                        self.ground_camera is not None
                        and ground_recorder is not None
                        and ctrl._sim_time - self._last_ground_frame_time
                        >= 1.0 / float(GROUND_CAMERA_CONFIG["frequency"])
                    ):
                        self._last_ground_frame_time = ctrl._sim_time
                        try:
                            # Reassert the fixed world pose after physics updates;
                            # unlike a vehicle-mounted sensor, this camera has no
                            # parent transform maintaining its placement.
                            self.ground_camera.set_world_pose(
                                position=self._ground_camera_position,
                                orientation=_look_at_world_orientation(
                                    self._ground_camera_position,
                                    self._ground_camera_look_at,
                                ),
                                camera_axes="world",
                            )
                            ground_recorder.write(self.ground_camera.get_rgba())
                        except Exception as exc:
                            carb.log_warn(f"[CAD PN] ground camera record failed: {exc}")
                    ctrl.set_oracle_target_state(target_p, target_v)
                    miss = float(np.linalg.norm(ctrl.p - target_p))
                    step_dt = max(0.0, ctrl._sim_time - previous_sim_time)
                    previous_sim_time = ctrl._sim_time
                    if ctrl._sim_time >= STARTUP_DELAY_S and not ctrl._has_detection:
                        detection_loss_s += step_dt
                    if miss < closest_miss:
                        closest_miss = miss
                        closest_pose = ctrl.p.copy()
                    if miss < HIT_RADIUS_M and not hit_logged:
                        hit_logged = True
                        carb.log_warn(
                            f"[CAD PN] HIT at t={ctrl._sim_time:.2f}s miss={miss:.2f}m "
                            f"p={np.round(ctrl.p, 2).tolist()}"
                        )
                    if (
                        self.telemetry is not None
                        and ctrl._sim_time - last_telemetry_time >= 1.0 / 30.0
                    ):
                        last_telemetry_time = ctrl._sim_time
                        self.telemetry.write(
                            ctrl, target_p, target_v, detection_loss_s
                        )
                    if (ctrl._sim_time - last_miss_log) >= 0.5:
                        last_miss_log = ctrl._sim_time
                        nose = ctrl.R.as_matrix()[:, 2]
                        los = target_p - ctrl.p
                        los_n = float(np.linalg.norm(los))
                        nose_cos = (
                            float(np.dot(nose, los / los_n)) if los_n > 1e-6 else 0.0
                        )
                        cmd = getattr(ctrl, "_cmd_dir", None)
                        carb.log_warn(
                            f"[CAD PN] t={ctrl._sim_time:.2f}s true_miss={miss:.2f}m "
                            f"nose_los_cos={nose_cos:.3f} "
                            f"cmd={np.round(cmd, 2).tolist() if cmd is not None else None} "
                            f"p={np.round(ctrl.p, 2).tolist()} "
                            f"tgt={np.round(target_p, 2).tolist()}"
                        )
                    if hit_logged and miss > closest_miss + 1.0:
                        end_reason = "hit_flyby_complete"
                        break
                if (
                    RUN_DURATION_S > 0.0
                    and self.pursuer_controller._sim_time >= RUN_DURATION_S
                ):
                    end_reason = "duration_limit"
                    break
        finally:
            carb.log_warn(
                f"[CAD PN] closest approach {closest_miss:.2f}m "
                f"at p={np.round(closest_pose, 2).tolist() if closest_pose is not None else None}"
            )
            summary = {
                "hit": bool(hit_logged),
                "hit_radius_m": HIT_RADIUS_M,
                "closest_approach_m": (
                    None if not np.isfinite(closest_miss) else float(closest_miss)
                ),
                "episode_time_s": float(self.pursuer_controller._sim_time),
                "pursuit_time_s": max(
                    0.0, float(self.pursuer_controller._sim_time - STARTUP_DELAY_S)
                ),
                "end_reason": end_reason,
                "seed": TARGET_SEED,
                "engagement": ENGAGEMENT,
                "motion": self.target_motion.kind,
                "motion_description": self.target_motion.describe(),
                "guidance_mode": GUIDANCE_MODE,
                "target_speed_mps": TARGET_SPEED,
                "pursuer_speed_mps": PURSUER_SPEED,
                "nav_constant": NAV_CONSTANT,
                "max_lateral_accel_mps2": MAX_LATERAL_ACCEL_MPS2,
                "max_turn_rate_rad_s": MAX_TURN_RATE_RAD_S,
                "bearing_filter_tau_s": BEARING_FILTER_TAU_S,
                "los_rate_filter_tau_s": LOS_RATE_FILTER_TAU_S,
                "bearing_coast_time_s": BEARING_COAST_TIME_S,
                "closing_floor_fraction": CLOSING_FLOOR_FRACTION,
                "camera_cone_deg": CAMERA_CONE_DEG,
                "detection_loss_s": float(detection_loss_s),
                "rejected_bearings": int(
                    self.pursuer_controller._bearing_pn.rejected_samples
                ),
                "telemetry_path": (
                    None if self.telemetry is None else self.telemetry.path
                ),
            }
            os.makedirs(os.path.dirname(os.path.abspath(self.summary_path)), exist_ok=True)
            with open(self.summary_path, "w", encoding="utf-8") as summary_file:
                json.dump(summary, summary_file, indent=2)
            carb.log_warn(f"[CAD PN] summary JSON: {self.summary_path}")
            if self.telemetry is not None:
                self.telemetry.close()
            for camera_name, recorder in self.recorders.items():
                recorder.close()
                carb.log_warn(
                    f"[CAD PN] saved {recorder.frame_count} {camera_name} frames to "
                    f"{recorder.path}"
                )
            carb.log_warn("CAD PN intercept simulation closing.")
            self.pursuer_controller._close_debug_preview()
            self.timeline.stop()
            simulation_app.close()


def main():
    PegasusApp().run()


if __name__ == "__main__":
    main()
