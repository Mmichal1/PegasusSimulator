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
| - The onboard camera is recorded to MP4 with no overlays (PEGASUS_RECORD=0 to
|   disable). Files go to examples/recordings/ unless PEGASUS_RECORD_PATH is set.
|
| Useful environment variables:
|   PEGASUS_TARGET_SEED=0
|   PEGASUS_TARGET_MOTION=random|stationary|straight|circle
|   PEGASUS_GUIDANCE_MODE=camera_pn|pure_pursuit|oracle_pn
|   PEGASUS_RUN_DURATION_S=22
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
PURSUER_SPAWN = np.array([0.0, 0.0, 1.5])
TARGET_SPAWN_NOMINAL = np.array([0.0, 3.0, 16.0])
TARGET_SPAWN_JITTER_M = (1.2, 1.2, 0.6)
TARGET_SPEED_MPS = (0.1, 0.2)
TARGET_CIRCLE_DIAMETER_M = (3.0, 8.0)
TARGET_MOTION_STARTUP_S = 1.0
TARGET_SEED = int(os.getenv("PEGASUS_TARGET_SEED", "0"))
TARGET_MOTION = os.getenv("PEGASUS_TARGET_MOTION", "random").strip().lower()

PURSUER_SPEED = 2.1
NAV_CONSTANT = float(os.getenv("PEGASUS_PN_NAV_CONSTANT", "4.5"))
MAX_LATERAL_ACCEL_MPS2 = float(
    os.getenv("PEGASUS_PN_MAX_LATERAL_ACCEL_MPS2", "6.0")
)
MAX_TURN_RATE_RAD_S = float(os.getenv("PEGASUS_PN_MAX_TURN_RATE_RAD_S", "2.5"))
BEARING_FILTER_TAU_S = float(os.getenv("PEGASUS_PN_BEARING_FILTER_TAU_S", "0.08"))
LOS_RATE_FILTER_TAU_S = float(os.getenv("PEGASUS_PN_LOS_RATE_FILTER_TAU_S", "0.15"))
BEARING_COAST_TIME_S = float(os.getenv("PEGASUS_PN_BEARING_COAST_TIME_S", "0.35"))
CLOSING_FLOOR_FRACTION = float(
    os.getenv("PEGASUS_PN_CLOSING_FLOOR_FRACTION", "0.35")
)
CAMERA_CONE_DEG = float(os.getenv("PEGASUS_PN_CAMERA_CONE_DEG", "30.0"))
GUIDANCE_MODE = os.getenv("PEGASUS_GUIDANCE_MODE", "camera_pn").strip().lower()
STARTUP_DELAY_S = 5.0

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
HIT_RADIUS_M = float(os.getenv("PEGASUS_HIT_RADIUS_M", "0.5"))

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
    "fov": 70.0,
    "clipping_range": (0.08, 100.0),
}


class CadPNInterceptController(PNInterceptController):
    """Rocket interceptor with camera-bearing PN and diagnostic ablation modes."""

    _MAX_SLEW_RAD_S = 5.5

    def __init__(
        self,
        *args,
        recorder: CameraMp4Recorder | None = None,
        guidance_mode: str = "camera_pn",
        bearing_filter_tau_s: float = 0.08,
        los_rate_filter_tau_s: float = 0.15,
        bearing_coast_time_s: float = 0.35,
        closing_floor_fraction: float = 0.35,
        camera_cone_deg: float = 30.0,
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
        self._hit_dir: np.ndarray | None = None
        self._cmd_dir: np.ndarray | None = None
        self._los_rate = np.zeros(3)
        self._guidance_accel = np.zeros(3)
        self._closing_speed = 0.0
        self._guidance_active = False
        self._oracle_target_position: np.ndarray | None = None
        self._oracle_target_velocity: np.ndarray | None = None
        self._recorder = recorder

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
        if self._recorder is not None and sensor_type == "MonocularCamera" and data:
            camera = data.get("camera")
            if camera is not None:
                try:
                    self._recorder.write(camera.get_rgba())
                except Exception as exc:
                    carb.log_warn(f"[CAD PN] camera record failed: {exc}")
        super().update_graphical_sensor(sensor_type, data)
        if sensor_type != "MonocularCamera":
            return
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
        a_along = float(np.clip(2.2 * (v_cmd - v_along), -4.0, 3.5))
        a_perp = -3.5 * v_perp
        perp_norm = float(np.linalg.norm(a_perp))
        if perp_norm > 8.0:
            a_perp *= 8.0 / perp_norm
        specific = (
            a_along * flight_dir
            + a_perp
            + self._guidance_accel
            + np.array([0.0, 0.0, self.g])
        )
        spec_norm = float(np.linalg.norm(specific))
        z_des = specific / spec_norm if spec_norm > 1e-6 else u_los.copy()
        # Stay inside the camera cone (about 30 deg) so the target remains visible.
        if float(np.dot(z_des, u_los)) < self._camera_cone_cos:
            z_des = 0.65 * u_los + 0.35 * z_des
            z_des = z_des / float(np.linalg.norm(z_des))
        if z_des[2] < 0.2:
            z_des = z_des.copy()
            z_des[2] = 0.2
            z_des = z_des / float(np.linalg.norm(z_des))

        u_1 = self.m * float(np.dot(specific, z_des))
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


def _default_record_path(motion) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = "rand" if TARGET_SEED is None else str(TARGET_SEED)
    name = f"chase_{stamp}_{motion.kind}_seed{seed}.mp4"
    return os.path.join(RECORD_DIR, name)


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


def setup_bright_lighting():
    prim_utils.create_prim(
        "/World/Light/DomeLight",
        "DomeLight",
        position=np.array([0.0, 0.0, 10.0]),
        attributes={"inputs:intensity": 3000.0, "inputs:color": (1.0, 1.0, 1.0)},
    )
    prim_utils.create_prim(
        "/World/Light/Sun",
        "DistantLight",
        orientation=Rotation.from_euler("XYZ", [-50.0, 0.0, 30.0], degrees=True).as_quat(),
        attributes={
            "inputs:intensity": 7000.0,
            "inputs:angle": 1.0,
            "inputs:color": (1.0, 0.98, 0.95),
        },
    )
    prim_utils.create_prim(
        "/World/Light/Fill",
        "DistantLight",
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
        )
        self.telemetry_path, self.summary_path = _diagnostic_paths(self.target_motion)
        self.telemetry = (
            InterceptTelemetryWriter(self.telemetry_path)
            if TELEMETRY_ENABLED
            else None
        )
        if self.telemetry is not None:
            carb.log_warn(f"[CAD PN] telemetry CSV: {self.telemetry.path}")
        self.recorder = None
        if RECORD_ENABLED:
            record_path = RECORD_PATH or _default_record_path(self.target_motion)
            self.recorder = CameraMp4Recorder(
                record_path, fps=float(CAMERA_CONFIG["frequency"])
            )
            carb.log_warn(f"[CAD PN] recording raw camera to {self.recorder.path}")

        self.pursuer_controller = CadPNInterceptController(
            detector=detector,
            recorder=self.recorder,
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
        pursuer_config.graphical_sensors = [MonocularCamera("camera", config=CAMERA_CONFIG)]

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

        midpoint = 0.5 * (PURSUER_SPAWN + self.target_motion.spawn)
        self.pg.set_viewport_camera(
            [8.0, -8.0, float(midpoint[2])],
            midpoint.tolist(),
        )
        self.world.reset()
        carb.log_warn(
            "[CAD PN] ready — single CAD rocket pursuer vs target "
            f"({self.target_motion.describe()}, guidance={GUIDANCE_MODE}, "
            f"N={NAV_CONSTANT})."
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
                "motion": self.target_motion.kind,
                "motion_description": self.target_motion.describe(),
                "guidance_mode": GUIDANCE_MODE,
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
            if self.recorder is not None:
                self.recorder.close()
                carb.log_warn(
                    f"[CAD PN] saved {self.recorder.frame_count} camera frames to "
                    f"{self.recorder.path}"
                )
            carb.log_warn("CAD PN intercept simulation closing.")
            self.pursuer_controller._close_debug_preview()
            self.timeline.stop()
            simulation_app.close()


def main():
    PegasusApp().run()


if __name__ == "__main__":
    main()
