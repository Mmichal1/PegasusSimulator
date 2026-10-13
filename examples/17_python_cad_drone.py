#!/usr/bin/env python
"""
| File: 17_python_cad_drone.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Spawn the user CAD drone as a Pegasus Multirotor.
|
| The raw STEP file (examples/models/dron_do_symulacji.stp) must already be converted
| to USD once in the Isaac Sim GUI (File > Import) as:
|     examples/models/dron_do_symulacji.usd
|
| This example then bakes a Pegasus-compatible vehicle USD:
|   /vehicle/body          rigid body, mass 1.0 kg, COM at the main-body centroid
|   /vehicle/rotor0..3     engine frames
|   /vehicle/rotorN/jointN revolute joints (spin animation)
|
| A hover controller holds the spawn pose so the articulation can be checked in-sim.
"""

import os
import sys

import carb
from isaacsim import SimulationApp

HEADLESS = os.getenv("PEGASUS_HEADLESS", "0") == "1"
RUN_DURATION_S = float(os.getenv("PEGASUS_RUN_DURATION_S", "0"))
TELEMETRY_INTERVAL_S = float(os.getenv("PEGASUS_CAD_TELEMETRY_INTERVAL_S", "0.5"))
simulation_app = SimulationApp({"headless": HEADLESS})

import numpy as np
import omni.timeline
from omni.isaac.core.world import World
from scipy.spatial.transform import Rotation

import isaacsim.core.utils.prims as prim_utils

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import SIMULATION_ENVIRONMENTS

EXAMPLES_DIR = os.path.dirname(os.path.realpath(__file__))
UTILS_DIR = os.path.join(EXAMPLES_DIR, "utils")
sys.path.insert(0, UTILS_DIR)

from cad_multirotor_usd import bake_cad_multirotor_usd
from nonlinear_controller import NonlinearController

MODELS_DIR = os.path.join(EXAMPLES_DIR, "models")
STEP_FILE = os.path.join(MODELS_DIR, "dron_do_symulacji.stp")
CAD_USD = os.path.join(MODELS_DIR, "dron_do_symulacji.usd")
VEHICLE_USD = os.path.join(MODELS_DIR, "dron_do_symulacji_multirotor.usd")

BODY_MASS_KG = 1.0
SPAWN_POSITION = np.array([0.0, 0.0, 0.5])
DRONE_PRIM_PATH = "/World/CadDrone"


class HoverController(NonlinearController):
    """Hold a fixed world position / yaw so the CAD articulation can be checked."""

    def __init__(self, hover_position, mass_kg: float = BODY_MASS_KG, **kwargs):
        super().__init__(**kwargs)
        self._hover = np.asarray(hover_position, dtype=float)
        self.m = float(mass_kg)

    def pd(self, t, s=0.6, reverse=False):
        return self._hover.copy()

    def d_pd(self, t, s=0.6, reverse=False):
        return np.zeros(3)

    def dd_pd(self, t, s=0.6, reverse=False):
        return np.zeros(3)

    def ddd_pd(self, t, s=0.6, reverse=False):
        return np.zeros(3)

    def yaw_d(self, t, s=0.6):
        return 0.0

    def d_yaw_d(self, t, s=0.6):
        return 0.0


def setup_lighting():
    prim_utils.create_prim(
        "/World/Light/DomeLight",
        "DomeLight",
        position=np.array([0.0, 0.0, 10.0]),
        attributes={"inputs:intensity": 500.0, "inputs:color": (1.0, 1.0, 1.0)},
    )
    prim_utils.create_prim(
        "/World/Light/Sun",
        "DistantLight",
        orientation=Rotation.from_euler("XYZ", [-45.0, 0.0, 45.0], degrees=True).as_quat(),
        attributes={"inputs:intensity": 1000.0, "inputs:color": (1.0, 0.98, 0.95)},
    )


def dump_rig(prim_path: str) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics
    from omni.isaac.core.utils.stage import get_current_stage

    stage = get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        carb.log_error(f"[CAD drone] missing prim {prim_path}")
        return

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
    size = rng.GetSize()
    carb.log_warn(
        f"[CAD drone] world bbox min={rng.GetMin()} max={rng.GetMax()} "
        f"size=({abs(size[0]):.4f}, {abs(size[1]):.4f}, {abs(size[2]):.4f}) m"
    )

    n_rb = n_joint = 0
    for prim in Usd.PrimRange(root):
        extra = [prim.GetTypeName()]
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            n_rb += 1
            extra.append("RigidBody")
        if prim.HasAPI(UsdPhysics.MassAPI):
            extra.append("MassAPI")
            mass = prim.GetAttribute("physics:mass")
            com = prim.GetAttribute("physics:centerOfMass")
            if mass and mass.HasAuthoredValue():
                extra.append(f"mass={mass.Get()}")
            if com and com.HasAuthoredValue():
                extra.append(f"com={com.Get()}")
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            extra.append("ArticulationRoot")
        if "Joint" in prim.GetTypeName():
            n_joint += 1
        name = prim.GetName()
        if (
            name in ("body", "vehicle")
            or name.startswith("rotor")
            or name.startswith("joint")
            or "RigidBody" in extra
            or "Joint" in extra[0]
        ):
            carb.log_warn(f"[CAD drone] {prim.GetPath()} [{', '.join(extra)}]")
    carb.log_warn(f"[CAD drone] rigidBodies={n_rb} joints={n_joint}")


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Box Room"])
        setup_lighting()

        self.controller = None
        self.vehicle = None

        if not os.path.exists(CAD_USD):
            carb.log_error(
                "[CAD drone] Converted CAD USD not found at "
                f"{CAD_USD}. Import {STEP_FILE} in the Isaac Sim GUI first."
            )
        else:
            info = bake_cad_multirotor_usd(CAD_USD, VEHICLE_USD, mass_kg=BODY_MASS_KG)
            carb.log_warn(
                "[CAD drone] baked vehicle USD: "
                f"body_size_m={np.round(info['body_size_m'], 4).tolist()} "
                f"rotors={np.round(info['rotor_positions_m'], 4).tolist()} "
                f"mass={info['mass_kg']} kg COM=(0,0,0) "
                f"I={tuple(round(v, 6) for v in info['inertia'])}"
            )

            config = MultirotorConfig()
            self.controller = HoverController(
                SPAWN_POSITION,
                mass_kg=BODY_MASS_KG,
                Ki=[0.5, 0.5, 0.5],
                Kr=[2.0, 2.0, 2.0],
            )
            config.backends = [self.controller]
            self.vehicle = Multirotor(
                DRONE_PRIM_PATH,
                VEHICLE_USD,
                0,
                SPAWN_POSITION.tolist(),
                Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
                config=config,
            )

        self.pg.set_viewport_camera(
            [1.2, -1.2, 1.1],
            SPAWN_POSITION.tolist(),
        )
        self.world.reset()
        if os.path.exists(VEHICLE_USD):
            dump_rig(DRONE_PRIM_PATH)

        self.stop_sim = False
        self._sim_elapsed_s = 0.0
        self._last_telemetry_s = -1.0

    def _log_hover_telemetry(self) -> None:
        ctrl = self.controller
        if ctrl is None or not ctrl.reveived_first_state:
            return
        rpy = ctrl.R.as_euler("xyz", degrees=True)
        hover_err = float(np.linalg.norm(ctrl.p - SPAWN_POSITION))
        carb.log_warn(
            f"[CAD drone] t={self._sim_elapsed_s:.2f}s "
            f"p={np.round(ctrl.p, 4).tolist()} "
            f"v={np.round(ctrl.v, 3).tolist()} "
            f"rpy_deg={np.round(rpy, 1).tolist()} "
            f"hover_err={hover_err:.3f}m "
            f"rotors={np.round(np.asarray(ctrl.input_ref, dtype=float), 1).tolist()}"
        )

    def run(self):
        self.timeline.play()
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
            self._sim_elapsed_s += float(self.world.get_physics_dt())
            if (
                TELEMETRY_INTERVAL_S > 0.0
                and (self._sim_elapsed_s - self._last_telemetry_s) >= TELEMETRY_INTERVAL_S
            ):
                self._last_telemetry_s = self._sim_elapsed_s
                self._log_hover_telemetry()
            if RUN_DURATION_S > 0.0 and self._sim_elapsed_s >= RUN_DURATION_S:
                break
        self._log_hover_telemetry()
        dump_rig(DRONE_PRIM_PATH)
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    PegasusApp().run()


if __name__ == "__main__":
    main()
