#!/usr/bin/env python
"""
| File: 17_python_cad_drone.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Spawns a *single* drone in the scene using a user-provided CAD model
| (examples/models/dron_do_symulacji.stp) as a purely visual asset.
|
| IMPORTANT - one-time manual conversion step (STEP -> USD):
|   Isaac Sim cannot load a raw ``.stp``/``.step`` file from a standalone Python app.
|   The CAD Converter extension (omni.kit.converter.cad) is disabled in the lightweight
|   ``SimulationApp`` used by these examples, so the conversion must be done once inside
|   the full Isaac Sim GUI:
|
|     1. Launch the Isaac Sim application (GUI, not this script).
|     2. Make sure the "CAD Converter" extension (omni.kit.converter.cad) is enabled
|        (Window > Extensions, search "CAD", enable + Autoload).
|     3. File > Import... and select:
|            examples/models/dron_do_symulacji.stp
|     4. Save / export the result next to the source as:
|            examples/models/dron_do_symulacji.usd
|        (or point MODEL_USD below at wherever you saved it).
|
|   Once that ``.usd`` exists, this example loads it and shows the drone in the Box Room.
|
| NOTE: This spawns the model as a *visual* prim only (it does not fly). The Pegasus
| ``Multirotor`` dynamics require a rigged USD (a ``/body`` rigid body plus ``/rotor0..3``
| bodies and ``joint0..3`` articulated joints), which a bare CAD shell does not have.
| Making this model flyable is a separate rigging step.
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import os

import numpy as np
import omni.timeline
from omni.isaac.core.world import World
from scipy.spatial.transform import Rotation

import isaacsim.core.utils.prims as prim_utils

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# ---------------------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

# Source CAD file (informational - it cannot be loaded directly, see the header docstring).
STEP_FILE = os.path.join(MODELS_DIR, "dron_do_symulacji.stp")
# The converted USD produced by the Isaac Sim GUI CAD Converter (see header docstring).
MODEL_USD = os.path.join(MODELS_DIR, "dron_do_symulacji.usd")

# Where to place the drone.
SPAWN_POSITION = np.array([0.0, 0.0, 1.0])          # metres, ENU (drone "hovering" at 1 m)
# Extra user rotation on top of the automatic up-axis correction (see below).
SPAWN_ORIENTATION = Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat()

# --- Sizing ---
# CAD/STEP files carry unit metadata that the converter may or may not bake correctly, so
# scaling by hand is error-prone. By default we AUTO-FIT the model to a sensible drone size:
# its largest dimension is scaled to this many metres. Set to None (or 0) to disable auto-fit
# and use the explicit MODEL_SCALE below instead.
AUTO_FIT_SIZE_M = 0.6
# Only used when AUTO_FIT_SIZE_M is disabled. This multiplies the automatic unit conversion
# (source metersPerUnit -> stage metres); set to 0.001 if a mm model still comes in 1000x big.
MODEL_SCALE = 1.0

DRONE_PRIM_PATH = "/World/CadDrone"


def _report_model_size(prim_path: str) -> None:
    """Print the world-space bounding box of the spawned model as a scale sanity check."""
    try:
        from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache

        cache = create_bbox_cache()
        aabb = compute_aabb(cache, prim_path, include_children=True)
        size = np.asarray(aabb[3:]) - np.asarray(aabb[:3])
        carb.log_warn(
            f"[CAD drone] bounding box size (m): "
            f"x={size[0]:.3f}, y={size[1]:.3f}, z={size[2]:.3f}"
        )
        largest = float(np.max(np.abs(size)))
        if largest > 50.0:
            carb.log_warn(
                "[CAD drone] Model looks very large - the STEP file is in mm; "
                "try setting MODEL_SCALE = 0.001."
            )
        elif 0.0 < largest < 0.02:
            carb.log_warn(
                "[CAD drone] Model looks very small - try setting MODEL_SCALE back to 1.0."
            )
    except Exception as exc:  # pragma: no cover (depends on Isaac Sim runtime)
        carb.log_warn(f"[CAD drone] Could not compute model bounding box: {exc}")


class PegasusApp:
    """A minimal standalone app that loads a CAD-derived drone model into the scene."""

    def __init__(self):
        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World (physics, asset spawning, etc.)
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Box Room"])

        # Add some lighting so the imported model is clearly visible
        self._setup_lighting()

        # Spawn the CAD drone model (visual only)
        self._spawn_cad_drone()

        # Reset the simulation environment so that everything is initialized
        self.world.reset()

        # Report the model size once the stage has composed (scale sanity check)
        if os.path.exists(MODEL_USD):
            _report_model_size(DRONE_PRIM_PATH)

        self.stop_sim = False

    def _setup_lighting(self):
        """A dome + sun light so the model is evenly lit regardless of the environment."""
        prim_utils.create_prim(
            "/World/Light/DomeLight",
            "DomeLight",
            position=np.array([0.0, 0.0, 10.0]),
            attributes={"inputs:intensity": 2000.0, "inputs:color": (1.0, 1.0, 1.0)},
        )
        prim_utils.create_prim(
            "/World/Light/Sun",
            "DistantLight",
            orientation=Rotation.from_euler("XYZ", [-45.0, 0.0, 45.0], degrees=True).as_quat(),
            attributes={"inputs:intensity": 3000.0, "inputs:color": (1.0, 0.98, 0.95)},
        )

    @staticmethod
    def _source_bbox_size(model_stage, prim_path):
        """Largest-dimension bounding box of the model, measured in its OWN coordinate units."""
        try:
            from pxr import Usd, UsdGeom

            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
            )
            prim = model_stage.GetPrimAtPath(prim_path) if prim_path else model_stage.GetPseudoRoot()
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            s = rng.GetSize()
            return (abs(s[0]), abs(s[1]), abs(s[2]))
        except Exception as exc:  # pragma: no cover (Isaac Sim runtime)
            carb.log_warn(f"[CAD drone] could not measure source bbox: {exc}")
            return None

    def _spawn_cad_drone(self):
        """Reference the converted USD model into the stage as a single visual prim.

        Uses explicit prim references (NOT the layer defaultPrim, which the CAD converter
        often leaves unset) plus auto unit-fitting and up-axis correction, so the model is
        reliably visible regardless of how the converter authored it.
        """
        if not os.path.exists(MODEL_USD):
            carb.log_error(
                "[CAD drone] Converted model not found at:\n"
                f"    {MODEL_USD}\n"
                "The raw STEP file cannot be loaded from a standalone script. Convert it once "
                "in the full Isaac Sim GUI (File > Import) and save it to the path above.\n"
                f"    Source STEP: {STEP_FILE}\n"
                "Launching the empty Box Room for now."
            )
            return

        try:
            from pxr import Gf, Usd, UsdGeom
            from omni.isaac.core.utils.stage import get_current_stage

            stage = get_current_stage()
            stage_mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
            stage_up = UsdGeom.GetStageUpAxis(stage)

            # --- Inspect the source layer so we don't depend on its defaultPrim ---
            model_stage = Usd.Stage.Open(MODEL_USD)
            model_mpu = UsdGeom.GetStageMetersPerUnit(model_stage) or 1.0
            model_up = UsdGeom.GetStageUpAxis(model_stage)
            model_default = model_stage.GetDefaultPrim()
            has_default = bool(model_default and model_default.IsValid())
            tops = list(model_stage.GetPseudoRoot().GetChildren())

            if has_default:
                ref_targets = [model_default.GetPath()]
            else:
                ref_targets = [p.GetPath() for p in tops]

            carb.log_warn(
                f"[CAD drone] source: defaultPrim={model_default.GetPath() if has_default else '<unset>'}, "
                f"top-level prims={len(tops)}, metersPerUnit={model_mpu}, upAxis={model_up}"
            )

            # --- Compute the scale (auto-fit is robust to any unit weirdness) ---
            if AUTO_FIT_SIZE_M:
                src_size = self._source_bbox_size(model_stage, None)
                largest = max(src_size) if src_size else 0.0
                if largest > 1e-9:
                    total_scale = float(AUTO_FIT_SIZE_M) / float(largest)
                    carb.log_warn(
                        f"[CAD drone] auto-fit: source largest dim={largest:.4g} (model units) "
                        f"-> {AUTO_FIT_SIZE_M} m (scale={total_scale:.6g})"
                    )
                else:
                    total_scale = (model_mpu / stage_mpu) * MODEL_SCALE
                    carb.log_warn("[CAD drone] auto-fit failed to measure model; using unit scale.")
            else:
                total_scale = (model_mpu / stage_mpu) * MODEL_SCALE
                carb.log_warn(
                    f"[CAD drone] unit scale={model_mpu / stage_mpu:.6g} * MODEL_SCALE={MODEL_SCALE} "
                    f"-> {total_scale:.6g}"
                )

            # --- Up-axis correction composed with the user orientation ---
            rotation = Rotation.from_quat(SPAWN_ORIENTATION)
            if model_up == "Y" and stage_up == "Z":
                rotation = rotation * Rotation.from_euler("x", 90.0, degrees=True)
            elif model_up == "Z" and stage_up == "Y":
                rotation = rotation * Rotation.from_euler("x", -90.0, degrees=True)
            q = rotation.as_quat()  # (x, y, z, w)

            # --- Reference the concrete prim(s); do NOT rely on defaultPrim ---
            drone_prim = stage.DefinePrim(DRONE_PRIM_PATH, "Xform")
            if len(ref_targets) == 1:
                drone_prim.GetReferences().AddReference(MODEL_USD, ref_targets[0])
            elif len(ref_targets) > 1:
                # Multiple roots with no defaultPrim: bring each in under a child prim.
                for i, target in enumerate(ref_targets):
                    child = stage.DefinePrim(f"{DRONE_PRIM_PATH}/part_{i}", "Xform")
                    child.GetReferences().AddReference(MODEL_USD, target)
            else:
                drone_prim.GetReferences().AddReference(MODEL_USD)  # last-resort defaultPrim

            xform = UsdGeom.Xformable(drone_prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in SPAWN_POSITION]))
            xform.AddOrientOp().Set(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
            xform.AddScaleOp().Set(Gf.Vec3f(total_scale, total_scale, total_scale))

            carb.log_warn(
                f"[CAD drone] Referenced {len(ref_targets) or 1} prim(s) from {MODEL_USD} "
                f"at {DRONE_PRIM_PATH}."
            )
        except Exception as exc:  # pragma: no cover (Isaac Sim runtime)
            carb.log_error(f"[CAD drone] Failed to spawn model: {exc}")

    def run(self):
        """Application main loop - just render the scene with the model in it."""
        self.timeline.play()

        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)

        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()
