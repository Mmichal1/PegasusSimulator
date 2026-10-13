#!/usr/bin/env python
"""Build a Pegasus-compatible quadrotor USD from the CAD drone mesh.

The CAD STEP/USD is visual-only. Pegasus Multirotor applies thrust to ``/rotor0..3``
and torque to ``/body``, and spins DOFs named ``joint0..3``. This baker:

  1. Measures the CAD assembly (main body + four arms) in the source layer.
  2. Places the mesh under ``/vehicle/body``, scaled mm->m and recentered so the
     main-body centroid is the origin (and therefore the center of mass).
  3. Adds four light rotor rigid bodies at the arm locations with revolute joints.

Mass is 1.0 kg on the body. Rotor masses are negligible (1 g each) so the vehicle
mass stays at 1 kg for practical purposes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

BODY_MASS_KG = 1.0
ROTOR_MASS_KG = 0.001
SOURCE_TO_METERS = 0.001  # CAD coordinates are millimetres

# Motor / rotor frames in the body frame [x, y, z] metres. COM is (0, 0, 0), Z up.
# Pegasus X-quad order:
#   rotor0  (+x, -y)    rotor1  (-x, +y)
#   rotor2  (+x, +y)    rotor3  (-x, -y)
#
# Set to None to auto-place from the CAD arm geometry, then apply the
# scale/offset below. Set to a 4x3 list to pin exact positions.
ROTOR_POSITIONS_M = [
    [ 0.07875, -0.07875, -0.04],  # rotor0  +x, -y
    [-0.07875,  0.07875, -0.04],  # rotor1  -x, +y
    [ 0.07875,  0.07875, -0.04],  # rotor2  +x, +y
    [-0.07875, -0.07875, -0.04],  # rotor3  -x, -y
]

# Applied after auto-placement (and also after an explicit list).
# ROTOR_XY_SCALE=1.2 moves all motors 20% further from the COM in XY.
# ROTOR_POSITION_OFFSET_M is added to every motor; +z is up.
ROTOR_XY_SCALE = 1.0
ROTOR_POSITION_OFFSET_M = [0.0, 0.0, 0.0]


def cad_to_enu(p: np.ndarray) -> np.ndarray:
    """Inventor CAD is Y-up; Isaac/Pegasus is Z-up. Rx(+90): (x,y,z)->(x,-z,y)."""
    p = np.asarray(p, dtype=float)
    return np.array([p[0], -p[2], p[1]])


@dataclass
class BBox:
    min: np.ndarray
    max: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.min + self.max)

    @property
    def size(self) -> np.ndarray:
        return self.max - self.min


def _bbox_of(stage, prim) -> BBox | None:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    return BBox(
        min=np.array(rng.GetMin(), dtype=float),
        max=np.array(rng.GetMax(), dtype=float),
    )


def _find_named(stage, needle: str) -> list:
    from pxr import Usd

    hits = []
    for prim in stage.Traverse():
        if needle.lower() in prim.GetName().lower():
            hits.append(prim)
    return hits


def _sort_rotors_xy(centers: list[np.ndarray]) -> list[np.ndarray]:
    """Order as Pegasus X-quad: 0 (+x,-y), 1 (-x,+y), 2 (+x,+y), 3 (-x,-y)."""
    if len(centers) != 4:
        return centers
    labeled = []
    for c in centers:
        labeled.append((int(c[0] >= 0), int(c[1] >= 0), c))
    # (sx, sy) -> rotor index
    order = {(1, 0): 0, (0, 1): 1, (1, 1): 2, (0, 0): 3}
    out: list[np.ndarray | None] = [None, None, None, None]
    for sx, sy, c in labeled:
        out[order[(sx, sy)]] = c
    if any(p is None for p in out):
        # Fall back to angle around Z if signs are degenerate.
        angles = [np.arctan2(c[1], c[0]) for c in centers]
        ranked = [c for _, c in sorted(zip(angles, centers), key=lambda t: t[0])]
        return ranked
    return out  # type: ignore[return-value]


def inspect_cad_geometry(cad_usd: str) -> dict:
    """Measure main-body and arm locations in metres, relative to the body center."""
    from pxr import Usd

    stage = Usd.Stage.Open(cad_usd)
    if stage is None:
        raise FileNotFoundError(cad_usd)

    default = stage.GetDefaultPrim()
    root = default if default and default.IsValid() else stage.GetPseudoRoot()
    root_bbox = _bbox_of(stage, root)
    if root_bbox is None:
        raise RuntimeError("CAD USD has an empty bounding box")

    body_prims = _find_named(stage, "baza")
    body_bbox = _bbox_of(stage, body_prims[0]) if body_prims else root_bbox
    if body_bbox is None:
        body_bbox = root_bbox

    arm_prims = _find_named(stage, "filar")
    if len(arm_prims) != 4:
        arm_prims = _find_named(stage, "podstawka")
        arm_prims = [p for p in arm_prims if "kamera" not in p.GetName().lower()]

    body_center_m = cad_to_enu(body_bbox.center * SOURCE_TO_METERS)
    body_size_m = np.abs(cad_to_enu(body_bbox.size * SOURCE_TO_METERS))
    root_size_m = np.abs(cad_to_enu(root_bbox.size * SOURCE_TO_METERS))

    candidates = []
    for prim in list(arm_prims) + _find_named(stage, "podstawka"):
        if "kamera" in prim.GetName().lower():
            continue
        bb = _bbox_of(stage, prim)
        if bb is None:
            continue
        rel = cad_to_enu(bb.center * SOURCE_TO_METERS) - body_center_m
        candidates.append(rel)

    rotor_m = []
    if candidates:
        uniq = []
        for c in candidates:
            if not any(np.linalg.norm(c - u) < 1e-4 for u in uniq):
                uniq.append(c)
        uniq.sort(key=lambda c: -float(np.hypot(c[0], c[1])))
        rotor_m = uniq[:4]
        if len(rotor_m) == 4:
            rotor_m = _sort_rotors_xy(rotor_m)
        xy_span = max((float(np.hypot(c[0], c[1])) for c in rotor_m), default=0.0)
        if xy_span < 0.08:
            hx, hy = 0.35 * root_size_m[0], 0.35 * root_size_m[1]
            hz = 0.5 * body_size_m[2]
            rotor_m = _sort_rotors_xy(
                [
                    np.array([hx, -hy, hz]),
                    np.array([-hx, hy, hz]),
                    np.array([hx, hy, hz]),
                    np.array([-hx, -hy, hz]),
                ]
            )

    default_path = str(default.GetPath()) if default and default.IsValid() else ""
    if ROTOR_POSITIONS_M is not None:
        rotor_m = [np.asarray(p, dtype=float) for p in ROTOR_POSITIONS_M]
    rotor_m = _apply_rotor_adjustments(rotor_m)
    return {
        "default_prim": default_path,
        "body_center_m": body_center_m,
        "body_size_m": body_size_m,
        "root_size_m": root_size_m,
        "rotor_positions_m": rotor_m,
        "arm_count": len(rotor_m),
    }


def _apply_rotor_adjustments(rotor_m: list[np.ndarray]) -> list[np.ndarray]:
    """Scale XY about the COM and add a uniform offset. Edit the constants above."""
    scale = float(ROTOR_XY_SCALE)
    offset = np.asarray(ROTOR_POSITION_OFFSET_M, dtype=float)
    out = []
    for p in rotor_m:
        q = np.asarray(p, dtype=float).copy()
        q[0] *= scale
        q[1] *= scale
        q += offset
        out.append(q)
    return out


def bake_cad_multirotor_usd(
    cad_usd: str,
    output_usd: str,
    mass_kg: float = BODY_MASS_KG,
) -> dict:
    """Write ``output_usd`` with /vehicle/{body, rotor0..3, joint0..3}."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    info = inspect_cad_geometry(cad_usd)
    if len(info["rotor_positions_m"]) != 4:
        raise RuntimeError(
            f"Expected 4 arm/rotor locations, found {info['arm_count']}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_usd)), exist_ok=True)
    if os.path.exists(output_usd):
        os.remove(output_usd)

    stage = Usd.Stage.CreateNew(output_usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    vehicle = UsdGeom.Xform.Define(stage, "/vehicle")
    stage.SetDefaultPrim(vehicle.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(vehicle.GetPrim())

    # ---- body: 1 kg, COM at origin = main-body centroid ----
    body = UsdGeom.Xform.Define(stage, "/vehicle/body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass_api.CreateMassAttr(float(mass_kg))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    sx, sy, sz = [float(v) for v in info["body_size_m"]]
    # Box inertia about the body centroid (COM).
    ixx = mass_kg / 12.0 * (sy * sy + sz * sz)
    iyy = mass_kg / 12.0 * (sx * sx + sz * sz)
    izz = mass_kg / 12.0 * (sx * sx + sy * sy)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(ixx, iyy, izz))

    visual = UsdGeom.Xform.Define(stage, "/vehicle/body/visual")
    xref = info["default_prim"] or "/"
    visual.GetPrim().GetReferences().AddReference(cad_usd, Sdf.Path(xref))
    visual.ClearXformOpOrder()
    # Scale mm->m, rotate Y-up CAD into Z-up, then shift the main-body centroid to origin.
    visual.AddTranslateOp().Set(
        Gf.Vec3d(*[-float(v) for v in info["body_center_m"]])
    )
    visual.AddOrientOp().Set(Gf.Quatf(0.70710678, 0.70710678, 0.0, 0.0))  # Rx +90 deg
    visual.AddScaleOp().Set(Gf.Vec3f(SOURCE_TO_METERS, SOURCE_TO_METERS, SOURCE_TO_METERS))

    collision = UsdGeom.Cube.Define(stage, "/vehicle/body/collision")
    collision.CreateSizeAttr(1.0)
    collision.ClearXformOpOrder()
    collision.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    collision.AddScaleOp().Set(Gf.Vec3d(max(sx, 0.02), max(sy, 0.02), max(sz, 0.02)))
    collision.CreateVisibilityAttr().Set("invisible")
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())

    identity = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    for i, pos in enumerate(info["rotor_positions_m"]):
        px, py, pz = [float(v) for v in pos]
        rotor_path = f"/vehicle/rotor{i}"
        rotor = UsdGeom.Xform.Define(stage, rotor_path)
        UsdPhysics.RigidBodyAPI.Apply(rotor.GetPrim())
        rmass = UsdPhysics.MassAPI.Apply(rotor.GetPrim())
        rmass.CreateMassAttr(ROTOR_MASS_KG)
        rmass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        rotor.ClearXformOpOrder()
        rotor.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))
        rotor.AddOrientOp().Set(identity)

        marker = UsdGeom.Cylinder.Define(stage, f"{rotor_path}/disk")
        marker.CreateRadiusAttr(0.03)
        marker.CreateHeightAttr(0.004)
        marker.CreateAxisAttr("Z")
        UsdPhysics.CollisionAPI.Apply(marker.GetPrim()).CreateCollisionEnabledAttr(False)

        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{rotor_path}/joint{i}")
        joint.CreateAxisAttr("Z")
        joint.CreateBody0Rel().SetTargets(["/vehicle/body"])
        joint.CreateBody1Rel().SetTargets([rotor_path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(px, py, pz))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr(identity)
        joint.CreateLocalRot1Attr(identity)

    stage.GetRootLayer().Save()
    info["output_usd"] = output_usd
    info["mass_kg"] = mass_kg
    info["inertia"] = (ixx, iyy, izz)
    return info
