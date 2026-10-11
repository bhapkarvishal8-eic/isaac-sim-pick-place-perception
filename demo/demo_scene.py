"""Scene construction for the final pick-and-place demo (runs inside Isaac Sim kit).

Why a new cell instead of poc_workspace.usd (see README.md for the full story):
  * The project's Franka+Robotiq articulation never ran stably under PD control
    (panda_joint1 divergence), so the old bridge teleported joints kinematically
    and faked grasps by re-parenting objects. Here the standard Franka asset with
    its own hand runs on real PhysX PD drives -- the same configuration NVIDIA's
    own pick-and-place example uses -- and grasps are real friction grasps.
  * The 10/15/20 cm boxes are wider than any parallel gripper opens (Franka hand
    8 cm, Robotiq 2F-85 8.5 cm). The demo boxes reuse the project's box.usd
    (same cardboard material) scaled to 5.5 cm so they can physically be grasped.
  * The old boxes sat 12-32 cm from the robot base and the shelf overlapped the
    base; objects are now in the Franka's comfortable top-down workspace.
  * fixtures/shelf.usd has its xformOp order reversed (scale applied after
    translate), which collapses its panels; the place station here is built
    from plain cubes.
"""

from __future__ import annotations

import math
import random

import numpy as np
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

FIXTURES = "/home/vishal/poc/stage1_perception_arm/phase2_sim_environment/fixtures"

BOX_SIDE = 0.055          # m, graspable by the Franka hand (opens to 0.08)
PICK_SLOTS = [            # nominal (x, y) of boxes in the pick area, robot base frame
    (0.44, -0.22),
    (0.56, -0.10),
    (0.43, 0.02),
    (0.57, 0.12),
]
PICK_JITTER = 0.025       # m, per-episode position randomization
YAW_JITTER_DEG = 30.0     # per-episode yaw randomization

# Place station: a grey platform the robot stacks delivered boxes onto.
STATION_CENTER = (-0.05, 0.52)
STATION_SIZE = (0.40, 0.20, 0.10)   # x, y, height
PLACE_SLOTS_X = [-0.18, -0.09, 0.0, 0.09]   # one slot per delivered box

PILL_CENTER = (0.33, -0.34)          # small pill tray, detected but not picked

# RealSense-style camera. Same 66 deg HFOV and similar oblique look-down angle as the
# training camera, aimed at the pick area.
CAM_RES = (1280, 720)
CAM_HFOV_DEG = 66.0
CAM_EYE = (0.50, -0.78, 0.62)
CAM_TARGET = (0.50, -0.05, 0.0)


def _material(stage, path, rgb, roughness=0.6, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _static_cube(stage, path, center, size, mat):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))   # translate first, then scale (correct order)
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    return cube


def look_at_quat(eye, target):
    """USD camera orientation (looks down -Z, +Y up) for eye->target, world up +Z."""
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    return m.GetInverse()   # camera-to-world


def build(seed: int | None = None):
    """Build lights, ground, station, boxes, pills and camera on the current stage.

    Robot is added separately by the caller (Franka helper class). Returns a dict
    with prim paths and ground-truth poses (ground truth is ONLY used for logging
    the perception error, never for control).
    """
    rng = random.Random(seed)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Physics scene
    if not stage.GetPrimAtPath("/World/PhysicsScene"):
        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)

    # Lighting
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(900.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    sun.CreateIntensityAttr(2500.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(45.0, 10.0, 30.0))

    # Ground / work table (top surface at z = 0)
    floor_mat = _material(stage, "/World/Looks/Floor", (0.78, 0.76, 0.74), roughness=0.8)
    _static_cube(stage, "/World/Table", (0.25, 0.0, -0.025), (2.0, 2.0, 0.05), floor_mat)

    # Place station
    station_mat = _material(stage, "/World/Looks/Station", (0.42, 0.44, 0.48), roughness=0.5, metallic=0.2)
    sx, sy = STATION_CENTER
    sw, sd, sh = STATION_SIZE
    _static_cube(stage, "/World/PlaceStation/Top", (sx, sy, sh / 2), (sw, sd, sh), station_mat)

    # Boxes: project's box.usd (cardboard), Small_10cm variant scaled to BOX_SIDE
    gt_boxes = []
    scale = BOX_SIDE / 0.10
    for i, (bx, by) in enumerate(PICK_SLOTS):
        x = bx + rng.uniform(-PICK_JITTER, PICK_JITTER)
        y = by + rng.uniform(-PICK_JITTER, PICK_JITTER)
        yaw = rng.uniform(-YAW_JITTER_DEG, YAW_JITTER_DEG)
        path = f"/World/Boxes/box_{i:02d}"
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(f"{FIXTURES}/box.usd")
        prim.GetVariantSets().GetVariantSet("Size").SetVariantSelection("Small_10cm")
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, BOX_SIDE / 2 + 0.001))
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(math.radians(yaw) / 2), 0, 0, math.sin(math.radians(yaw) / 2)))
        xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
        gt_boxes.append({"path": path, "xy": (x, y), "yaw_deg": yaw})

    # Pills: project's pill.usd, made kinematic (8 mm pills are not simulated -- they
    # tunnel through the ground in PhysX, see enable_ros2_bridge_demo.py notes)
    pill_colors = [(0.85, 0.15, 0.15), (0.15, 0.55, 0.85), (0.9, 0.85, 0.2), (0.2, 0.75, 0.35)]
    px0, py0 = PILL_CENTER
    for i in range(10):
        path = f"/World/Pills/pill_{i:02d}"
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(f"{FIXTURES}/pill.usd")
        UsdGeom.Xformable(prim).AddTranslateOp().Set(
            Gf.Vec3d(px0 + rng.uniform(-0.07, 0.07), py0 + rng.uniform(-0.05, 0.05), 0.0016))
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
        geom = UsdGeom.Cylinder(stage.GetPrimAtPath(path + "/geom"))
        geom.CreateDisplayColorAttr([Gf.Vec3f(*pill_colors[i % len(pill_colors)])])

    # Camera
    cam = UsdGeom.Camera.Define(stage, "/World/RealsenseCamera")
    h_ap = 20.955
    focal = h_ap / 2.0 / math.tan(math.radians(CAM_HFOV_DEG) / 2.0)
    cam.CreateHorizontalApertureAttr(h_ap)
    cam.CreateVerticalApertureAttr(h_ap * CAM_RES[1] / CAM_RES[0])
    cam.CreateFocalLengthAttr(focal)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20.0))
    cam_xf = UsdGeom.Xformable(cam)
    cam_xf.AddTransformOp().Set(look_at_quat(CAM_EYE, CAM_TARGET))

    # Presentation camera (for the recorded video only)
    pres = UsdGeom.Camera.Define(stage, "/World/PresentationCamera")
    pres.CreateHorizontalApertureAttr(h_ap)
    pres.CreateVerticalApertureAttr(h_ap * 9 / 16)
    pres.CreateFocalLengthAttr(15.0)
    pres.CreateClippingRangeAttr(Gf.Vec2f(0.05, 50.0))
    UsdGeom.Xformable(pres).AddTransformOp().Set(look_at_quat((1.55, -0.30, 0.95), (0.18, 0.12, 0.18)))

    return {"boxes": gt_boxes, "camera_path": "/World/RealsenseCamera",
            "presentation_camera_path": "/World/PresentationCamera"}


def camera_intrinsics():
    fx = CAM_RES[0] / 2.0 / math.tan(math.radians(CAM_HFOV_DEG) / 2.0)
    return {"fx": fx, "fy": fx, "cx": CAM_RES[0] / 2.0, "cy": CAM_RES[1] / 2.0,
            "width": CAM_RES[0], "height": CAM_RES[1]}


def camera_to_world(camera_path="/World/RealsenseCamera"):
    """4x4 numpy, row-vector convention (p_world = p_usd_cam @ M)."""
    from pxr import Usd
    stage = omni.usd.get_context().get_stage()
    m = UsdGeom.Xformable(stage.GetPrimAtPath(camera_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m)
