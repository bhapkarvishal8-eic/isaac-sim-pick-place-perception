"""Warehouse cell for the v2 demo (pills + boxes, vacuum tool). Runs inside Isaac Sim.

World frame = Franka base; the packing-table top is at z = 0, so the warehouse
floor sits at z = -TABLE_TOP. Everything the robot touches lies on the table.

Objects are the project's own Phase 3 assets at their trained scale:
  * fixtures/box.usd, Small_10cm variant (10 cm cardboard boxes)
  * fixtures/pill.usd (8 mm x 3 mm tablets, four colors)
"""

from __future__ import annotations

import math
import random

import numpy as np
import omni.usd
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

FIXTURES = "/home/vishal/poc/stage1_perception_arm/phase2_sim_environment/fixtures"

TABLE_TOP = 0.994            # packing_table.usd top surface height (measured)
TABLE_CENTER_X = 0.26        # table rotated 90 deg: 0.76 m deep along x, 2.47 m long along y

BOX_SIDE = 0.10
BOX_SLOTS = [(0.37, -0.09), (0.53, -0.05), (0.38, 0.11), (0.54, 0.15)]
BOX_JITTER = 0.012
BOX_YAW_DEG = 25.0

PILL_MAT_CENTER = (0.43, -0.30)
PILL_MAT_SIZE = (0.20, 0.14)
N_PILLS = 5
PILL_MIN_SPACING = 0.035
PILL_RADIUS, PILL_HEIGHT = 0.004, 0.003

TOTE_CENTER = (0.24, 0.56)        # table's own grey tote (container_h20), boxes go here
TOTE_SLOTS = [(0.10, 0.50), (0.26, 0.50), (0.10, 0.63), (0.26, 0.63)]
BIN_CENTER = (0.16, -0.52)        # small KLT bin, pills go here

CAM_RES = (1280, 720)
CAM_HFOV_DEG = 66.0
CAM_EYE = (0.42, -0.74, 0.58)
CAM_TARGET = (0.40, -0.10, 0.0)
PRES_EYE = (1.55, -0.95, 1.05)
PRES_TARGET = (0.22, 0.02, 0.12)
PILL_CAM_EYE = (1.00, -1.02, 0.62)     # close-up used while pills are picked (pills are 8 mm)
PILL_CAM_TARGET = (0.30, -0.40, 0.10)


def look_at_quat(eye, target):
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    return m.GetInverse()


def _material(stage, path, rgb, roughness=0.6, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _bbox(prim):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def _place_on_table(prim, cx, cy, yaw_deg=0.0):
    """Move a referenced prop so its bbox is centered at (cx, cy) and rests on z=0."""
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    t = xf.AddTranslateOp(opSuffix="cell")
    t.Set(Gf.Vec3d(0, 0, 0))
    xf.AddRotateZOp(opSuffix="cell").Set(yaw_deg)
    r = _bbox(prim)
    c = (r.GetMin() + r.GetMax()) / 2
    t.Set(Gf.Vec3d(cx - c[0], cy - c[1], -r.GetMin()[2]))


def _sample_pills(rng):
    pts = []
    cx, cy = PILL_MAT_CENTER
    w, d = PILL_MAT_SIZE
    while len(pts) < N_PILLS:
        p = (cx + rng.uniform(-w / 2 + 0.02, w / 2 - 0.02), cy + rng.uniform(-d / 2 + 0.02, d / 2 - 0.02))
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) > PILL_MIN_SPACING for q in pts):
            pts.append(p)
    return pts


def build(seed: int | None = None):
    rng = random.Random(seed)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    R = get_assets_root_path()

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateEnableCCDAttr(True)

    # --- warehouse room + packing table -------------------------------------------------
    wh = stage.DefinePrim("/World/Warehouse", "Xform")
    wh.GetReferences().AddReference(R + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd")
    wxf = UsdGeom.Xformable(wh)
    wxf.ClearXformOpOrder()
    wxf.AddTranslateOp(opSuffix="cell").Set(Gf.Vec3d(0.0, 0.0, -TABLE_TOP))

    table = stage.DefinePrim("/World/PackingTable", "Xform")
    table.GetReferences().AddReference(R + "/Isaac/Props/PackingTable/packing_table.usd")
    txf = UsdGeom.Xformable(table)
    txf.ClearXformOpOrder()
    txf.AddTranslateOp(opSuffix="cell").Set(Gf.Vec3d(TABLE_CENTER_X, 0.0, -TABLE_TOP))
    txf.AddRotateZOp(opSuffix="cell").Set(90.0)
    # the table's tote is re-placed as the box destination below
    tote_in_table = stage.GetPrimAtPath("/World/PackingTable/container_h20")
    if tote_in_table.IsValid():
        tote_in_table.SetActive(False)
    # a collider for the table top (the prop mesh is visual-first)
    top = UsdGeom.Cube.Define(stage, "/World/TableTopCollider")
    top.CreateSizeAttr(1.0)
    top.AddTranslateOp().Set(Gf.Vec3d(TABLE_CENTER_X, 0.0, -0.02))
    top.AddScaleOp().Set(Gf.Vec3f(0.76, 2.46, 0.04))
    top.CreatePurposeAttr(UsdGeom.Tokens.guide)       # invisible, collision only
    UsdPhysics.CollisionAPI.Apply(top.GetPrim())

    # --- lighting over the work cell (the warehouse has its own ambient lights) ---------
    key = UsdLux.RectLight.Define(stage, "/World/CellLights/Overhead")
    key.CreateIntensityAttr(2500.0)
    key.CreateWidthAttr(1.2)
    key.CreateHeightAttr(0.6)
    kxf = UsdGeom.Xformable(key)
    kxf.AddTranslateOp().Set(Gf.Vec3d(0.30, -0.05, 1.45))   # RectLight emits along -Z

    # --- destinations --------------------------------------------------------------------
    tote = stage.DefinePrim("/World/Tote", "Xform")
    tote.GetReferences().AddReference(R + "/Isaac/Props/PackingTable/packing_table.usd", "/Root/container_h20")
    _place_on_table(tote, *TOTE_CENTER, yaw_deg=0.0)
    tote_r = _bbox(tote)

    kbin = stage.DefinePrim("/World/PillBin", "Xform")
    kbin.GetReferences().AddReference(R + "/Isaac/Props/KLT_Bin/small_KLT.usd")
    _place_on_table(kbin, *BIN_CENTER, yaw_deg=0.0)
    for p in Usd.PrimRange(kbin):   # the bin is a fixed fixture, not a dynamic body
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
    bin_r = _bbox(kbin)

    # pill mat (light grey, like the Phase 3 training floor)
    import os
    mat_rgb = tuple(float(v) for v in os.environ.get("PILL_MAT_RGB", "0.62,0.63,0.62").split(","))
    mat_mat = _material(stage, "/World/Looks/PillMat", mat_rgb, roughness=0.9)
    mat = UsdGeom.Cube.Define(stage, "/World/PillMat")
    mat.CreateSizeAttr(1.0)
    mat.AddTranslateOp().Set(Gf.Vec3d(PILL_MAT_CENTER[0], PILL_MAT_CENTER[1], 0.001))
    mat.AddScaleOp().Set(Gf.Vec3f(PILL_MAT_SIZE[0], PILL_MAT_SIZE[1], 0.002))
    UsdPhysics.CollisionAPI.Apply(mat.GetPrim())
    _bind(mat.GetPrim(), mat_mat)

    # --- boxes (project box.usd, 10 cm) --------------------------------------------------
    gt_boxes = []
    for i, (bx, by) in enumerate(BOX_SLOTS):
        x = bx + rng.uniform(-BOX_JITTER, BOX_JITTER)
        y = by + rng.uniform(-BOX_JITTER, BOX_JITTER)
        yaw = rng.uniform(-BOX_YAW_DEG, BOX_YAW_DEG)
        path = f"/World/Boxes/box_{i:02d}"
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(f"{FIXTURES}/box.usd")
        prim.GetVariantSets().GetVariantSet("Size").SetVariantSelection("Small_10cm")
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, BOX_SIDE / 2 + 0.001))
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(math.radians(yaw) / 2), 0, 0, math.sin(math.radians(yaw) / 2)))
        gt_boxes.append({"path": path, "xy": (x, y), "yaw_deg": yaw})

    # --- pills (project pill.usd), dynamic with CCD so they don't tunnel -----------------
    # no yellow: pale yellow on the light mat is too low-contrast for the detector once
    # its neighbours are picked (seed 11: score fell from 0.49 to <0.05)
    colors = [(0.85, 0.15, 0.15), (0.15, 0.55, 0.85), (0.55, 0.2, 0.75), (0.2, 0.75, 0.35)]
    gt_pills = []
    for i, (px, py) in enumerate(_sample_pills(rng)):
        path = f"/World/Pills/pill_{i:02d}"
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(f"{FIXTURES}/pill.usd")
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(px, py, 0.002 + PILL_HEIGHT / 2 + 0.0005))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateEnableCCDAttr(True)
        rb.CreateSolverPositionIterationCountAttr(16)
        geom = stage.GetPrimAtPath(path + "/geom")
        UsdGeom.Cylinder(geom).CreateDisplayColorAttr([Gf.Vec3f(*colors[i % len(colors)])])
        col = PhysxSchema.PhysxCollisionAPI.Apply(geom)
        col.CreateContactOffsetAttr(0.0015)
        col.CreateRestOffsetAttr(0.0)
        gt_pills.append({"path": path, "xy": (px, py)})

    # --- cameras -------------------------------------------------------------------------
    h_ap = 20.955
    cam = UsdGeom.Camera.Define(stage, "/World/RealsenseCamera")
    cam.CreateHorizontalApertureAttr(h_ap)
    cam.CreateVerticalApertureAttr(h_ap * CAM_RES[1] / CAM_RES[0])
    cam.CreateFocalLengthAttr(h_ap / 2.0 / math.tan(math.radians(CAM_HFOV_DEG) / 2.0))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 60.0))
    UsdGeom.Xformable(cam).AddTransformOp().Set(look_at_quat(CAM_EYE, CAM_TARGET))

    pres = UsdGeom.Camera.Define(stage, "/World/PresentationCamera")
    pres.CreateHorizontalApertureAttr(h_ap)
    pres.CreateVerticalApertureAttr(h_ap * 9 / 16)
    pres.CreateFocalLengthAttr(16.0)
    pres.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
    UsdGeom.Xformable(pres).AddTransformOp().Set(look_at_quat(PRES_EYE, PRES_TARGET))

    close = UsdGeom.Camera.Define(stage, "/World/PillCloseupCamera")
    close.CreateHorizontalApertureAttr(h_ap)
    close.CreateVerticalApertureAttr(h_ap * 9 / 16)
    close.CreateFocalLengthAttr(18.0)
    close.CreateClippingRangeAttr(Gf.Vec2f(0.02, 200.0))
    UsdGeom.Xformable(close).AddTransformOp().Set(look_at_quat(PILL_CAM_EYE, PILL_CAM_TARGET))

    return {"boxes": gt_boxes, "pills": gt_pills,
            "camera_path": "/World/RealsenseCamera", "presentation_camera_path": "/World/PresentationCamera",
            "pill_camera_path": "/World/PillCloseupCamera",
            "tote_range": ((tote_r.GetMin()[0], tote_r.GetMin()[1], tote_r.GetMin()[2]),
                           (tote_r.GetMax()[0], tote_r.GetMax()[1], tote_r.GetMax()[2])),
            "bin_range": ((bin_r.GetMin()[0], bin_r.GetMin()[1], bin_r.GetMin()[2]),
                          (bin_r.GetMax()[0], bin_r.GetMax()[1], bin_r.GetMax()[2]))}


# ------------------------------------------------------------------ vacuum tool
TOOL_TIP = 0.135      # m along panda_hand +Z from the hand origin to the suction-cup face


def add_vacuum_tool(robot_path="/World/robot"):
    """Visual vacuum tool under panda_hand + an Isaac Sim Surface Gripper at the cup face."""
    from usd.schema.isaac import robot_schema
    stage = omni.usd.get_context().get_stage()
    hand = f"{robot_path}/panda_hand"
    steel = _material(stage, "/World/Looks/ToolSteel", (0.55, 0.57, 0.60), roughness=0.35, metallic=0.8)
    rubber = _material(stage, "/World/Looks/CupRubber", (0.05, 0.05, 0.06), roughness=0.9)
    accent = _material(stage, "/World/Looks/ToolAccent", (0.95, 0.45, 0.05), roughness=0.5)

    def cyl(name, radius, z0, z1, mat):
        c = UsdGeom.Cylinder.Define(stage, f"{hand}/VacuumTool/{name}")
        c.CreateRadiusAttr(radius)
        c.CreateHeightAttr(z1 - z0)
        c.CreateAxisAttr("Z")
        c.AddTranslateOp().Set(Gf.Vec3d(0, 0, (z0 + z1) / 2))
        _bind(c.GetPrim(), mat)

    cyl("Flange", 0.035, 0.000, 0.012, steel)
    cyl("Body", 0.016, 0.012, 0.085, accent)
    cyl("Stem", 0.006, 0.085, 0.118, steel)
    cyl("Bellows", 0.009, 0.118, 0.128, rubber)
    cyl("Cup", 0.011, 0.128, TOOL_TIP, rubber)

    # attachment point: a D6 joint from panda_hand at the cup face, forward axis +Z
    jp = f"{hand}/VacuumTool/AttachmentPoint"
    joint = UsdPhysics.Joint.Define(stage, jp)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(hand)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, TOOL_TIP))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateExcludeFromArticulationAttr(True)
    jprim = joint.GetPrim()
    for axis in ("transX", "transY"):
        lim = UsdPhysics.LimitAPI.Apply(jprim, axis)
        lim.CreateLowAttr(0.0)
        lim.CreateHighAttr(0.0)
    lim = UsdPhysics.LimitAPI.Apply(jprim, "transZ")
    lim.CreateLowAttr(0.0)
    lim.CreateHighAttr(0.004)
    for axis in ("rotX", "rotY", "rotZ"):
        lim = UsdPhysics.LimitAPI.Apply(jprim, axis)
        lim.CreateLowAttr(-2.0)
        lim.CreateHighAttr(2.0)
    for axis in ("transZ", "rotX", "rotY", "rotZ"):
        d = UsdPhysics.DriveAPI.Apply(jprim, axis)
        d.CreateStiffnessAttr(1000.0)
        d.CreateDampingAttr(100.0)
    robot_schema.ApplyAttachmentPointAPI(jprim)
    jprim.GetAttribute("isaac:forwardAxis").Set("Z")
    jprim.GetAttribute("isaac:clearanceOffset").Set(0.0)

    gp = f"{hand}/VacuumTool/SurfaceGripper"
    robot_schema.CreateSurfaceGripper(stage, gp)
    g = stage.GetPrimAtPath(gp)
    g.GetRelationship("isaac:attachmentPoints").SetTargets([Sdf.Path(jp)])
    g.GetAttribute("isaac:maxGripDistance").Set(0.012)
    return gp


def camera_intrinsics():
    fx = CAM_RES[0] / 2.0 / math.tan(math.radians(CAM_HFOV_DEG) / 2.0)
    return {"fx": fx, "fy": fx, "cx": CAM_RES[0] / 2.0, "cy": CAM_RES[1] / 2.0,
            "width": CAM_RES[0], "height": CAM_RES[1]}


def camera_to_world(camera_path="/World/RealsenseCamera"):
    stage = omni.usd.get_context().get_stage()
    m = UsdGeom.Xformable(stage.GetPrimAtPath(camera_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m)
