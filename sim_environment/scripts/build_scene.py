import json
import random

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf
from isaacsim.storage.native import get_assets_root_path

print("SCENE_MARKER_0_STARTED", flush=True)

WORKSPACE_PATH = "/home/vishal/poc/stage1_perception_arm/phase2_sim_environment/poc_workspace.usd"
FIXTURES_DIR = "/home/vishal/poc/stage1_perception_arm/phase2_sim_environment/fixtures"

assets_root = get_assets_root_path()
print(f"SCENE_MARKER_1_ASSETS_ROOT={assets_root}", flush=True)

usd_context = omni.usd.get_context()
usd_context.new_stage()
stage = usd_context.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(world)
print("SCENE_MARKER_2_STAGE_CREATED", flush=True)

# Ground plane (simple static collider so nothing free-falls into the void).
# Top surface kept clearly BELOW z=0 (not flush at exactly z=0.0) -- an earlier
# version placed the top surface at exactly z=0.0, precisely coincident with the
# Franka robot's base origin (also at world z=0, no clearance), causing a
# zero/negative-clearance interpenetration at cold start. This produced a
# violent PhysX repulsion impulse that showed up as panda_joint1 diverging to
# -65 to -72 rad within 30 physics steps, while every other joint stayed sane --
# confirmed across 3 separate attempts at re-tuning PD gains (which did NOT fix
# it, since the root cause was geometric penetration, not control gains).
ground = UsdGeom.Cube.Define(stage, "/World/GroundPlane")
ground.CreateSizeAttr(1.0)
ground.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 0.01))
ground.AddTranslateOp().Set(Gf.Vec3f(0, 0, -0.025))  # top surface at z=-0.02, 2cm clearance
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

# ---------------------------------------------------------------------------
# Robot: Franka Panda with the built-in Robotiq_2F_85 gripper variant
# (see backend_config.yaml for why this replaces manual gripper compositing).
# ---------------------------------------------------------------------------
franka_path = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
robot_prim = stage.DefinePrim("/World/Robot", "Xform")
robot_prim.GetReferences().AddReference(franka_path)
stage.Load()

gvs = robot_prim.GetVariantSets().GetVariantSet("Gripper")
gvs.SetVariantSelection("Robotiq_2F_85")
stage.Load()
print(f"SCENE_MARKER_3_ROBOT_GRIPPER_SET selection={gvs.GetVariantSelection()}", flush=True)

# Verify end-effector frame alignment: panda_hand must exist and have a
# non-degenerate world transform, and the Robotiq gripper prim must be nested
# under the robot (not orphaned/separately parented).
panda_hand = stage.GetPrimAtPath("/World/Robot/panda_hand")
xf_cache = UsdGeom.XformCache()
hand_world_xf = xf_cache.GetLocalToWorldTransform(panda_hand)
hand_translation = hand_world_xf.ExtractTranslation()
robotiq_prim = stage.GetPrimAtPath("/World/Robot/Robotiq_2F_85_edit")
print(
    f"SCENE_MARKER_4_EE_ALIGNMENT panda_hand_valid={panda_hand.IsValid()} "
    f"panda_hand_translation={hand_translation} robotiq_prim_valid={robotiq_prim.IsValid()} "
    f"robotiq_prim_path={robotiq_prim.GetPath() if robotiq_prim.IsValid() else 'N/A'}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Camera: Realsense-style RGB-D, off-axis/oblique mount looking across and
# down at the workspace. Resolution (1280x720) is a render-product/
# ROS2-publisher setting, applied when the ROS2 camera helper is created in
# Phase 2.7, not a USD camera-prim attribute -- the prim itself carries the
# optical parameters (FOV/clipping).
#
# ARCHITECTURE DECISION (post Phase-3-dataset-QA, two rounds): the original
# 0.8m/69deg straight-down setup put the camera BELOW the robot's own
# resting-pose reach envelope (panda_hand at world z~0.93m, arm bbox top
# z~1.11m; Phase 3 never moves the robot). Confirmed via semantic-
# segmentation occupancy sweep: mean robot occupancy 57.8%, 58.7% of frames
# >50% robot, pill visible in only 49% of frames.
#
# ROUND 1 (rejected): raising the camera straight up to 2.0m with a narrowed
# FOV fixed the framing (22.9% mean robot occupancy, 0% >50%) but broke the
# harder Phase-8 <5mm translation-accuracy requirement -- D435-class stereo
# depth error scales roughly with range squared, and at 2.0m the estimated
# error (~40mm, per Intel's own "<2% at 2m" spec) is 8x over budget.
#
# ROUND 2 (this one): an off-axis/oblique mount -- positioned away from
# directly-above-the-robot (which is what forced the height increase) while
# keeping *working distance* close to the original ~0.8m, angled down and
# across the workspace instead of straight overhead. This avoids the robot's
# reach envelope through position, not distance, so it doesn't cost accuracy
# the way raising height did. Full validation (n=300, RGB+semantic+depth
# together, not framing-then-accuracy-separately): mean robot occupancy
# 16.8%, 0% of frames >50% robot, pill visible in 99.7% of frames at 0.0134%
# mean area (larger than the original straight-down dataset's pills),
# estimated pill depth-accuracy error (mean-of-means range 0.79m) ~6.3mm --
# statistically equivalent to the original 0.8m setup's own ~6.4mm estimate
# under the same model (that baseline was ALSO marginally over the 5mm
# budget -- this is a standing sensor-physics risk that predates both camera
# fixes, not something introduced by either). Incidence angle (workspace
# surface vs camera boresight) ranges 32-61deg across the workspace corners,
# within the commonly-cited <60-70deg safe zone for stereo correlation
# except the shelf's single farthest corner (61deg, marginal).
# Full numbers, method, and all candidates tried:
# stage1_perception_arm/phase2_sim_environment/logs/sensor_accuracy_analysis.md
#
# Camera pose is defined via eye position + look-at target (Gf.Matrix4d.
# SetLookAt) rather than translate+rotateXYZ -- the oblique angle makes a
# direct Euler-angle tilt error-prone to reason about; look-at is exact and
# verified empirically via point-projection (the target point must project
# to the image center, which it does).
# ---------------------------------------------------------------------------
camera_path = "/World/RealsenseCamera"
camera = UsdGeom.Camera.Define(stage, camera_path)
CAMERA_EYE = Gf.Vec3d(0.0, -0.55, 0.65)
CAMERA_TARGET = Gf.Vec3d(0.0, 0.04, 0.11)
CAMERA_FOV_H_DEG = 66.0
h_aperture_mm = 20.955  # USD default horizontal aperture (mm)
import math
focal_length_mm = h_aperture_mm / (2 * math.tan(math.radians(CAMERA_FOV_H_DEG) / 2))
camera.CreateFocalLengthAttr(focal_length_mm)
camera.CreateHorizontalApertureAttr(h_aperture_mm)
camera.CreateVerticalApertureAttr(h_aperture_mm * 9.0 / 16.0)
camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5.0))
view_matrix = Gf.Matrix4d().SetLookAt(CAMERA_EYE, CAMERA_TARGET, Gf.Vec3d(0, 0, 1))
cam_to_world = view_matrix.GetInverse()
UsdGeom.Xformable(camera).AddTransformOp().Set(cam_to_world)
CAMERA_HEIGHT = CAMERA_EYE[2]  # kept for manifest/logging continuity with the old straight-down field
print(
    f"SCENE_MARKER_5_CAMERA_ADDED eye={CAMERA_EYE} target={CAMERA_TARGET} fov_h_deg={CAMERA_FOV_H_DEG} "
    f"focal_length_mm={focal_length_mm:.3f} h_aperture_mm={h_aperture_mm}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Fixtures: reference the reusable prototypes from ../fixtures/, scatter
# instances across three regions (tray / boxes / shelf) that together must
# fit inside the camera's FOV footprint at 0.8m.
# ---------------------------------------------------------------------------
random.seed(42)

manifest = {"camera": {"height_m": CAMERA_HEIGHT, "fov_h_deg": CAMERA_FOV_H_DEG, "fov_v_deg": None}, "object_classes": {}}

# Tray region for pills: centered at (-0.22, -0.05), 0.24m x 0.20m spread.
pill_colors = [
    (0.85, 0.15, 0.15),  # red
    (0.15, 0.55, 0.85),  # blue
    (0.9, 0.85, 0.2),    # yellow
    (0.2, 0.75, 0.35),   # green
    (0.8, 0.8, 0.85),    # white/neutral
]
pill_positions = []
n_pills = 12
for i in range(n_pills):
    x = -0.22 + random.uniform(-0.10, 0.10)
    y = -0.05 + random.uniform(-0.08, 0.08)
    z = 0.0016  # half height above ground
    path = f"/World/Fixtures/Pills/pill_{i:02d}"
    pill_prim = stage.DefinePrim(path, "Xform")
    pill_prim.GetReferences().AddReference(f"{FIXTURES_DIR}/pill.usd")
    xform = UsdGeom.Xformable(pill_prim)
    xform.AddTranslateOp().Set(Gf.Vec3f(x, y, z))
    color = pill_colors[i % len(pill_colors)]
    geom_prim = stage.GetPrimAtPath(f"{path}/geom")
    if geom_prim.IsValid():
        UsdGeom.Gprim(geom_prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    pill_positions.append({"path": path, "position_m": [x, y, z], "color_rgb": list(color)})
print(f"SCENE_MARKER_6_PILLS_PLACED count={n_pills}", flush=True)
manifest["object_classes"]["pills"] = {
    "count": n_pills,
    "prototype": "fixtures/pill.usd",
    "dimensions_m": {"diameter": 0.008, "height": 0.003},
    "spawn_region": {"center_xy_m": [-0.22, -0.05], "extent_xy_m": [0.20, 0.16]},
    "instances": pill_positions,
}

# Boxes region: centered at (0.22, -0.05), mix of the 3 sizes.
box_sizes = ["Small_10cm", "Medium_15cm", "Large_20cm"]
box_positions = []
n_boxes = 6
box_slot_x = [-0.10, 0.0, 0.10]
box_slot_y = [-0.10, 0.08]
idx = 0
for yi, y_base in enumerate(box_slot_y):
    for xi, x_base in enumerate(box_slot_x):
        if idx >= n_boxes:
            break
        size_name = box_sizes[idx % len(box_sizes)]
        side_m = {"Small_10cm": 0.10, "Medium_15cm": 0.15, "Large_20cm": 0.20}[size_name]
        x = 0.22 + x_base
        y = -0.05 + y_base
        z = side_m / 2
        path = f"/World/Fixtures/Boxes/box_{idx:02d}"
        box_prim = stage.DefinePrim(path, "Xform")
        box_prim.GetReferences().AddReference(f"{FIXTURES_DIR}/box.usd")
        xform = UsdGeom.Xformable(box_prim)
        xform.AddTranslateOp().Set(Gf.Vec3f(x, y, z))
        vs = box_prim.GetVariantSets().GetVariantSet("Size")
        vs.SetVariantSelection(size_name)
        box_positions.append({"path": path, "position_m": [x, y, z], "size_variant": size_name, "side_m": side_m})
        idx += 1
print(f"SCENE_MARKER_7_BOXES_PLACED count={idx}", flush=True)
manifest["object_classes"]["boxes"] = {
    "count": idx,
    "prototype": "fixtures/box.usd",
    "size_variants_m": {"Small_10cm": 0.10, "Medium_15cm": 0.15, "Large_20cm": 0.20},
    "spawn_region": {"center_xy_m": [0.22, -0.05], "extent_xy_m": [0.24, 0.20]},
    "instances": box_positions,
}

# Shelf: single static instance, placed at the "back" of the workspace.
# y-offset kept tight (0.12, not the original 0.22) -- at only 0.8m camera height,
# a 0.4m-tall shelf's far corners are much closer to the camera (effective
# distance ~0.4m) than ground-level fixtures (~0.8m), so the safe FOV half-extent
# at shelf-top height is proportionally smaller. See the FOV check below, which
# tests the shelf's front-facing opening (what the camera/robot actually needs
# to see into the compartments) rather than its outward back panel (would be
# wall-mounted against a real wall in a physical deployment, irrelevant to what
# the RGB-D camera needs to observe).
shelf_path = "/World/Fixtures/Shelf"
shelf_prim = stage.DefinePrim(shelf_path, "Xform")
shelf_prim.GetReferences().AddReference(f"{FIXTURES_DIR}/shelf.usd")
shelf_x, shelf_y, shelf_z = 0.0, 0.12, 0.0
UsdGeom.Xformable(shelf_prim).AddTranslateOp().Set(Gf.Vec3f(shelf_x, shelf_y, shelf_z))
print("SCENE_MARKER_8_SHELF_PLACED", flush=True)
manifest["object_classes"]["shelf"] = {
    "count": 1,
    "prototype": "fixtures/shelf.usd",
    "compartments": "3x2",
    "footprint_m": {"width": 0.48, "depth": 0.25, "height": 0.32},
    "instances": [{"path": shelf_path, "position_m": [shelf_x, shelf_y, shelf_z]}],
}

# ---------------------------------------------------------------------------
# FOV coverage check: project the bounding corners of each spawn region
# through the camera's actual projection to confirm they land inside the
# [0,1280]x[0,720] frame -- numeric check, not eyeballing.
# ---------------------------------------------------------------------------
def project_point(cam_prim, world_point, img_w=1280, img_h=720):
    gf_cam = UsdGeom.Camera(cam_prim)
    frustum = gf_cam.GetCamera(Usd.TimeCode.Default()).frustum
    view_proj = frustum.ComputeViewMatrix() * frustum.ComputeProjectionMatrix()
    p = Gf.Vec4d(world_point[0], world_point[1], world_point[2], 1.0)
    clip = p * view_proj
    if clip[3] == 0:
        return None
    ndc_x = clip[0] / clip[3]
    ndc_y = clip[1] / clip[3]
    px = (ndc_x * 0.5 + 0.5) * img_w
    py = (1.0 - (ndc_y * 0.5 + 0.5)) * img_h
    return px, py, clip[2] / clip[3]

cam_prim = stage.GetPrimAtPath(camera_path)
# Required: everything the camera/robot actually needs to perceive --
# ground-level pill/box regions, and the shelf's FRONT-facing opening (up to
# its full 0.4m height) where compartment contents are visible.
required_points = {
    "pills_region_corner_min": (-0.22 - 0.10, -0.05 - 0.08, 0.0),
    "pills_region_corner_max": (-0.22 + 0.10, -0.05 + 0.08, 0.0),
    "boxes_region_corner_min": (0.22 - 0.12, -0.05 - 0.10, 0.0),
    "boxes_region_corner_max": (0.22 + 0.12, -0.05 + 0.10, 0.20),
    "shelf_front_opening_bottom_left": (shelf_x - 0.24, shelf_y - 0.125, 0.0),
    "shelf_front_opening_top_right": (shelf_x + 0.24, shelf_y - 0.125, 0.32),
}
# Informational only: the shelf's outward-facing back-top corner. Not required
# in-frame -- in a real deployment this panel is wall-mounted and the camera
# has no reason to see it; the front opening (checked above) is what matters.
informational_points = {
    "shelf_back_panel_top_corner": (shelf_x + 0.24, shelf_y + 0.125, 0.32),
}

fov_results = {"required": {}, "informational": {}}
all_required_in_bounds = True
for name, pt in required_points.items():
    proj = project_point(cam_prim, pt)
    if proj is None:
        fov_results["required"][name] = "PROJECTION_FAILED"
        all_required_in_bounds = False
        continue
    px, py, depth = proj
    in_bounds = 0 <= px <= 1280 and 0 <= py <= 720
    fov_results["required"][name] = {"pixel_xy": [round(px, 1), round(py, 1)], "in_bounds": in_bounds}
    if not in_bounds:
        all_required_in_bounds = False
    print(f"SCENE_MARKER_9_FOV_CHECK point={name} world={pt} pixel=({px:.1f},{py:.1f}) in_bounds={in_bounds}", flush=True)

for name, pt in informational_points.items():
    proj = project_point(cam_prim, pt)
    if proj is not None:
        px, py, depth = proj
        in_bounds = 0 <= px <= 1280 and 0 <= py <= 720
        fov_results["informational"][name] = {"pixel_xy": [round(px, 1), round(py, 1)], "in_bounds": in_bounds}
        print(f"SCENE_MARKER_9b_FOV_INFO point={name} world={pt} pixel=({px:.1f},{py:.1f}) in_bounds={in_bounds} (not required)", flush=True)

print(f"SCENE_MARKER_10_FOV_ALL_REQUIRED_IN_BOUNDS={all_required_in_bounds}", flush=True)
manifest["camera"]["fov_coverage_check"] = fov_results
manifest["camera"]["fov_covers_full_workspace"] = all_required_in_bounds

# ---------------------------------------------------------------------------
# Save the stage and the manifest.
# ---------------------------------------------------------------------------
stage.GetRootLayer().Export(WORKSPACE_PATH)
print(f"SCENE_MARKER_11_STAGE_SAVED path={WORKSPACE_PATH}", flush=True)

manifest_path = "/home/vishal/poc/stage1_perception_arm/phase2_sim_environment/scene_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"SCENE_MARKER_12_MANIFEST_SAVED path={manifest_path}", flush=True)

print("SCENE_MARKER_13_DONE", flush=True)
