"""Phase 3 synthetic data generation -- small-scale (10-20 frame) validation pass
before the full 50k run. Uses poc_workspace.usd's existing fixtures (pills,
boxes, shelf) with domain randomization, per master_prompt_stage1.md Phase 3:
  - Random lighting (direction, intensity, color temperature)
  - Random textures on objects and background
  - Random object poses (within tray/shelf/workspace bounds)
  - Random camera viewpoint (+-15 deg tilt, +-10cm position)
Annotations: RGB, depth, instance segmentation masks, 2D bboxes, 6D poses,
camera intrinsics. Output: BasicWriter (COCO-adjacent: rgb/depth/seg/bbox/
camera params) + PoseWriter (centerpose format, 6D pose) on the same render
product.
"""
import asyncio
import json
import math
import os
import random

import numpy as np
from PIL import Image

import omni.usd
import omni.replicator.core as rep
import omni.timeline
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.utils.semantics import add_labels
from pxr import Gf, UsdGeom, UsdLux, UsdShade, Sdf

print("SDG_MARKER_0_STARTED", flush=True)

WORKSPACE_PATH = "/home/vishal/poc/stage1_perception_arm/phase2_sim_environment/poc_workspace.usd"
OUTPUT_DIR = "/home/vishal/poc/stage1_perception_arm/phase3_synthetic_data/validation_run"
NUM_FRAMES = 12
CAMERA_PATH = "/World/RealsenseCamera"

usd_context = omni.usd.get_context()
usd_context.open_stage(WORKSPACE_PATH)
app_utils.update_app(steps=3)
stage = usd_context.get_stage()
print(f"SDG_MARKER_1_STAGE_OPENED path={WORKSPACE_PATH}", flush=True)

# ---------------------------------------------------------------------------
# Semantic labels: required for instance segmentation / bbox annotations to
# know which prims belong to which class. Applied to the fixture instances
# already placed by build_scene.py.
# ---------------------------------------------------------------------------
labeled_count = {"pill": 0, "box": 0, "shelf": 0}
for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    if "/World/Fixtures/Pills/" in path_str and prim.GetPath().name.startswith("pill_"):
        add_labels(prim, labels=["pill"], instance_name="class")
        labeled_count["pill"] += 1
    elif "/World/Fixtures/Boxes/" in path_str and prim.GetPath().name.startswith("box_"):
        add_labels(prim, labels=["box"], instance_name="class")
        labeled_count["box"] += 1
    elif path_str == "/World/Fixtures/Shelf":
        add_labels(prim, labels=["shelf"], instance_name="class")
        labeled_count["shelf"] += 1
print(f"SDG_MARKER_2_LABELS_APPLIED counts={labeled_count}", flush=True)

# ---------------------------------------------------------------------------
# Collect randomizable prim references.
# ---------------------------------------------------------------------------
pill_prims = [stage.GetPrimAtPath(f"/World/Fixtures/Pills/pill_{i:02d}") for i in range(12)]
pill_prims = [p for p in pill_prims if p.IsValid()]
box_prims = [stage.GetPrimAtPath(f"/World/Fixtures/Boxes/box_{i:02d}") for i in range(6)]
box_prims = [p for p in box_prims if p.IsValid()]
print(f"SDG_MARKER_3_PRIMS_COLLECTED pills={len(pill_prims)} boxes={len(box_prims)}", flush=True)

# Cardboard material for boxes -- randomize its diffuse color per capture (texture variation stand-in)
box_material_shader_path = None
for box_prim in box_prims:
    shader_prim = stage.GetPrimAtPath(f"{box_prim.GetPath()}/geom/CardboardMaterial/PreviewSurface")
    if shader_prim.IsValid():
        box_material_shader_path = str(shader_prim.GetPath())
        break
print(f"SDG_MARKER_4_BOX_MATERIAL_SHADER path={box_material_shader_path}", flush=True)

# ---------------------------------------------------------------------------
# Lighting: a distant light (direction/intensity/color temperature randomized)
# plus a dome light for ambient background variation.
# ---------------------------------------------------------------------------
distant_light = UsdLux.DistantLight.Define(stage, "/World/SDG_DistantLight")
distant_light.CreateIntensityAttr(3000.0)
distant_light.GetPrim().CreateAttribute("inputs:colorTemperature", Sdf.ValueTypeNames.Float).Set(6500.0)
distant_light.GetPrim().CreateAttribute("inputs:enableColorTemperature", Sdf.ValueTypeNames.Bool).Set(True)
dome_light = UsdLux.DomeLight.Define(stage, "/World/SDG_DomeLight")
dome_light.CreateIntensityAttr(500.0)
print("SDG_MARKER_5_LIGHTS_CREATED", flush=True)

app_utils.update_app(steps=2)

# ---------------------------------------------------------------------------
# Render product + writers.
# ---------------------------------------------------------------------------
render_product = rep.create.render_product(CAMERA_PATH, (1280, 720))
print(f"SDG_MARKER_6_RENDER_PRODUCT_CREATED path={render_product.path}", flush=True)
# Interleave several small update steps before attaching writers -- an earlier
# Gate M2 script segfaulted creating a render product in one uninitialized
# burst; small interleaved steps let the RTX pipeline initialize gradually.
for _ in range(10):
    app_utils.update_app(steps=1)

rep.orchestrator.set_capture_on_play(False)

basic_writer = rep.writers.get("BasicWriter")
basic_writer.initialize(
    output_dir=f"{OUTPUT_DIR}/basic_writer",
    rgb=True,
    distance_to_image_plane=True,
    instance_segmentation=True,
    bounding_box_2d_tight=True,
    camera_params=True,
    semantic_segmentation=True,
)
basic_writer.attach([render_product])
print("SDG_MARKER_7_BASIC_WRITER_ATTACHED", flush=True)

pose_writer = rep.writers.get("PoseWriter")
pose_writer.initialize(
    output_dir=f"{OUTPUT_DIR}/pose_writer",
    format="centerpose",
    write_debug_images=False,  # disabled for the real run -- visualization aid only, not needed at 50k scale
    skip_empty_frames=False,
)
pose_writer.attach([render_product])
print("SDG_MARKER_8_POSE_WRITER_ATTACHED", flush=True)

BASIC_WRITER_DIR = f"{OUTPUT_DIR}/basic_writer"
FRAME_PADDING = 4  # BasicWriter's default frame_padding


def convert_and_delete_depth(frame_index: int) -> int:
    """Convert this frame's raw float32 distance_to_image_plane .npy to a
    uint16 PNG in millimeters (standard RGB-D pose-dataset convention; FoundationPose
    is the actual depth consumer, not CenterPose -- confirmed CenterPose's
    PoseWriter output is RGB+pose/keypoints only, no depth reference) and
    delete the raw npy. Called per-frame, right after that frame's writes are
    confirmed flushed to disk (via wait_until_complete_async), so temporary
    storage never accumulates beyond one frame's raw npy at a time -- at 50k
    frames, keeping all raw npys around simultaneously would need ~180GB of
    transient space even though the final PNG dataset only needs ~1-2GB.
    Returns the resulting PNG file size in bytes (0 if the npy was missing).
    """
    npy_path = f"{BASIC_WRITER_DIR}/distance_to_image_plane_{frame_index:0{FRAME_PADDING}}.npy"
    png_path = f"{BASIC_WRITER_DIR}/distance_to_image_plane_{frame_index:0{FRAME_PADDING}}.png"
    if not os.path.exists(npy_path):
        return 0
    depth_m = np.load(npy_path)
    depth_mm = depth_m * 1000.0
    depth_mm = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm, mode="I;16").save(png_path, optimize=True)
    os.remove(npy_path)
    return os.path.getsize(png_path)

# ---------------------------------------------------------------------------
# Randomizers.
# ---------------------------------------------------------------------------
random.seed(123)

# Off-axis/oblique mount -- matches the corrected base pose in poc_workspace.usd.
# Round 1 (straight-up height increase to 2.0m) fixed framing but broke the
# <5mm depth-accuracy budget at range. Round 2 (this oblique mount) keeps
# working distance close to the original ~0.8m by moving the camera off to
# the side and angling it across the workspace instead of straight down. See
# build_scene.py's camera-section comment and
# phase2_sim_environment/logs/sensor_accuracy_analysis.md for the full
# investigation writeup.
CAMERA_EYE_BASE = (0.0, -0.55, 0.65)
CAMERA_TARGET_BASE = (0.0, 0.04, 0.11)
CAMERA_EYE_JITTER = 0.04
CAMERA_TARGET_JITTER = 0.03


def _get_or_add_op(xformable, op_type, op_suffix=""):
    """Reuse an existing xformOp of the given type if present, else add one.
    Avoids 'xformOp already exists in xformOpOrder' errors from blindly
    removing-and-re-adding ops every call (RemoveProperty doesn't reliably
    clear xformOpOrder mid-iteration -- hit this directly on the 2nd frame)."""
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    return xformable.AddXformOp(op_type, opSuffix=op_suffix)


def randomize_camera():
    """+-4cm eye jitter, +-3cm look-at target jitter, per corrected Phase 3 spec."""
    cam_prim = stage.GetPrimAtPath(CAMERA_PATH)
    xformable = UsdGeom.Xformable(cam_prim)
    ex = CAMERA_EYE_BASE[0] + random.uniform(-CAMERA_EYE_JITTER, CAMERA_EYE_JITTER)
    ey = CAMERA_EYE_BASE[1] + random.uniform(-CAMERA_EYE_JITTER, CAMERA_EYE_JITTER)
    ez = CAMERA_EYE_BASE[2] + random.uniform(-CAMERA_EYE_JITTER, CAMERA_EYE_JITTER)
    tx = CAMERA_TARGET_BASE[0] + random.uniform(-CAMERA_TARGET_JITTER, CAMERA_TARGET_JITTER)
    ty = CAMERA_TARGET_BASE[1] + random.uniform(-CAMERA_TARGET_JITTER, CAMERA_TARGET_JITTER)
    tz = CAMERA_TARGET_BASE[2] + random.uniform(-CAMERA_TARGET_JITTER, CAMERA_TARGET_JITTER)
    view_matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(ex, ey, ez), Gf.Vec3d(tx, ty, tz), Gf.Vec3d(0, 0, 1))
    cam_to_world = view_matrix.GetInverse()
    transform_op = _get_or_add_op(xformable, UsdGeom.XformOp.TypeTransform)
    transform_op.Set(cam_to_world)


def randomize_pill_positions():
    for i, prim in enumerate(pill_prims):
        xformable = UsdGeom.Xformable(prim)
        x = -0.22 + random.uniform(-0.10, 0.10)
        y = -0.05 + random.uniform(-0.08, 0.08)
        z = 0.0016
        for op in xformable.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                op.Set(Gf.Vec3f(x, y, z))
                break


def randomize_box_positions():
    for i, prim in enumerate(box_prims):
        xformable = UsdGeom.Xformable(prim)
        x = 0.22 + random.uniform(-0.12, 0.12)
        y = -0.05 + random.uniform(-0.10, 0.10)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                cur = op.Get()
                op.Set(Gf.Vec3f(x, y, cur[2] if cur else 0.05))
                break


def randomize_box_material_color():
    if not box_material_shader_path:
        return
    shader_prim = stage.GetPrimAtPath(box_material_shader_path)
    color_attr = shader_prim.GetAttribute("inputs:diffuseColor")
    if color_attr.IsValid():
        # Cardboard-ish hue variation: vary around a brown base, not fully random colors
        base = (0.72, 0.55, 0.35)
        jitter = lambda v: max(0.0, min(1.0, v + random.uniform(-0.15, 0.15)))
        color_attr.Set(Gf.Vec3f(jitter(base[0]), jitter(base[1]), jitter(base[2])))


def randomize_lighting():
    distant_light.CreateIntensityAttr().Set(random.uniform(1500.0, 5000.0))
    distant_light.GetPrim().GetAttribute("inputs:colorTemperature").Set(random.uniform(4000.0, 9000.0))
    dome_light.CreateIntensityAttr().Set(random.uniform(200.0, 900.0))
    # Direction: rotate the distant light
    dxformable = UsdGeom.Xformable(distant_light.GetPrim())
    rotate_op = _get_or_add_op(dxformable, UsdGeom.XformOp.TypeRotateXYZ)
    rotate_op.Set(Gf.Vec3f(random.uniform(20.0, 80.0), random.uniform(0.0, 360.0), 0.0))


print("SDG_MARKER_9_RANDOMIZERS_DEFINED", flush=True)

# ---------------------------------------------------------------------------
# Capture loop.
# ---------------------------------------------------------------------------
timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(0)
timeline.set_end_time(1000000)
timeline.set_looping(False)
timeline.play()
app_utils.update_app(steps=2)

# rep.orchestrator.step() (sync) requires a "standalone workflow" (SimulationApp)
# and errors when called from a kit-binary --exec script ("OrchestratorError:
# ... may not be made from within Kit. Please use the async function
# step_async."). Use step_async() instead, scheduled via asyncio.ensure_future
# and pumped to completion with app_utils.update_app() -- the standard pattern
# for driving async Kit/Replicator calls from a synchronous exec script
# (confirmed against omni.replicator.core's own internal usage of this exact
# pattern for scheduling async test coroutines).
capture_done = {"count": 0}
depth_png_sizes = []


async def capture_all_frames():
    for i in range(NUM_FRAMES):
        randomize_camera()
        randomize_pill_positions()
        randomize_box_positions()
        randomize_box_material_color()
        randomize_lighting()
        await rep.orchestrator.step_async(delta_time=0.0, pause_timeline=False)
        # Wait for this frame's writer backend I/O to actually land on disk
        # before touching the npy -- step_async() scheduling the write is not
        # the same as the write having completed.
        await rep.orchestrator.wait_until_complete_async()
        png_size = convert_and_delete_depth(i)
        depth_png_sizes.append(png_size)
        capture_done["count"] = i + 1
        print(f"SDG_MARKER_10_FRAME_CAPTURED index={i} depth_png_bytes={png_size}", flush=True)


task = asyncio.ensure_future(capture_all_frames())
while capture_done["count"] < NUM_FRAMES and not task.done():
    app_utils.update_app(steps=1)
if task.exception() is not None:
    raise task.exception()
print(f"SDG_MARKER_10b_ALL_FRAMES_DONE count={capture_done['count']}", flush=True)

wait_task = asyncio.ensure_future(rep.orchestrator.wait_until_complete_async())
while not wait_task.done():
    app_utils.update_app(steps=1)
if wait_task.exception() is not None:
    raise wait_task.exception()
print("SDG_MARKER_11_WAIT_COMPLETE", flush=True)

timeline.stop()

# Authoritative per-frame size: sum every byte actually on disk under
# OUTPUT_DIR after the full pipeline (writers + depth conversion) has run,
# divided by frame count -- not an estimate from individual file samples.
total_bytes = 0
for root, _, files in os.walk(OUTPUT_DIR):
    for f in files:
        total_bytes += os.path.getsize(os.path.join(root, f))
total_mb = total_bytes / 1024 / 1024
per_frame_mb = total_mb / NUM_FRAMES
print(
    f"SDG_MARKER_13_SIZE_SUMMARY total_bytes={total_bytes} total_mb={total_mb:.3f} "
    f"num_frames={NUM_FRAMES} per_frame_mb={per_frame_mb:.4f}",
    flush=True,
)
n_full = 50000
full_run_gb = per_frame_mb * n_full / 1024
available_gb = 175.5
print(
    f"SDG_MARKER_14_50K_PROJECTION per_frame_mb={per_frame_mb:.4f} "
    f"projected_total_gb={full_run_gb:.2f} available_gb={available_gb} "
    f"fits={full_run_gb <= available_gb} margin_gb={available_gb - full_run_gb:.2f}",
    flush=True,
)

print("SDG_MARKER_12_DONE", flush=True)
