"""Phase 3 synthetic data generation -- full-scale run (NUM_FRAMES, default 50000).
Same validated pipeline as phase3_sdg_validate.py (domain randomization + BasicWriter
+ PoseWriter + per-frame depth npy->uint16mm-PNG conversion), plus crash-resume: on
startup, scans OUTPUT_DIR for the last fully-written frame and continues from there
instead of restarting at 0, so a crash partway through an unattended multi-hour run
doesn't waste already-completed frames. Rerunning this script unchanged after a
crash resumes automatically -- no flags needed.
"""
import asyncio
import os
import random
import time

import numpy as np
from PIL import Image

import omni.usd
import omni.replicator.core as rep
import omni.timeline
import omni.kit.app
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.utils.semantics import add_labels
from pxr import Gf, UsdGeom, UsdLux, Sdf

print("SDG_MARKER_0_STARTED", flush=True)

WORKSPACE_PATH = "/home/vishal/poc/stage1_perception_arm/phase2_sim_environment/poc_workspace.usd"
OUTPUT_DIR = "/home/vishal/poc/stage1_perception_arm/phase3_synthetic_data/full_run_50k"
NUM_FRAMES = 50000
CAMERA_PATH = "/World/RealsenseCamera"
PROGRESS_INTERVAL = 1000

BASIC_WRITER_DIR = f"{OUTPUT_DIR}/basic_writer"
POSE_WRITER_DIR = f"{OUTPUT_DIR}/pose_writer"
FRAME_PADDING = 4  # BasicWriter's default frame_padding
POSE_FRAME_PADDING = 6  # PoseWriter's default frame_padding


def frame_complete(i: int) -> bool:
    """A frame is complete iff its depth PNG exists and its source npy is gone --
    depth conversion is the last step of each frame's per-iteration work, and it
    only runs after wait_until_complete_async() confirms every writer (BasicWriter
    AND PoseWriter, same orchestrator queue) has flushed that frame to disk. So
    this single check implies the whole frame -- both writers -- landed cleanly."""
    depth_png = f"{BASIC_WRITER_DIR}/distance_to_image_plane_{i:0{FRAME_PADDING}}.png"
    depth_npy = f"{BASIC_WRITER_DIR}/distance_to_image_plane_{i:0{FRAME_PADDING}}.npy"
    pose_json = f"{POSE_WRITER_DIR}/{i:0{POSE_FRAME_PADDING}}.json"
    return os.path.exists(depth_png) and not os.path.exists(depth_npy) and os.path.exists(pose_json)


def find_resume_index(upper_bound: int) -> int:
    """Binary search for the first not-fully-written frame index. Frames are
    written strictly in increasing order (single sequential loop, single writer
    instance per run), so completeness is monotonic and binary search is valid."""
    lo, hi = 0, upper_bound
    while lo < hi:
        mid = (lo + hi) // 2
        if frame_complete(mid):
            lo = mid + 1
        else:
            hi = mid
    return lo


resume_index = find_resume_index(NUM_FRAMES)
print(f"SDG_MARKER_RESUME resume_index={resume_index} num_frames={NUM_FRAMES}", flush=True)

if resume_index >= NUM_FRAMES:
    print("SDG_MARKER_ALREADY_COMPLETE", flush=True)
    import sys
    sys.exit(0)

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
for _ in range(10):
    app_utils.update_app(steps=1)

rep.orchestrator.set_capture_on_play(False)

basic_writer = rep.writers.get("BasicWriter")
basic_writer.initialize(
    output_dir=BASIC_WRITER_DIR,
    rgb=True,
    distance_to_image_plane=True,
    instance_segmentation=True,
    bounding_box_2d_tight=True,
    camera_params=True,
    semantic_segmentation=True,
)
# Resume support: BasicWriter.initialize() resets _frame_id to 0 (plain instance
# attribute, read/written directly by write() for output filenames -- confirmed
# via source inspection of writers_default/basicwriter.py). Re-pointing it at
# resume_index makes the writer continue the existing numbered sequence instead
# of overwriting frames 0..resume_index-1. initialize()'s DiskBackend only ever
# does os.makedirs(exist_ok=True) -- never clears the output dir -- so this is safe.
#
# CRITICAL: write() also silently resets _frame_id back to 0 on its first call
# whenever the sequence_id it computes internally (always "" for our OgnOnFrame
# trigger, since we don't use "on_time" triggers) doesn't match self._sequence_id
# (left at the int 0 by initialize()) -- "" != 0 is always True, so without this
# next line the very first write of a resumed run silently wipes the frame_id
# override above and starts overwriting frames 0, 1, 2... instead of resuming at
# resume_index. Confirmed by direct instrumentation: resuming at frame_id=22
# without this fix corrupted frames 0-7 while silently no-oping on 22-29.
basic_writer._sequence_id = ""
basic_writer._frame_id = resume_index
basic_writer.attach([render_product])
print(f"SDG_MARKER_7_BASIC_WRITER_ATTACHED frame_id_start={basic_writer._frame_id}", flush=True)

pose_writer = rep.writers.get("PoseWriter")
pose_writer.initialize(
    output_dir=POSE_WRITER_DIR,
    format="centerpose",
    write_debug_images=False,  # disabled for the real run -- visualization aid only, not needed at 50k scale
    skip_empty_frames=False,
)
pose_writer._frame_id = resume_index
pose_writer.attach([render_product])
print(f"SDG_MARKER_8_POSE_WRITER_ATTACHED frame_id_start={pose_writer._frame_id}", flush=True)


def convert_and_delete_depth(frame_index: int) -> int:
    """Convert this frame's raw float32 distance_to_image_plane .npy to a
    uint16 PNG in millimeters and delete the raw npy. Returns the PNG size in
    bytes (0 if the npy was missing)."""
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
# Randomizers. NOTE: random.seed(123) always restarts the RNG stream from the
# same point regardless of resume_index (a resume does not replay the exact
# RNG state a continuous run would have had at that frame). This only affects
# which specific random scene configuration each frame index gets -- frame
# content diversity, not correctness -- so it's left simple rather than trying
# to fast-forward RNG state to match an uninterrupted run exactly.
# ---------------------------------------------------------------------------
random.seed(123)

# Off-axis/oblique mount -- matches the corrected base pose in poc_workspace.usd.
# See build_scene.py's camera-section comment and
# phase2_sim_environment/logs/sensor_accuracy_analysis.md for the full
# rationale (round 1, a straight-up height increase to 2.0m, fixed framing
# but broke the <5mm depth-accuracy budget at range; round 2, this oblique
# mount, keeps working distance close to the original ~0.8m by moving the
# camera off to the side and angling it across the workspace instead of
# straight down, avoiding the robot through position rather than distance).
CAMERA_EYE_BASE = (0.0, -0.55, 0.65)
CAMERA_TARGET_BASE = (0.0, 0.04, 0.11)
# Position jitter on both eye and look-at target -- tighter than the old
# straight-down +-10cm since this FOV (66deg, still narrower than the
# original 69deg-but-now-oblique) is more sensitive to aim drift, per the
# lesson from round 1's unscaled-tilt failure mode (occupancy-sweep
# validated: n=300, 0% frames >50% robot, 99.7% pill visible).
CAMERA_EYE_JITTER = 0.04
CAMERA_TARGET_JITTER = 0.03


def _get_or_add_op(xformable, op_type, op_suffix=""):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    return xformable.AddXformOp(op_type, opSuffix=op_suffix)


def randomize_camera():
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
        base = (0.72, 0.55, 0.35)
        jitter = lambda v: max(0.0, min(1.0, v + random.uniform(-0.15, 0.15)))
        color_attr.Set(Gf.Vec3f(jitter(base[0]), jitter(base[1]), jitter(base[2])))


def randomize_lighting():
    distant_light.CreateIntensityAttr().Set(random.uniform(1500.0, 5000.0))
    distant_light.GetPrim().GetAttribute("inputs:colorTemperature").Set(random.uniform(4000.0, 9000.0))
    dome_light.CreateIntensityAttr().Set(random.uniform(200.0, 900.0))
    dxformable = UsdGeom.Xformable(distant_light.GetPrim())
    rotate_op = _get_or_add_op(dxformable, UsdGeom.XformOp.TypeRotateXYZ)
    rotate_op.Set(Gf.Vec3f(random.uniform(20.0, 80.0), random.uniform(0.0, 360.0), 0.0))


print("SDG_MARKER_9_RANDOMIZERS_DEFINED", flush=True)

# ---------------------------------------------------------------------------
# Capture loop -- starts at resume_index, not 0.
# ---------------------------------------------------------------------------
timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(0)
timeline.set_end_time(1000000)
timeline.set_looping(False)
timeline.play()
app_utils.update_app(steps=2)

capture_done = {"count": resume_index}
loop_start_time = time.time()


async def capture_all_frames():
    for i in range(resume_index, NUM_FRAMES):
        randomize_camera()
        randomize_pill_positions()
        randomize_box_positions()
        randomize_box_material_color()
        randomize_lighting()
        await rep.orchestrator.step_async(delta_time=0.0, pause_timeline=False)
        await rep.orchestrator.wait_until_complete_async()
        png_size = convert_and_delete_depth(i)
        capture_done["count"] = i + 1
        if (i + 1) % PROGRESS_INTERVAL == 0 or (i + 1) == NUM_FRAMES:
            elapsed = time.time() - loop_start_time
            done_this_session = (i + 1) - resume_index
            rate = elapsed / done_this_session if done_this_session else 0.0
            remaining = NUM_FRAMES - (i + 1)
            eta_s = remaining * rate
            print(
                f"SDG_MARKER_PROGRESS completed={i + 1} total={NUM_FRAMES} "
                f"pct={100.0 * (i + 1) / NUM_FRAMES:.1f} elapsed_s={elapsed:.1f} "
                f"rate_s_per_frame={rate:.3f} eta_s={eta_s:.0f}",
                flush=True,
            )


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

# Writer.attach() wires up an auto-trigger (OgnOnFrame) independent of the
# explicit step_async() calls above -- confirmed by direct testing that this
# keeps firing on every idle app-update tick as long as the Kit process stays
# alive, silently resetting _frame_id and overwriting already-written frames
# (observed corrupting frames 0-3 during a resume test where the process was
# left running ~2 minutes after its capture loop finished). Detach immediately
# so nothing can write again after this point.
basic_writer.detach()
pose_writer.detach()
print("SDG_MARKER_11b_WRITERS_DETACHED", flush=True)

total_bytes = 0
file_count = 0
for root, _, files in os.walk(OUTPUT_DIR):
    for f in files:
        total_bytes += os.path.getsize(os.path.join(root, f))
        file_count += 1
total_mb = total_bytes / 1024 / 1024
per_frame_mb = total_mb / NUM_FRAMES
print(
    f"SDG_MARKER_13_SIZE_SUMMARY total_bytes={total_bytes} total_mb={total_mb:.3f} "
    f"num_frames={NUM_FRAMES} per_frame_mb={per_frame_mb:.4f} file_count={file_count}",
    flush=True,
)

print("SDG_MARKER_12_DONE", flush=True)

# The Kit app does not self-terminate after an --exec script returns (observed
# throughout this project) -- it idles indefinitely instead, which is exactly
# the window the writer-auto-trigger corruption above happened in. Quit
# explicitly so the process exits on its own the moment this script is done,
# rather than depending on an external kill to happen before anything else
# fires on the idle loop.
omni.kit.app.get_app().post_quit()
