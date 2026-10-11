"""V2 DEMO: warehouse cell, vacuum tool, pills + boxes (run via kit --exec).

Each cycle:
  1. OBSERVE   arm parks out of view; RealSense-style camera renders RGB + depth
  2. DETECT    RT-DETR (trained on the Phase 3 SDG data, no retraining); boxes from the
               full-resolution pass, pills from full + 0.75x + 0.6x passes (pills render
               larger here than in the training frames)
  3. LOCALIZE  boxes: top face from depth (center, height, yaw, size check);
               pills: the 3 mm bump in depth under each detection (rejects false hits)
  4. PICK      Franka (real PhysX PD drives, differential IK) with a vacuum tool:
               Isaac Sim Surface Gripper at the cup face; grasp confirmed by the gripper's
               attach status (the sim equivalent of a vacuum-pressure sensor)
  5. PLACE     pills -> KLT bin, boxes -> tote (squared up), then re-observe

Ground truth is read only to report perception error and final placement.

Env: DEMO_LIVE=1 (GUI + in-app perception panel, default) or 0 (headless composite video)
     DEMO_SEED, DEMO_OUT, DEMO_QUIT_WHEN_DONE=1
"""

import json
import math
import os
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = "/home/vishal/poc/stage1_perception_arm/final_demo"
sys.path.insert(0, HERE)

import omni.kit.app
import omni.timeline
import omni.usd
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils

_em = omni.kit.app.get_app().get_extension_manager()
_em.set_extension_enabled_immediate("isaacsim.robot.experimental.manipulators.examples", True)
_em.set_extension_enabled_immediate("isaacsim.robot.surface_gripper", True)

from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.experimental.manipulators.examples.franka.franka import Franka
from isaacsim.robot.surface_gripper import _surface_gripper
from isaacsim.storage.native import get_assets_root_path
import omni.replicator.core as rep
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from pxr import Usd, UsdGeom

import demo_scene_v2 as S
from localize import localize_box, localize_pill, project

LIVE = os.environ.get("DEMO_LIVE", "1") == "1"
SEED = int(os.environ.get("DEMO_SEED", "11"))
OUT = os.environ.get("DEMO_OUT", "/home/vishal/poc/logs/final_demo/v2_latest")
QUIT_WHEN_DONE = os.environ.get("DEMO_QUIT_WHEN_DONE", "0" if LIVE else "1") == "1"
VENV_PY = "/home/vishal/poc/stage1_perception_arm/phase4_perception_training/rtdetr_venv/bin/python"

DT = 1.0 / 60.0
OBSERVE_Q = [1.25, -0.45, 0.0, -2.3, 0.0, 1.9, 0.785]
READY_Q = [0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741]
BOX_THRESH, PILL_THRESH = 0.30, 0.22
BOX_ZONE = ((0.28, 0.66), (-0.22, 0.28))
PILL_REGION = ((S.PILL_MAT_CENTER[0] - S.PILL_MAT_SIZE[0] / 2, S.PILL_MAT_CENTER[0] + S.PILL_MAT_SIZE[0] / 2),
               (S.PILL_MAT_CENTER[1] - S.PILL_MAT_SIZE[1] / 2, S.PILL_MAT_CENTER[1] + S.PILL_MAT_SIZE[1] / 2))
MAT_TOP = 0.002
CARRY_Z = 0.42          # hand height while carrying boxes over the tote wall
TOTE_FLOOR = 0.009

os.makedirs(os.path.join(OUT, "perception"), exist_ok=True)
if not LIVE:
    os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
LOG = open(os.path.join(OUT, "demo_log.txt"), "w")


def log(msg):
    line = f"[DEMO t={time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


# ------------------------------------------------------------------ quaternions (w,x,y,z)
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def slerp(a, b, t):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + t * (b - a)
        return r / np.linalg.norm(r)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def down_quat(yaw_deg=0.0):
    a = math.radians(yaw_deg)
    return qmul(np.array([math.cos(a / 2), 0, 0, math.sin(a / 2)]), np.array([0.0, 1.0, 0.0, 0.0]))


def quat_angle(a, b):
    return 2 * math.degrees(math.acos(min(1.0, abs(float(np.dot(a, b))))))


# ------------------------------------------------------------------ scene
omni.usd.get_context().new_stage()
app_utils.update_app(steps=2)
SimulationManager.set_physics_sim_device("cpu")
scene = S.build(seed=SEED)
stage_utils.add_reference_to_stage(
    usd_path=get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    path="/World/robot", variants=[("Gripper", "None"), ("Mesh", "Quality")])
GRIPPER = S.add_vacuum_tool("/World/robot")
robot = Franka(robot_path="/World/robot", create_robot=False)
vacuum = _surface_gripper.acquire_surface_gripper_interface()
stage = omni.usd.get_context().get_stage()

sensor_rp = rep.create.render_product(scene["camera_path"], S.CAM_RES)
ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
ann_rgb.attach([sensor_rp])
ann_depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
ann_depth.attach([sensor_rp])
pres_rgb = None
if not LIVE:
    pres_rp = rep.create.render_product(scene["presentation_camera_path"], (1280, 720))
    pres_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    pres_rgb.attach([pres_rp])
K = S.camera_intrinsics()

status = {"phase": "INIT", "cycle": 0, "pills": 0, "boxes": 0, "detail": ""}
panel = {"perception": [], "robot": []}
last_vis = {"img": None}
tracked_pills = []   # pills validated in an earlier view (world xy)
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
except Exception:
    FONT = FONT_S = ImageFont.load_default()

# ------------------------------------------------------------------ live UI panel
ui_state = {}
if LIVE:
    import omni.ui as ui
    from omni.kit.viewport.utility import get_active_viewport
    from omni.kit.viewport.actions.actions import toggle_camera_visibility

    vp = get_active_viewport()
    vp.camera_path = scene["presentation_camera_path"]
    toggle_camera_visibility(vp, visible=False)
    vp.fill_frame = True     # render at the widget size: no letterbox bars
    import carb.settings
    _st = carb.settings.get_settings()
    for _k in ["deviceMemory", "hostMemory", "processMemory", "renderResolution", "renderProgress", "cameraSpeed"]:
        _st.set(f"/persistent/app/viewport/{vp.id}/hud/{_k}/visible", False)
    _st.set(f"/persistent/app/viewport/{vp.id}/hud/visible", True)   # keep the FPS readout (proof of real time)
    for name in ["Stage", "Property", "Content", "Console", "Layer", "Render Settings", "Robot Inspector",
                 "Semantics Schema Editor", "Isaac Sim Assets", "Simulation Data Visualizer"]:
        w = ui.Workspace.get_window(name)
        if w:
            w.visible = False

    C_BG, C_TXT, C_DIM, C_ACC, C_OK = 0xFF1E1812, 0xFFFFFFFF, 0xFFC3B7AA, 0xFF78DCFF, 0xFF8CE6A0
    win = ui.Window("Perception  -  RealSense RGB-D + RT-DETR", width=660, height=1000)
    provider = ui.ByteImageProvider()
    with win.frame:
        with ui.ZStack():
            ui.Rectangle(style={"background_color": C_BG})
            with ui.VStack(spacing=6, style={"margin": 8}):
                ui.Label("RealSense view  |  RT-DETR detections validated in 3D",
                         height=24, style={"font_size": 18, "color": C_ACC})
                ui.ImageWithProvider(provider, height=360,
                                     fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                ui.Label("green: box   magenta: pill   red: rejected by the depth check",
                         height=20, style={"font_size": 14, "color": C_DIM})
                ui_state["phase"] = ui.Label("", height=34, style={"font_size": 26, "color": C_TXT})
                ui_state["counts"] = ui.Label("", height=26, style={"font_size": 20, "color": C_OK})
                ui_state["detail"] = ui.Label("", height=24, word_wrap=True, style={"font_size": 16, "color": C_DIM})
                ui.Label("Last perception", height=24, style={"font_size": 20, "color": C_TXT})
                ui_state["perc"] = ui.Label("", height=150, word_wrap=True,
                                            alignment=ui.Alignment.LEFT_TOP, style={"font_size": 15, "color": C_OK})
                ui.Label("Robot", height=24, style={"font_size": 20, "color": C_TXT})
                ui_state["robot"] = ui.Label("", word_wrap=True, alignment=ui.Alignment.LEFT_TOP,
                                             style={"font_size": 15, "color": C_DIM})
    app_utils.update_app(steps=3)
    vp_win = ui.Workspace.get_window("Viewport")
    if vp_win:
        win.dock_in(vp_win, ui.DockPosition.RIGHT, 0.34)
    app_utils.update_app(steps=3)


def refresh_ui():
    if not LIVE:
        return
    ui_state["phase"].text = status["phase"]
    ui_state["counts"].text = (f"cycle {status['cycle']}   |   pills in bin: {status['pills']}/{len(scene['pills'])}"
                               f"   |   boxes in tote: {status['boxes']}/{len(scene['boxes'])}")
    ui_state["detail"].text = status["detail"]
    ui_state["perc"].text = "\n".join(panel["perception"][:9])
    ui_state["robot"].text = "\n".join(panel["robot"][-3:])


def use_camera(path):
    """Live viewport shot: close-up while pills are picked (they are 8 mm), wide otherwise."""
    if LIVE and vp.camera_path != path:
        vp.camera_path = path


def set_phase(p, detail=None):
    status["phase"] = p
    if detail is not None:
        status["detail"] = detail
    refresh_ui()


def show_image(img):
    last_vis["img"] = img.resize((640, 360))
    if LIVE:
        rgba = np.asarray(last_vis["img"].convert("RGBA"), dtype=np.uint8)
        try:
            provider.set_data_array(np.ascontiguousarray(rgba), [rgba.shape[1], rgba.shape[0]])
        except Exception:
            provider.set_bytes_data(rgba.flatten().tolist(), [rgba.shape[1], rgba.shape[0]])


# ------------------------------------------------------------------ headless composite (fallback video)
frame_idx = [0]
step_count = [0]


def record_frame():
    data = pres_rgb.get_data()
    if data is None or getattr(data, "ndim", 0) != 3:
        return
    scene_img = Image.fromarray(np.asarray(data)[:, :, :3])
    img = Image.new("RGB", (1920, 720), (18, 22, 30))
    img.paste(scene_img, (0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, 1280, 64], fill=(15, 20, 30, 190))
    d.text((16, 8), status["phase"], font=FONT, fill=(255, 255, 255))
    d.text((16, 38), f"cycle {status['cycle']} | pills in bin {status['pills']} | boxes in tote {status['boxes']}"
           f" | {status['detail']}", font=FONT_S, fill=(200, 220, 255))
    if last_vis["img"] is not None:
        img.paste(last_vis["img"], (1280, 64))
    y = 440
    for line in panel["perception"][:7] + [""] + panel["robot"][-3:]:
        d.text((1296, y), line, font=FONT_S, fill=(190, 230, 200))
        y += 22
    img.save(os.path.join(OUT, "frames", f"f_{frame_idx[0]:06d}.jpg"), quality=90)
    frame_idx[0] += 1


def step(n=1):
    for _ in range(n):
        app_utils.update_app(steps=1)
        step_count[0] += 1
        if pres_rgb is not None and step_count[0] % 2 == 0:
            record_frame()


# ------------------------------------------------------------------ detector process
detector = subprocess.Popen([VENV_PY, os.path.join(HERE, "rtdetr_detect.py"), "--server"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                            env={"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/home/vishal")}, cwd=HERE)
log(f"RT-DETR detector: {detector.stdout.readline().strip()}")


# ------------------------------------------------------------------ robot helpers
def ee_pose():
    _, p, q = robot.get_current_state()
    return p[0].astype(float), q[0].astype(float)


def joints():
    return robot.get_dof_positions().numpy()[0].astype(float)


def move_joints(q_goal, seconds=1.8):
    q0 = joints()[:7]
    q_goal = np.asarray(q_goal, float)
    n = max(int(seconds / DT), 10)
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * math.cos(math.pi * i / n)
        robot.set_dof_position_targets((q0 + s * (q_goal - q0)).reshape(1, -1), dof_indices=list(range(7)))
        step()
    step(15)


def move_cartesian(goal_p, goal_q, speed=0.25, tol=0.003, settle_max=150):
    p0, q0 = ee_pose()
    goal_p = np.asarray(goal_p, float)
    n = max(int(max(float(np.linalg.norm(goal_p - p0)) / speed, quat_angle(q0, goal_q) / 90.0) / DT), 15)
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * math.cos(math.pi * i / n)
        robot.set_end_effector_pose(position=p0 + s * (goal_p - p0), orientation=slerp(q0, goal_q, s))
        step()
    ok = 0
    for _ in range(settle_max):
        robot.set_end_effector_pose(position=goal_p, orientation=goal_q)
        step()
        p, q = ee_pose()
        if np.linalg.norm(p - goal_p) < tol and quat_angle(q, goal_q) < 2.0:
            ok += 1
            if ok >= 8:
                break
        else:
            ok = 0
    return float(np.linalg.norm(ee_pose()[0] - goal_p))


def vacuum_on():
    vacuum.close_gripper(GRIPPER)
    step(20)
    held = list(vacuum.get_gripped_objects(GRIPPER))
    return held


def vacuum_off(steps=40):
    vacuum.open_gripper(GRIPPER)
    step(steps)


def hand_z_for_contact(surface_z):
    return surface_z + S.TOOL_TIP + 0.001


# ------------------------------------------------------------------ perception
def world_pos(path):
    t = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    return np.array([t[0], t[1], t[2]])


def draw_validated(rgb, boxes, pills, rejected):
    img = ImageEnhance.Contrast(Image.fromarray(rgb)).enhance(1.15)
    d = ImageDraw.Draw(img, "RGBA")
    for r in rejected:
        x1, y1, x2, y2 = r["xyxy"]
        d.rectangle([x1, y1, x2, y2], outline=(235, 60, 60, 170), width=2)
    for p in pills:
        x1, y1, x2, y2 = p["xyxy"]
        d.ellipse([x1 - 9, y1 - 9, x2 + 9, y2 + 9], outline=(235, 60, 235), width=4)
        d.text((x2 + 10, y1 - 12), "pill (tracked)" if p.get("tracked") else f"pill {p['score']:.2f}",
               font=FONT_S, fill=(210, 30, 210))
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        d.rectangle([x1, y1, x2, y2], outline=(40, 210, 90), width=5)
        d.text((x1 + 4, y1 + 4), f"box {b['score']:.2f}", font=FONT, fill=(20, 150, 60))
    return img


def perceive(tag):
    set_phase("PERCEPTION: RT-DETR + depth")
    step(10)
    rgb = np.asarray(ann_rgb.get_data())[:, :, :3]
    depth = np.asarray(ann_depth.get_data()).squeeze()
    img_path = os.path.join(OUT, "perception", f"{tag}_rgb.png")
    json_path = os.path.join(OUT, "perception", f"{tag}_det.json")
    Image.fromarray(rgb).save(img_path)
    detector.stdin.write(f"{img_path} {json_path} {os.path.join(OUT, 'perception', tag + '_raw.png')} "
                         f"{BOX_THRESH} {PILL_THRESH}\n")
    detector.stdin.flush()
    reply = detector.stdout.readline().strip()
    if not reply.startswith("OK"):
        log(f"detector error: {reply}")
        return [], []
    dets = json.load(open(json_path))["detections"]
    M = S.camera_to_world(scene["camera_path"])
    boxes, pills, rejected = [], [], []
    for d in dets:
        if d["label"] == "box":
            L = localize_box(d, depth, K, M, side_range=(0.07, 0.13))
            if L and BOX_ZONE[0][0] < L["center"][0] < BOX_ZONE[0][1] and BOX_ZONE[1][0] < L["center"][1] < BOX_ZONE[1][1]:
                boxes.append(L)
            else:
                rejected.append(d)
        else:
            L = localize_pill(d, depth, K, M, MAT_TOP, region=PILL_REGION)
            if L is None:
                rejected.append(d)
            elif all(math.hypot(L["center"][0] - q["center"][0], L["center"][1] - q["center"][1]) > 0.008
                     for q in pills):
                pills.append(L)
    # tracking: a pill validated in an earlier view that the detector misses now (low
    # confidence once it is isolated) is kept if its 3 mm depth bump is still there
    n_tracked = 0
    for t in tracked_pills:
        if any(math.hypot(t[0] - q["center"][0], t[1] - q["center"][1]) < 0.008 for q in pills):
            continue
        u, v = project((t[0], t[1], MAT_TOP + 0.003), K, M)
        L = localize_pill({"xyxy": [u - 5, v - 5, u + 5, v + 5], "score": 0.0}, depth, K, M, MAT_TOP,
                          region=PILL_REGION)
        if L and math.hypot(L["center"][0] - t[0], L["center"][1] - t[1]) < 0.006:
            L["tracked"] = True
            pills.append(L)
            n_tracked += 1
    tracked_pills[:] = [(p["center"][0], p["center"][1]) for p in pills]
    # perception error vs ground truth (reporting only)
    gt_b = [world_pos(b["path"]) for b in scene["boxes"]]
    gt_p = [world_pos(p["path"]) for p in scene["pills"]]
    for L, gts in [(b, gt_b) for b in boxes] + [(p, gt_p) for p in pills]:
        L["err_mm"] = min(math.hypot(g[0] - L["center"][0], g[1] - L["center"][1]) for g in gts) * 1000
    panel["perception"] = [f"{len(pills)} pills, {len(boxes)} boxes validated; {len(rejected)} candidates rejected"
                           + (f"; {n_tracked} pill(s) tracked by depth" if n_tracked else "")]
    panel["perception"] += [(f"pill trk " if p.get("tracked") else f"pill {p['score']:.2f}") + f"  ({p['center'][0]:.3f}, {p['center'][1]:.3f}) m   err {p['err_mm']:.1f} mm"
                            for p in pills]
    panel["perception"] += [f"box  {b['score']:.2f}  ({b['center'][0]:.3f}, {b['center'][1]:.3f}) m  yaw {b['yaw_deg']:+.0f}  "
                            f"err {b['err_mm']:.1f} mm" for b in boxes]
    for line in panel["perception"]:
        log("  " + line)
    vis = draw_validated(rgb, boxes, pills, rejected)
    vis.save(os.path.join(OUT, "perception", f"{tag}_validated.png"))
    show_image(vis)
    refresh_ui()
    return pills, boxes


# ------------------------------------------------------------------ tasks
def pick_pill(p):
    x, y, top = p["center"]
    src = "tracked by depth" if p.get("tracked") else f"pill conf {p['score']:.2f}"
    set_phase("PICK PILL", f"{src} at ({x:.3f}, {y:.3f}) m, top {top * 1000:.1f} mm")
    move_joints(READY_Q, 1.4)
    q = down_quat(0.0)
    move_cartesian([x, y, hand_z_for_contact(top) + 0.08], q)
    set_phase("DESCEND: vacuum cup to pill")
    err = move_cartesian([x, y, hand_z_for_contact(top)], q, speed=0.05, tol=0.0015)
    set_phase("VACUUM ON")
    held = vacuum_on()
    msg = f"cycle {status['cycle']}: pill contact err {err * 1000:.1f} mm, vacuum {'HOLDING' if held else 'NO SEAL'}"
    log(msg)
    panel["robot"].append(msg)
    if not held:
        vacuum_off(10)
        move_cartesian([x, y, hand_z_for_contact(top) + 0.08], q)
        return False
    set_phase("LIFT PILL")
    move_cartesian([x, y, 0.26], q, speed=0.18)
    set_phase("TRANSFER TO PILL BIN")
    bx, by = S.BIN_CENTER
    move_cartesian([bx, by, 0.34], q)
    move_cartesian([bx, by, 0.215 + S.TOOL_TIP], q)
    set_phase("RELEASE INTO BIN")
    vacuum_off(45)
    move_cartesian([bx, by, 0.34], q)
    return True


def pick_box(b, slot):
    x, y, _ = b["center"]
    top = b["top_z"]
    yaw = b["yaw_deg"]
    set_phase("PICK BOX", f"box conf {b['score']:.2f} at ({x:.3f}, {y:.3f}) m, yaw {yaw:+.0f} deg")
    move_joints(READY_Q, 1.4)
    qg = down_quat(yaw)
    q0 = down_quat(0.0)
    move_cartesian([x, y, hand_z_for_contact(top) + 0.10], qg)
    set_phase("DESCEND: vacuum cup to box top")
    err = move_cartesian([x, y, hand_z_for_contact(top)], qg, speed=0.08, tol=0.002)
    set_phase("VACUUM ON")
    held = vacuum_on()
    msg = f"cycle {status['cycle']}: box contact err {err * 1000:.1f} mm, vacuum {'HOLDING' if held else 'NO SEAL'}"
    log(msg)
    panel["robot"].append(msg)
    if not held:
        vacuum_off(10)
        move_cartesian([x, y, hand_z_for_contact(top) + 0.10], qg)
        return False
    set_phase("LIFT BOX")
    move_cartesian([x, y, CARRY_Z], qg, speed=0.18)
    set_phase("TRANSFER TO TOTE (squaring up)")
    sx, sy = S.TOTE_SLOTS[slot % len(S.TOTE_SLOTS)]
    move_cartesian([0.30, 0.30, CARRY_Z], slerp(qg, q0, 0.5))
    move_cartesian([sx, sy, CARRY_Z], q0)
    move_cartesian([sx, sy, TOTE_FLOOR + S.BOX_SIDE + 0.012 + S.TOOL_TIP], q0, speed=0.10)
    set_phase("RELEASE INTO TOTE")
    vacuum_off(45)
    move_cartesian([sx, sy, CARRY_Z], q0)
    return True


def count_delivered():
    (bx0, by0, _), (bx1, by1, _) = scene["bin_range"]
    (tx0, ty0, _), (tx1, ty1, _) = scene["tote_range"]
    pills = sum(1 for p in scene["pills"] if bx0 < world_pos(p["path"])[0] < bx1 and by0 < world_pos(p["path"])[1] < by1)
    boxes = sum(1 for b in scene["boxes"] if tx0 < world_pos(b["path"])[0] < tx1 and ty0 < world_pos(b["path"])[1] < ty1)
    return pills, boxes


try:
    omni.timeline.get_timeline_interface().play()
    step(2)
    q_obs = OBSERVE_Q + [0.0, 0.0]
    robot.set_dof_positions(q_obs)
    robot.set_dof_position_targets(q_obs)
    set_phase("SCENE READY", f"{len(scene['pills'])} pills on the mat, {len(scene['boxes'])} boxes on the table")
    for i in range(600):   # let the warehouse assets stream in and the sensor render
        step()
        if i > 90 and getattr(ann_rgb.get_data(), "ndim", 0) == 3:
            break
    step(30)
    log(f"scene ready: seed={SEED}, {len(scene['pills'])} pills, {len(scene['boxes'])} boxes, out={OUT}")

    t0 = time.time()
    cycle, empty, fails = 0, 0, 0
    while cycle < 20 and fails < 6:
        set_phase("OBSERVE")
        move_joints(OBSERVE_Q, 1.6)
        pills, boxes = perceive(f"view{cycle + 1:02d}")
        if not pills and not boxes:
            empty += 1
            if empty >= 2:
                break
            continue
        empty = 0
        cycle += 1
        status["cycle"] = cycle
        if pills:
            target = min(pills, key=lambda p: math.hypot(p["center"][0], p["center"][1]))
            use_camera(scene["pill_camera_path"])
            ok = pick_pill(target)
        else:
            target = min(boxes, key=lambda b: math.hypot(b["center"][0], b["center"][1]))
            use_camera(scene["presentation_camera_path"])
            ok = pick_box(target, status["boxes"])
        step(30)
        status["pills"], status["boxes"] = count_delivered()
        if ok:
            panel["robot"].append(f"cycle {cycle}: delivered -> pills in bin {status['pills']}, boxes in tote {status['boxes']}")
            log(f"CYCLE {cycle}: OK  pills_in_bin={status['pills']} boxes_in_tote={status['boxes']}")
        else:
            fails += 1
            log(f"CYCLE {cycle}: retry")
        refresh_ui()

    use_camera(scene["presentation_camera_path"])
    move_joints(OBSERVE_Q, 1.6)
    status["pills"], status["boxes"] = count_delivered()
    set_phase("DONE", f"{status['pills']}/{len(scene['pills'])} pills in the bin, "
                      f"{status['boxes']}/{len(scene['boxes'])} boxes in the tote")
    step(90)
    summary = {"seed": SEED, "pills": len(scene["pills"]), "pills_in_bin": status["pills"],
               "boxes": len(scene["boxes"]), "boxes_in_tote": status["boxes"], "cycles": cycle,
               "failed_attempts": fails, "wall_time_s": round(time.time() - t0, 1)}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    log(f"DEMO COMPLETE: {json.dumps(summary)}")
except Exception:
    log("DEMO FAILED:\n" + traceback.format_exc())
finally:
    try:
        detector.stdin.close()
        detector.terminate()
    except Exception:
        pass
    LOG.close()
    if QUIT_WHEN_DONE and LIVE:   # hold the final frame while the screen recorder stops
        t_end = time.time() + 5.0
        while time.time() < t_end:
            app_utils.update_app(steps=1)
    if QUIT_WHEN_DONE:
        import carb.settings
        carb.settings.get_settings().set("/app/file/ignoreUnsavedOnExit", True)   # no "save stage?" prompt
        omni.kit.app.get_app().post_quit()
