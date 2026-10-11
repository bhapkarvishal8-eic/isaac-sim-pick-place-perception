"""FINAL DEMO: perception-driven pick-and-place in Isaac Sim 6.0.1 (run via kit --exec).

Loop, until the pick area is empty:
  1. OBSERVE  arm parks out of the camera view; RealSense-style camera renders RGB + depth
  2. DETECT   RT-DETR (the model trained on this project's Phase 3 SDG data, epoch 19)
              runs on the RGB frame (rtdetr_detect.py server, onnxruntime)
  3. LOCALIZE each "box" detection is lifted to 3D from the depth image
              (center + height + yaw, localize.py); false positives rejected
  4. PICK     top-down grasp aligned to the estimated yaw -- Franka on real PhysX PD
              drives, differential IK (damped least squares) tracking smooth
              Cartesian waypoints; real friction grasp, verified by finger width
  5. PLACE    box carried to the next free slot on the place station, squared up
              (yaw 0) and released

Ground-truth object poses are read ONLY to report perception error and placement
success in the log/video; control uses perception output exclusively.

Environment variables:
  DEMO_HEADLESS=1   quit when finished (default: keep the window open)
  DEMO_RECORD=1     save presentation-camera frames for the video (default 1)
  DEMO_SEED=<int>   randomize box positions/yaws (default 7)
  DEMO_OUT=<dir>    output directory (default /home/vishal/poc/logs/final_demo/latest)
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

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.robot.experimental.manipulators.examples", True)

from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.experimental.manipulators.examples.franka.franka import Franka
import omni.replicator.core as rep
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from pxr import Usd, UsdGeom

import demo_scene
from localize import localize_box

HEADLESS = os.environ.get("DEMO_HEADLESS", "0") == "1"
RECORD = os.environ.get("DEMO_RECORD", "1") == "1"
SEED = int(os.environ.get("DEMO_SEED", "7"))
OUT = os.environ.get("DEMO_OUT", "/home/vishal/poc/logs/final_demo/latest")
VENV_PY = "/home/vishal/poc/stage1_perception_arm/phase4_perception_training/rtdetr_venv/bin/python"

DT = 1.0 / 60.0
OBSERVE_Q = [1.25, -0.45, 0.0, -2.3, 0.0, 1.9, 0.785]
READY_Q = [0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741]
HAND_TO_GRASP = 0.100          # panda_hand origin above the box center at grasp
APPROACH_CLEARANCE = 0.14      # above grasp height
CARRY_Z = 0.34                 # hand height while carrying
PICK_ZONE = ((0.30, 0.72), (-0.36, 0.30))   # x range, y range accepted as pickable
DET_THRESH = 0.5

os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
os.makedirs(os.path.join(OUT, "perception"), exist_ok=True)
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
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


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


def grasp_quat(yaw_deg):
    """Top-down hand orientation rotated about world Z by yaw."""
    a = math.radians(yaw_deg)
    return qmul(np.array([math.cos(a / 2), 0, 0, math.sin(a / 2)]), np.array([0.0, 1.0, 0.0, 0.0]))


def quat_angle(a, b):
    return 2 * math.degrees(math.acos(min(1.0, abs(float(np.dot(a, b))))))


# ------------------------------------------------------------------ scene
omni.usd.get_context().new_stage()
app_utils.update_app(steps=2)
SimulationManager.set_physics_sim_device("cpu")
scene = demo_scene.build(seed=SEED)
robot = Franka(robot_path="/World/robot", create_robot=True)
stage = omni.usd.get_context().get_stage()

sensor_rp = rep.create.render_product(scene["camera_path"], demo_scene.CAM_RES)
ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
ann_rgb.attach([sensor_rp])
ann_depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
ann_depth.attach([sensor_rp])
pres_rgb = None
if RECORD:
    pres_rp = rep.create.render_product(scene["presentation_camera_path"], (1280, 720))
    pres_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    pres_rgb.attach([pres_rp])

if not HEADLESS:
    try:
        from omni.kit.viewport.utility import get_active_viewport
        vp = get_active_viewport()
        vp.camera_path = scene["presentation_camera_path"]
        # hide camera gizmos: the RealSense camera sits between the viewer and the cell
        from omni.kit.viewport.actions.actions import toggle_camera_visibility
        toggle_camera_visibility(vp, visible=False)
    except Exception as e:  # cosmetic only
        log(f"viewport camera not set: {e}")

K = demo_scene.camera_intrinsics()

# ------------------------------------------------------------------ detector process
det_env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/home/vishal")}
detector = subprocess.Popen([VENV_PY, os.path.join(HERE, "rtdetr_detect.py"), "--server"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, env=det_env, cwd=HERE)
ready_line = detector.stdout.readline().strip()
log(f"RT-DETR detector: {ready_line}")


# ------------------------------------------------------------------ recording / overlay
status = {"phase": "INIT", "cycle": 0, "placed": 0, "detail": ""}
panel = {"perception": [], "robot": []}
last_vis = {"img": None}
frame_idx = [0]
step_count = [0]
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
except Exception:
    FONT = FONT_S = ImageFont.load_default()


def record_frame():
    """1920x720 frame: 3D scene (left 1280) + perception/status side panel (right 640)."""
    if pres_rgb is None:
        return
    data = pres_rgb.get_data()
    if data is None or getattr(data, "size", 0) == 0:
        return
    scene_img = Image.fromarray(np.asarray(data)[:, :, :3])
    # the bright, white cell renders washed out; add contrast for the video only
    scene_img = ImageEnhance.Brightness(ImageEnhance.Contrast(scene_img).enhance(1.45)).enhance(0.9)
    img = Image.new("RGB", (1920, 720), (18, 22, 30))
    img.paste(scene_img, (0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, 1280, 64], fill=(15, 20, 30, 190))
    d.text((16, 8), f"Perception-driven pick & place  |  {status['phase']}", font=FONT, fill=(255, 255, 255))
    d.text((16, 38), f"cycle {status['cycle']}   |   boxes placed: {status['placed']} / {len(scene['boxes'])}"
           f"   |   {status['detail']}", font=FONT_S, fill=(200, 220, 255))
    # side panel
    x0 = 1280
    d.text((x0 + 16, 10), "RealSense RGB-D  +  RT-DETR", font=FONT, fill=(255, 220, 120))
    d.text((x0 + 16, 38), "green = detection validated in 3D (depth), red = rejected", font=FONT_S,
           fill=(170, 180, 195))
    if last_vis["img"] is not None:
        img.paste(last_vis["img"], (x0, 64))
    y = 64 + 360 + 14
    d.text((x0 + 16, y), "Last perception", font=FONT, fill=(255, 255, 255))
    y += 32
    for line in panel["perception"][:6]:
        d.text((x0 + 16, y), line, font=FONT_S, fill=(190, 230, 200))
        y += 23
    y = 720 - 3 * 24 - 12
    for line in panel["robot"][-3:]:
        d.text((x0 + 16, y), line, font=FONT_S, fill=(200, 215, 255))
        y += 24
    img.save(os.path.join(OUT, "frames", f"f_{frame_idx[0]:06d}.jpg"), quality=90)
    frame_idx[0] += 1


def step(n=1):
    for _ in range(n):
        app_utils.update_app(steps=1)
        step_count[0] += 1
        if RECORD and step_count[0] % 2 == 0:
            record_frame()


# ------------------------------------------------------------------ robot helpers
def ee_pose():
    _, p, q = robot.get_current_state()
    return p[0].astype(float), q[0].astype(float)


def joints():
    return robot.get_dof_positions().numpy()[0].astype(float)


def move_joints(q_goal, seconds=2.0):
    q0 = joints()[:7]
    q_goal = np.asarray(q_goal, float)
    n = max(int(seconds / DT), 10)
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * math.cos(math.pi * i / n)   # smooth ease-in/out
        robot.set_dof_position_targets((q0 + s * (q_goal - q0)).reshape(1, -1), dof_indices=list(range(7)))
        step()
    step(20)


def move_cartesian(goal_p, goal_q, speed=0.22, tol=0.004, settle_max=150):
    """Track a smooth straight-line waypoint path with differential IK, then settle."""
    p0, q0 = ee_pose()
    goal_p = np.asarray(goal_p, float)
    dist = float(np.linalg.norm(goal_p - p0))
    ang = quat_angle(q0, goal_q)
    n = max(int(max(dist / speed, ang / 90.0) / DT), 15)
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * math.cos(math.pi * i / n)
        robot.set_end_effector_pose(position=p0 + s * (goal_p - p0), orientation=slerp(q0, goal_q, s))
        step()
    ok_count = 0
    for _ in range(settle_max):
        robot.set_end_effector_pose(position=goal_p, orientation=goal_q)
        step()
        p, q = ee_pose()
        if np.linalg.norm(p - goal_p) < tol and quat_angle(q, goal_q) < 2.0:
            ok_count += 1
            if ok_count >= 8:
                break
        else:
            ok_count = 0
    p, q = ee_pose()
    err = float(np.linalg.norm(p - goal_p))
    return err < 0.01, err


def gripper(close, steps=45):
    if close:
        robot.close_gripper()
    else:
        robot.open_gripper()
    step(steps)
    return float(np.sum(joints()[7:9]))     # finger opening (m)


# ------------------------------------------------------------------ perception
def gt_box_positions():
    out = {}
    for b in scene["boxes"]:
        m = UsdGeom.Xformable(stage.GetPrimAtPath(b["path"])).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = m.ExtractTranslation()
        out[b["path"]] = np.array([t[0], t[1], t[2]])
    return out


def perceive(tag):
    status["phase"] = "PERCEPTION"
    status["detail"] = "RT-DETR + depth localization"
    step(12)   # let the sensor frame refresh
    rgb = np.asarray(ann_rgb.get_data())[:, :, :3]
    depth = np.asarray(ann_depth.get_data()).squeeze()
    img_path = os.path.join(OUT, "perception", f"{tag}_rgb.png")
    json_path = os.path.join(OUT, "perception", f"{tag}_det.json")
    vis_path = os.path.join(OUT, "perception", f"{tag}_det.png")
    Image.fromarray(rgb).save(img_path)
    detector.stdin.write(f"{img_path} {json_path} {vis_path} {DET_THRESH}\n")
    detector.stdin.flush()
    reply = detector.stdout.readline().strip()
    if not reply.startswith("OK"):
        log(f"detector error: {reply}")
        return []
    dets = json.load(open(json_path))["detections"]
    cam_to_world = demo_scene.camera_to_world(scene["camera_path"])
    gt = gt_box_positions()
    boxes = []
    rejected = []
    pills = nms([d for d in dets if d["label"] == "pill"])
    n_pill = len(pills)
    for d in dets:
        if d["label"] != "box":
            continue
        loc = localize_box(d, depth, K, cam_to_world)
        if loc is None:
            rejected.append(d)
            log(f"  rejected box detection score={d['score']:.2f} bbox={[round(v) for v in d['xyxy']]} (not box-shaped in 3D)")
            continue
        x, y, _ = loc["center"]
        if not (PICK_ZONE[0][0] < x < PICK_ZONE[0][1] and PICK_ZONE[1][0] < y < PICK_ZONE[1][1]):
            continue
        # perception error vs ground truth (reporting only)
        nearest = min(gt.values(), key=lambda g: np.hypot(g[0] - x, g[1] - y))
        loc["gt_err_mm"] = float(np.hypot(nearest[0] - x, nearest[1] - y) * 1000)
        boxes.append(loc)
        log(f"  box score={loc['score']:.2f} at ({x:.3f},{y:.3f}) h={loc['top_z']:.3f} yaw={loc['yaw_deg']:+.1f}deg "
            f"side={loc['side'] * 1000:.0f}mm  [perception error vs GT: {loc['gt_err_mm']:.1f} mm]")
    log(f"perception[{tag}]: {len(dets)} detections -> {len(boxes)} pickable boxes, {n_pill} pills")
    panel["perception"] = [f"{len(boxes)} boxes validated, {len(rejected)} rejected, {n_pill} pills"] + [
        f"box {b['score']:.2f}  ({b['center'][0]:.3f}, {b['center'][1]:.3f}) m  yaw {b['yaw_deg']:+.0f}deg  err {b['gt_err_mm']:.1f}mm"
        for b in boxes]
    final_vis = draw_validated(rgb, boxes, rejected, pills)
    final_vis.save(os.path.join(OUT, "perception", f"{tag}_validated.png"))
    last_vis["img"] = final_vis.resize((640, 360))
    return boxes


def nms(dets, iou_thr=0.3):
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        return inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    keep = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(iou(d["xyxy"], k["xyxy"]) < iou_thr for k in keep):
            keep.append(d)
    return keep


def draw_validated(rgb, boxes, rejected, pills):
    """Sensor view with RT-DETR detections after 3D validation (what the robot acts on)."""
    img = Image.fromarray(rgb)
    img = ImageEnhance.Contrast(img).enhance(1.3)
    d = ImageDraw.Draw(img, "RGBA")
    for r in rejected:
        x1, y1, x2, y2 = r["xyxy"]
        d.rectangle([x1 + 2, y1 + 2, x2 - 2, y2 - 2], outline=(230, 60, 60, 150), width=2)
        d.text((x1 + 8, y2 - 30), f"box {r['score']:.2f} rejected: not box-shaped in 3D", font=FONT_S,
               fill=(230, 60, 60, 220))
    for p in pills:
        x1, y1, x2, y2 = p["xyxy"]
        d.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=(220, 60, 220), width=2)
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        d.rectangle([x1, y1, x2, y2], outline=(40, 200, 90), width=4)
        d.text((x1, max(0, y1 - 44)), f"box {b['score']:.2f}", font=FONT, fill=(20, 150, 60))
        d.text((x1, max(0, y1 - 20)), f"({b['center'][0]:.3f},{b['center'][1]:.3f}) {b['yaw_deg']:+.0f}deg",
               font=FONT_S, fill=(20, 120, 50))
    if pills:
        d.text((12, 690), f"pills detected: {len(pills)}", font=FONT, fill=(200, 40, 200))
    return img


# ------------------------------------------------------------------ main task
def pick_and_place(box, slot_idx):
    x, y, zc = box["center"]
    yaw = box["yaw_deg"]
    q_grasp = grasp_quat(yaw)
    q_place = grasp_quat(0.0)
    grasp_z = zc + HAND_TO_GRASP
    sx = demo_scene.PLACE_SLOTS_X[slot_idx % len(demo_scene.PLACE_SLOTS_X)]
    sy = demo_scene.STATION_CENTER[1]
    place_z = demo_scene.STATION_SIZE[2] + demo_scene.BOX_SIDE / 2 + HAND_TO_GRASP + 0.006

    status["detail"] = f"target ({x:.3f}, {y:.3f})  yaw {yaw:+.1f} deg  conf {box['score']:.2f}"
    status["phase"] = "MOVE TO READY"
    move_joints(READY_Q, 1.6)
    gripper(False, 10)
    status["phase"] = "APPROACH (above box, yaw-aligned)"
    move_cartesian([x, y, grasp_z + APPROACH_CLEARANCE], q_grasp)
    status["phase"] = "DESCEND TO GRASP"
    ok, err = move_cartesian([x, y, grasp_z], q_grasp, speed=0.08, tol=0.003)
    log(f"  at grasp pose (tracking error {err * 1000:.1f} mm)")
    panel["robot"].append(f"cycle {status['cycle']}: grasp pose reached, IK error {err * 1000:.1f} mm")
    status["phase"] = "CLOSE GRIPPER"
    width = gripper(True, 50)
    held = width > 0.6 * demo_scene.BOX_SIDE
    log(f"  gripper closed, finger opening {width * 1000:.1f} mm -> {'GRASPED' if held else 'MISSED'}")
    panel["robot"].append(f"cycle {status['cycle']}: fingers at {width * 1000:.1f} mm -> {'GRASPED' if held else 'MISSED'}")
    if not held:
        gripper(False, 20)
        move_cartesian([x, y, grasp_z + APPROACH_CLEARANCE], q_grasp)
        return False
    status["phase"] = "LIFT"
    move_cartesian([x, y, CARRY_Z], q_grasp, speed=0.15)
    status["phase"] = "TRANSPORT TO STATION"
    move_cartesian([0.30, 0.25, CARRY_Z], slerp(q_grasp, q_place, 0.5))
    move_cartesian([sx, sy, place_z + 0.10], q_place)
    status["phase"] = "PLACE"
    move_cartesian([sx, sy, place_z], q_place, speed=0.06, tol=0.003)
    width_before_release = float(np.sum(joints()[7:9]))
    status["phase"] = "RELEASE"
    gripper(False, 40)
    status["phase"] = "RETREAT"
    move_cartesian([sx, sy, place_z + 0.12], q_place)
    still_held = width_before_release > 0.6 * demo_scene.BOX_SIDE
    return still_held


def placed_on_station():
    n = 0
    for p in gt_box_positions().values():
        sx, sy = demo_scene.STATION_CENTER
        sw, sd, sh = demo_scene.STATION_SIZE
        if abs(p[0] - sx) < sw / 2 and abs(p[1] - sy) < sd / 2 and p[2] > sh:
            n += 1
    return n


try:
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    step(2)
    robot.reset_to_default_pose()
    q_obs = OBSERVE_Q + [0.04, 0.04]
    robot.set_dof_positions(q_obs)
    robot.set_dof_position_targets(q_obs)
    step(60)
    log(f"scene ready: seed={SEED}, {len(scene['boxes'])} boxes in the pick area, output -> {OUT}")

    t_start = time.time()
    empty_views = 0
    cycle = 0
    while cycle < 10:
        status["phase"] = "OBSERVE"
        move_joints(OBSERVE_Q, 1.8)
        boxes = perceive(f"cycle{cycle + 1:02d}")
        if not boxes:
            empty_views += 1
            if empty_views >= 2:
                break
            continue
        empty_views = 0
        # pick the box closest to the robot first (keeps the path clear of the others)
        box = min(boxes, key=lambda b: math.hypot(b["center"][0], b["center"][1]))
        cycle += 1
        status["cycle"] = cycle
        log(f"CYCLE {cycle}: picking box at ({box['center'][0]:.3f},{box['center'][1]:.3f}) yaw {box['yaw_deg']:+.1f}")
        ok = pick_and_place(box, status["placed"])
        step(30)
        on_station = placed_on_station()
        if ok and on_station > status["placed"]:
            status["placed"] = on_station
            log(f"CYCLE {cycle}: SUCCESS -- box placed on station ({on_station} total)")
            panel["robot"].append(f"cycle {cycle}: box placed on station ({on_station} total)")
        else:
            status["placed"] = on_station
            log(f"CYCLE {cycle}: not placed (will re-perceive and retry)")

    status["phase"] = "DONE"
    status["detail"] = f"pick area empty -- {status['placed']}/{len(scene['boxes'])} boxes on station"
    move_joints(OBSERVE_Q, 1.8)
    step(60)
    summary = {"seed": SEED, "boxes": len(scene["boxes"]), "placed_on_station": status["placed"],
               "cycles": cycle, "wall_time_s": round(time.time() - t_start, 1), "frames": frame_idx[0]}
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
    if HEADLESS:
        omni.kit.app.get_app().post_quit()
