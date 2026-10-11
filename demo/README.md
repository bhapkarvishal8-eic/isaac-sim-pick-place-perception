# V2 demo — warehouse cell, pills + boxes, vacuum tool (2026-09-27)

The robot now works on a packing table inside NVIDIA's warehouse environment. The same RT-DETR model detects
**pills and boxes**: pills are picked into a KLT bin, and 10 cm boxes are picked into a tote. No model was retrained.

```bash
cd demo
./run_v2_demo.sh --record          # live Isaac Sim window + real-time screen recording -> screen_recording.mp4
./run_v2_demo.sh                   # live, no recording (window stays open)
./run_v2_demo.sh --headless        # no window; composite demo.mp4 rendered frame by frame
./run_v2_demo.sh --record --seed 5 # different pill / box layout
```
Outputs go to `logs/final_demo/v2_<mode>_seed<N>/`. That folder holds `screen_recording.mp4`, which is an
unedited real-time capture of the Isaac Sim window: the 3D view is on the left and the in-app "Perception" panel is on the right,
and the viewport FPS counter is left on. It also holds `demo_log.txt`, `summary.json` and `perception/`.

| Piece | What it is | File |
|---|---|---|
| Scene | `Simple_Warehouse/full_warehouse.usd`, `PackingTable` (grey tote reused from it), `KLT_Bin`, pill mat, project `box.usd` (10 cm) and `pill.usd` | `demo_scene_v2.py` |
| Tool | vacuum cup on the Franka flange, using an Isaac Sim **Surface Gripper**. The grasp is confirmed by the gripper's attach status, which is the sim equivalent of a vacuum-pressure switch | `demo_scene_v2.py: add_vacuum_tool` |
| Detection | RT-DETR r18 (Phase 3 SDG, classes `pill`/`box`). Boxes come from one pass. Pills come from 1.0/0.75/0.6 downscaled passes, because pills render larger here than the ~5 px they had in the training frames | `rtdetr_detect.py: detect_multiscale` |
| Localization | Boxes: the top face is found from depth, then the largest cluster gives center, height, yaw and a size check. Pills: the 3 mm depth bump on the mat; no bump means the detection is rejected. A pill validated earlier is kept as "tracked" if its bump is still there, because isolated pills score lower | `localize.py`, `pick_place_v2.py: perceive` |
| Control | Franka differential IK on real PhysX PD drives. Pills go first (nearest first) to the bin, then boxes are squared up to yaw 0 and put in the tote slots. The robot re-observes after every pick | `pick_place_v2.py` |

**Measured results (2026-09-27): 72/72 objects placed over 8 runs in 4 layouts, 0 failed picks**
(seed 11: 1 headless and 4 live GUI runs; seeds 1–3 headless). Each run placed 5/5 pills in the bin and 4/4 boxes in the tote.
- Pill localization error 0.5–1.0 mm; box error 5.6–7.9 mm, yaw within ~2°. Vacuum-cup contact error 0.9–3.2 mm.
- Depth tracking of pills was needed in every seed: an isolated pill often scores below the 0.22 threshold after its
  neighbours are picked. Every false candidate (for example, the KLT bin detected as a "box") was rejected by the 3D check.
- One run takes about 6–7 min. The live window runs at 32–50 FPS on the RTX 5070 Ti.
- LinkedIn videos in `logs/final_demo/linkedin/`: `v2_warehouse_pick_place_realtime.mp4` (full 6 min,
  unedited) and `v2_warehouse_pick_place_90s.mp4` (86 s cut; real-time open and close, 3×/7× middle, speed labelled on screen).
  The post text is in `linkedin_post_draft.md`.

Why a vacuum tool: pills are 8 mm × 3 mm and the boxes are 10 cm. A parallel gripper can't do both (the Franka hand opens
8 cm), and warehouse pick cells use suction for exactly this mix.

Why only pills and boxes: the Phase 3 dataset, and so the detector, has only these two classes. The shelf was scene
geometry and was never a class.

# V1 demo — perception-driven pick & place (Isaac Sim 6.0.1)

A Franka Panda sees cardboard boxes with a RealSense-style RGB-D camera, detects
them with the RT-DETR model trained on this project's Phase 3 synthetic data,
localizes each one in 3D from depth, physically grasps it, and stacks it on a
place station — repeating until the pick area is empty. No model was retrained
for this demo.

## Run it

```bash
cd demo
./run_final_demo.sh                # live, in the Isaac Sim window (stays open when done)
./run_final_demo.sh --video        # headless, writes demo.mp4 with status + detection overlays
./run_final_demo.sh --seed 3       # different random box positions / orientations
```

Outputs go to `logs/final_demo/<live|video>_seed<N>/`:
`demo_log.txt` (every perception result and grasp), `summary.json`,
`perception/` (camera frames, raw RT-DETR output, validated detections), `demo.mp4`.

## What happens each cycle

| Step | What runs | File |
|---|---|---|
| Observe | arm parks out of the camera view; camera renders RGB + depth (1280×720, 66° HFOV) | `pick_place_final.py` |
| Detect | RT-DETR r18 (`exported_models/rtdetrv2_r18vd_pillbox_epoch19.onnx`), same preprocessing as training | `rtdetr_detect.py` |
| Localize | depth pixels in each box detection → world points → top face → center, height, yaw; detections that aren't box-shaped in 3D are rejected | `localize.py` |
| Pick | yaw-aligned top-down grasp; real PhysX PD drives + differential IK (damped least squares) on smooth Cartesian waypoints; grasp verified by finger width | `pick_place_final.py` |
| Place | carry, square the box up (yaw 0), release on the next station slot, re-observe | `pick_place_final.py` |

Ground-truth poses are read only to log the perception error and confirm placement — control uses perception output only.

## Measured results (2026-09-26)

- **Pick & place: 40/40 boxes placed across 10 runs** (seeds 1–5 headless, seed 7 ×3 video,
  seeds 3 and 5 live in the GUI), 0 missed grasps, 0 failed placements.
  Per-seed log: `logs/final_demo/robustness_summary.txt`.
- Detection: every box found in every observation, confidence 0.90–0.98; pills detected (display only).
  The detector's one recurring false positive (a large "box" over the table edge) is rejected by the 3D check every time.
- Localization error vs ground truth: 3.0–4.8 mm position, <2° yaw.
- IK tracking at the grasp pose: 1.3–1.6 mm. Fingers close to 56.6 mm on the 55 mm boxes (real grasp).
- One run (4 boxes) takes ~2–2.5 min live; the video is ~2 min at 30 fps.

## Video layout

`demo.mp4` is 1920×720: the 3D scene on the left (phase + counters in the header), and on the right the
RealSense view with 3D-validated RT-DETR detections, the latest perception numbers (confidence, position,
yaw, error vs ground truth) and the robot's grasp/placement results.

## Why this is a new cell instead of `poc_workspace.usd`

The earlier ROS2/MoveIt stack (`run_demo.sh`) reached the pre-grasp pose and then
failed every later motion with MoveIt `START_STATE_INVALID`: the simulated
`panda_joint6` settled at ~3.80 rad, above MoveIt's 3.7525 rad limit. Underneath
that were structural problems that no quick fix would remove:

1. **The Franka+Robotiq articulation never ran under PD control** (`panda_joint1`
   divergence, never root-caused), so that bridge teleported joints every frame
   and faked grasps by re-parenting the object to the hand. Here the standard
   Franka asset with its own hand runs on real PD drives (the same setup as
   NVIDIA's shipped pick-and-place example) and grasps are real friction grasps.
2. **The boxes (10/15/20 cm) are wider than any parallel gripper opens**
   (Franka hand 8 cm, Robotiq 2F-85 8.5 cm). The demo reuses the project's own
   `fixtures/box.usd` (same cardboard material) at 5.5 cm.
3. **Objects sat 12–32 cm from the robot base and the shelf overlapped the base.**
   Objects are now in the Franka's comfortable top-down workspace.
4. **`fixtures/shelf.usd` has its transform ops in the wrong order** (scale
   applied after translate), which collapses its panels into a flat shape. The
   place station here is built from plain, correctly transformed cubes.

The camera keeps the training camera's 66° field of view and oblique look-down
angle, so the trained detector works without retraining (validated above).

## Not in this demo

- FoundationPose / CenterPose: a top-down grasp of a cube only needs position +
  yaw, which RGB-D localization gives directly. The CenterPose pill model from
  the Sep 11 retrain was never exported to ONNX/TensorRT.
- Picking pills: 8 mm × 3 mm pills are detected and shown, not picked — they
  are too small for reliable simulated friction grasping.
- ROS2 / MoveIt2: everything runs inside one Isaac Sim process, with the detector in
  a side process.
