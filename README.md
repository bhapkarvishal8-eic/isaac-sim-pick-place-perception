# Perception-Driven Pick & Place in NVIDIA Isaac Sim (real-time)

**Author:** Vishal Bhapkar · **Period:** Jul – Oct 2026 · **Status:** PoC complete (simulation only)
**Developed on:** Isaac Sim 6.0.1 standalone, Ubuntu
**Hardware:** RTX 5070 Ti · Ryzen 9 5950X · 31 GiB RAM · Ubuntu

---

## 1. Summary

A Franka Panda arm in NVIDIA Isaac Sim 6.0.1 looks at a packing table through a RealSense-style RGB-D camera.
It detects **pills (8 mm × 3 mm) and cartons (10 cm)** with an RT-DETR detector trained only on synthetic data,
localizes each object in 3D from depth, picks it with a vacuum tool, and sorts it: pills go to a KLT bin and cartons go to a tote.
After every pick it observes the table again, and it stops when the table is empty.

**Headline result: 72/72 objects placed over 8 runs in 4 random layouts, with 0 failed picks** (v2 demo).
The earlier v1 demo placed 40/40 boxes over 10 runs.

No real images were used anywhere in the loop. Control uses perception output only. Ground truth is read only to
log the perception error and confirm placement.

---

## 2. End-to-end pipeline: from synthetic data to a working robot

```
 [1] Sim environment ─▶ [2] SDG (Replicator, 50k frames) ─▶ [3] Train RT-DETR ─▶ [4] Export ONNX/TensorRT
                                                                                     │
 [7] Results: 72/72 ◀── [6] Closed-loop pick & place ◀── [5] 3D localization from depth
```

### Step 1: Simulation environment (Gate M1/M2)
- Isaac Sim 6.0.1 on an RTX 5070 Ti, started through the kit binary (the `SimulationApp` path segfaulted on this setup). ROS2 Jazzy bridge verified.
- A pharma packaging cell (`poc_workspace.usd`): a Franka Panda, a table with **pills (8 × 3 mm)** and **cardboard boxes**, a shelf, and a **RealSense D435-style RGB-D camera** (1280×720, 66° HFOV).
- A D435 depth-accuracy model was built to choose the camera placement (`sim_environment/sensor_accuracy_analysis.md`).

### Step 2: Synthetic data generation, SDG (Gate M3)
Script: `sdg/phase3_sdg_generate.py` (Omniverse Replicator). COCO conversion: `sdg/convert_to_coco.py`.

| Item | Value |
|---|---|
| Frames | **50,000** RGB-D frames at 1280×720 |
| Writers | `BasicWriter`: RGB, depth (`distance_to_image_plane`, saved as uint16 mm PNG), 2D bounding boxes and semantic labels. `PoseWriter` (CenterPose format): 6D pose and projected 3D cuboid per object |
| Classes | `pill` (12 in the scene), `box` (6 in the scene) |
| Domain randomization per frame | Camera eye and target jitter. Pill positions in a 20 × 16 cm region and box positions in a 24 × 20 cm region. Cardboard colour ±0.15 per channel. Distant light 1500–5000 intensity, 4000–9000 K, elevation 20–80°, any azimuth. Dome light 200–900 |
| Speed / size | ~0.46 s per frame, about 6.4 h unattended, 61 GB |
| Robustness | Crash-resume: on restart the script binary-searches the last fully written frame and continues from there |
| Labels | Converted to COCO (`sdg/convert_to_coco.py`) for RT-DETR |

**Key fix:** the first full run used a straight-down camera, and the robot's own body filled **57.8% of the frame** on average.
The camera was moved to an **oblique off-axis mount**, and all 50k frames were regenerated.

### Step 3: Detector training (Gate M4a)
- **RT-DETRv2-r18vd** (PyTorch, upstream repo; NVIDIA TAO did not support it), 640×640 input. Config: `training/configs/rtdetrv2_r18vd_pillbox.yml`.
- Multi-scale and fixed-scale runs were compared to recover small-pill accuracy. The best model is from epoch 19.
- **Results:** box mAP@0.5 **0.987** (target >0.92, PASS). Pill mAP@0.5 **0.742** (target >0.85). This deviation was accepted: pills are only 5×5 px in the training frames.

### Step 4: Export
- ONNX (opset 18): `models/rtdetrv2_r18vd_pillbox_epoch19.onnx` (+ `.onnx.data`), verified with onnxruntime.
- TensorRT FP16 engine built and verified on the RTX 5070 Ti (`trtexec &&&& PASSED`). No Jetson Orin engine (no hardware).
- FoundationPose (boxes) was also verified end-to-end through the ROS2 bridge. The final demo didn't need it (see section 5).

### Step 5: 3D localization from depth
RT-DETR proposes, and the depth image confirms:
- Box: depth pixels → world points → top face → center, height, yaw and a size check.
- Pill: the detection must sit on the 3 mm depth bump on the mat.
- Anything that isn't object-shaped in 3D is rejected.

Details are in section 3.

### Step 6: Closed-loop pick & place
Observe → detect → localize → pick (differential IK + vacuum) → place → observe again, until the table is empty. Details are in section 3.

### Step 7: Result
**v1: 40/40 boxes. v2: 72/72 objects (pills + cartons), 0 failed picks.** Details are in section 4.

### Phase / gate summary

| Phase | Result |
|---|---|
| M1 Workstation | PASS |
| M2 Sim environment | PASS, with an open issue: `panda_joint1` PD instability in the Franka+Robotiq asset |
| M3 SDG 50k | PASS |
| M4a RT-DETR | PASS, with the pill mAP deviation |
| M4b FoundationPose (boxes) | PASS. CenterPose (pills): trained, but validation was 0, so it is not usable |
| Final demo v1 / v2 | **40/40 / 72/72** |

---

## 3. System architecture (v2)

```
 Isaac Sim 6.0.1 (one process)                          Side process
 ┌──────────────────────────────────────────────┐      ┌───────────────────────────┐
 │ Warehouse scene + packing table + KLT bin     │      │ RT-DETR r18 (ONNX Runtime │
 │ RGB-D camera 1280×720, 66° HFOV, oblique      │─RGB─▶│ CUDA), classes pill/box   │
 │                                              │      │ multi-scale for pills     │
 │ Localization (depth → 3D, validate shape)    │◀─det─┤                           │
 │ Task logic: pills first, then boxes          │      └───────────────────────────┘
 │ Differential IK (damped least squares)       │
 │ PhysX PD joint drives (Franka)               │
 │ Surface Gripper vacuum cup (attach = grasp)  │
 │ In-app "Perception" panel (right side)       │
 └──────────────────────────────────────────────┘
```

Each cycle:

1. **Observe:** the arm parks out of the camera view, and the camera renders RGB and depth.
2. **Detect:** RT-DETR runs one pass for boxes. Pills get passes at 1.0, 0.75 and 0.6 downscale, because pills render larger here than the ~5 px they had in training.
3. **Localize:**
   - Boxes: the top face comes from depth, and the largest cluster gives center, height, yaw and a size check.
   - Pills: the detection must sit on a 3 mm depth bump on the mat. No bump means the detection is rejected. A pill the detector later loses is kept as "tracked" while its bump is still there.
4. **Pick:** a top-down approach on smooth Cartesian waypoints. The vacuum attach status confirms the grasp, the sim equivalent of a vacuum-pressure switch.
5. **Place:** pills go into the KLT bin. Boxes are squared up to yaw 0 and placed into tote slots. Then the arm observes again.

### Key files (`demo/`)

| File | Role |
|---|---|
| `run_v2_demo.sh` | Launcher (`--record`, `--headless`, `--seed N`) |
| `demo_scene_v2.py` | Warehouse scene, packing table, bin, tote, vacuum tool (`add_vacuum_tool`) |
| `pick_place_v2.py` | Perception → localization → IK → grasp/place loop and the overlay panel |
| `rtdetr_detect.py` | RT-DETR ONNX inference, `detect_multiscale` |
| `localize.py` | Depth → world points, box top face, pill bump validation |
| `run_final_demo.sh`, `pick_place_final.py`, `demo_scene.py` | v1 demo |

### Run it

```bash
cd demo
./run_v2_demo.sh --record          # live window + real-time screen recording
./run_v2_demo.sh --headless        # renders composite demo.mp4 frame by frame
./run_v2_demo.sh --record --seed 5 # different layout
```
Outputs go to `logs/final_demo/v2_<mode>_seed<N>/` (path set in the script): `screen_recording.mp4` or `demo.mp4`, `demo_log.txt`, `summary.json`, `perception/`.

---

## 4. Results

### v2: warehouse, pills + cartons, vacuum tool (2026-09-27)

| Run | Pills in bin | Boxes in tote | Failed attempts | Wall time |
|---|---|---|---|---|
| seed 1 (headless) | 5/5 | 4/4 | 0 | 418 s |
| seed 2 (headless) | 5/5 | 4/4 | 0 | 430 s |
| seed 3 (headless) | 5/5 | 4/4 | 0 | 398 s |
| seed 11 (headless + 4 live GUI runs) | 5/5 each | 4/4 each | 0 | 368–424 s |
| **Total (8 runs)** | **40/40** | **32/32** | **0** | **72/72 objects** |

- Pill localization error: **0.5–1.0 mm**. Box error: **5.6–7.9 mm**, yaw within ~2°.
- Vacuum-cup contact error: 0.9–3.2 mm.
- Detector confidence: boxes 0.94–0.98. Isolated pills often drop below the 0.22 threshold after their neighbours are picked, so depth tracking was needed in every seed.
- Every false candidate was rejected by the 3D check, for example the KLT bin detected as a "box".
- The live window runs at **32–50 FPS** on the RTX 5070 Ti. One full run takes about 6–7 minutes.

### v1: boxes only, Franka hand (2026-09-26)

- **40/40 boxes over 10 runs** (seeds 1–5 headless, seed 7 ×3 video, seeds 3 and 5 live), 0 missed grasps.
- Localization 3.0–4.8 mm and <2° yaw. IK tracking at the grasp pose 1.3–1.6 mm. The fingers close to 56.6 mm on 55 mm boxes, a real friction grasp.

---

## 5. Engineering decisions and lessons

- **Vacuum instead of a parallel gripper:** an 8 mm pill and a 10 cm carton can't share a parallel gripper, since the Franka hand opens to 8 cm. Real warehouse cells use suction for this mix.
- **A new cell instead of the original ROS2/MoveIt2 stack:** that stack reached the pre-grasp pose, then failed every later motion with MoveIt `START_STATE_INVALID` (`panda_joint6` settled at 3.82 rad, above the 3.7525 rad limit). Underneath that, the Franka+Robotiq articulation never ran stably under PD control (the `panda_joint1` divergence). The final demo uses the stock Franka on real PD drives.
- **Depth as the validator:** RT-DETR proposes and depth confirms. This removed every false positive and kept tiny pills trackable when the detector lost them.
- **No pose network needed for top-down picks:** position + yaw from RGB-D was enough for cuboid boxes and flat pills.
- **Isaac Sim gotchas:** quitting the full kit needs `/app/file/ignoreUnsavedOnExit=true`. `fixtures/shelf.usd` has its transform ops in the wrong order (scale after translate). `get_dof_efforts()` reads 0.0.

---

## 6. Limitations (honest scope)

- **Simulation only.** No real camera or real robot has been tested yet. Sim depth is perfect, so real RealSense noise will hurt the pill-bump validation.
- The working demo is **not the original spec stack**: it has no ROS2, MoveIt2, BehaviorTree, FoundationPose or CenterPose in the loop. The spec loop fails after the approach move (it was audited on 2026-10-05).
- The CenterPose pill model trained for 15 epochs, but validation 3D IoU and 2D MPE were 0.0, so it is not usable.
- `panda_joint1` instability in the Franka+Robotiq asset is still open.
- No Jetson Orin engine has been built, and Phases 7 (edge) and 8 (benchmark) have not started.

---

## 7. Repository layout

```
sim_environment/   Isaac Sim cell: poc_workspace.usd, box/pill/shelf fixtures, build scripts, D435 accuracy analysis
sdg/               Replicator SDG (50k generator + 12-frame validator) and COCO converter
training/          RT-DETRv2 configs (used inside the upstream RT-DETRv2 repo) + ONNX export spec
models/            Trained RT-DETRv2-r18 pill/box detector, ONNX (Git LFS)
demo/              v1 and v2 pick & place demos (scene, detection, localization, control, launchers)
```

## 8. Setup notes

- **Isaac Sim 6.0.1 standalone** is required. The scripts are launched through the kit binary (`./kit/kit apps/isaacsim.exp.base.python.kit --exec <script>`), wrapped by `demo/run_v2_demo.sh`.
- This is a PoC snapshot: **absolute paths from the original workstation are hard-coded** at the top of the launcher scripts and modules (Isaac Sim install, repo root, the detector's Python venv, output folders). Edit them for your machine.
- The detector runs in a side process with `onnxruntime-gpu`, `numpy`, `opencv-python` and `Pillow`.
- The v2 scene loads `Simple_Warehouse/full_warehouse.usd` from NVIDIA's Isaac Sim asset server.
- Training: clone [RT-DETR](https://github.com/lyuwenyu/RT-DETR) (`rtdetrv2_pytorch`), put `training/configs/*.yml` in `configs/rtdetrv2/`, and point the dataloader at the COCO output of `sdg/convert_to_coco.py`.
- The 50k dataset (61 GB) is not included. Regenerate it with `sdg/phase3_sdg_generate.py`.
- `models/` uses **Git LFS**. Run `git lfs install` before cloning.

## 9. Demo videos

- Full 6-minute **unedited real-time** Isaac Sim screen recording (seed 11). The 3D view is on the left and the live Perception panel is on the right.
- 86-second cut: real-time open and close, with the middle sped up and labelled on screen.

**Stack:** Isaac Sim 6.0.1 · Omniverse Replicator · PhysX · RT-DETRv2 · ONNX Runtime (CUDA) · TensorRT · Python

**Next steps:** a real RealSense + Jetson Orin test, 6-DoF pose for irregular items, and fixing the Robotiq articulation so the ROS2/MoveIt2 spec stack can run.
