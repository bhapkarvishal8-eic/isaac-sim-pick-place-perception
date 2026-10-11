# Sensor accuracy analysis — RealSense-class depth accuracy vs. camera geometry

Status: **load-bearing reference**, referenced by two rounds of camera-geometry
fixes (`build_scene.py`, `phase3_sdg_generate.py`, `phase3_sdg_validate.py`).
Do not let this analysis live only in conversation history — update this file
if the camera geometry changes again.

## Why this exists

Phase 8's own benchmark spec (`docs/master_prompt_stage1.md`) requires
**translation error < 5mm** on an **8mm-diameter pill**. Camera-geometry fixes
made purely to solve *framing* (robot occupying too much of the frame) can
silently break this harder, separate requirement if they increase working
distance — RealSense-class depth sensors lose accuracy with range. This file
tracks that tradeoff explicitly across every candidate geometry tried, so it
doesn't get re-litigated from scratch on the next camera change.

## The accuracy model

RealSense D400-series (stereo-IR) depth sensors report error that scales
**quadratically with range** — a direct consequence of stereo triangulation
geometry (`error_z ≈ Z² × disparity_error / (baseline × focal_length_px)`,
where baseline and focal length are fixed for a given camera, leaving Z² as
the dominant term). Intel's own D435 datasheet publishes a single anchor
point: **"<2% accuracy at 2m"**, i.e. ≤40mm RMS error at 2000mm range. We use
that as the calibration point and scale quadratically:

```
error_mm(Z) = 40mm × (Z_meters / 2.0)²
```

**Caveats, stated plainly:**
- This is a **conservative, single-anchor-point estimate**, not a precise
  prediction. Real D400-series accuracy curves are not perfectly quadratic
  across the whole operating range — independent characterizations often show
  *better* than this formula predicts at short range (well under 1m) and are
  roughly consistent with it in the 1-3m regime where the "<2% at 2m" spec was
  calibrated.
- This estimates **raw single-pixel sensor noise**, not final pipeline
  accuracy. FoundationPose/CenterPose fit a known object model against many
  observed points, which can average down random noise below the raw
  per-pixel floor — especially for larger objects (boxes) with many surface
  points. This averaging benefit is much weaker for pills (tiny, few depth
  pixels, minimal averaging headroom), so pills remain the most exposed class
  regardless of which camera geometry is used — consistent with the project's
  own differentiated accuracy targets (>90% pills vs >95% boxes).
- Numbers here are a **risk signal for design decisions**, not a substitute
  for the actual Phase 8 pose-accuracy benchmark once the perception pipeline
  exists.

## Incidence-angle risk (separate from range)

Beyond range, viewing a surface at a steep angle off its normal degrades
stereo-correlation quality independent of distance. There's no formally
published per-degree penalty for D400-series sensors, but Intel's best-
practices guidance commonly cites staying under roughly **60-70° off the
camera boresight** for reliable correlation on a surface. This is checked
per-candidate below as a qualitative pass/fail against that commonly-cited
threshold, not a quantified mm penalty.

## Candidates evaluated

### 0. Original: 0.8m straight-down, 69° FOV (baseline, pre-fix)
- Central working distance: 0.8m → estimated error **6.40mm** — already
  marginally over the 5mm budget under this conservative model. **This is a
  standing risk that predates both camera fixes below** — not something
  either fix introduced.
- Framing: mean robot occupancy 57.8%, 58.7% of frames >50% robot, pill
  visible in only 49% of frames (see Phase 3 dataset QA, robot-occupancy
  investigation).
- Incidence angle: 0° (camera looks straight down, normal incidence) — this
  was the only candidate with zero incidence-angle risk.

### 1. Round 1 (REJECTED): straight-up height increase to 2.0m, 30.8° FOV, ±6.7° tilt jitter
- Fixes framing cleanly: mean robot occupancy 22.9%, 0% of frames >50%
  robot, pill visible in 96.7% of frames.
- **Breaks accuracy budget badly.** Empirical pill-region range 0.96–1.20m
  (mean-of-means 1.08m) → estimated error **9.2–14.5mm**, mean **11.7mm** —
  2-3x over budget. Farthest points (shelf) reach 16.4mm — 3x over.
- Root cause: height-only fixes trade the exact thing the accuracy spec
  constrains (range) for framing. Rejected for this reason.

### 2. Round 2, first oblique attempt: eye=(0,−0.70,0.90), target=(0,0.06,0.14), FOV=58°, tilt jitter ±5cm/±3cm
- Framing: mean robot occupancy 17.4%, 0% >50% robot, pill visible 99.7%,
  all three regions co-visible in 99.7% of frames — better than round 1 on
  every framing metric.
- Accuracy: empirical pill-region range 0.96–1.20m (mean-of-means 1.08m) →
  estimated error **9.2–14.5mm** — still 2-3x over budget, only marginally
  better than round 1's numbers despite the much shorter *nominal* eye-to-
  target distance (0.9m vs 2.0m). The oblique angle put the actual workspace
  regions farther from the camera than the nominal distance suggested.
  Central incidence angle 45°.
- Superseded by candidate 3 below (closer variant, same mount concept).

### 3. Round 2, final: eye=(0,−0.55,0.65), target=(0,0.04,0.11), FOV=66°, eye jitter ±4cm, target jitter ±3cm — **ADOPTED**
Analytic FOV coverage (all 6 required workspace corners, `build_scene.py`
`project_point` method) confirmed in-bounds before any rendering. Full
validation: n=300, RGB+semantic-segmentation+depth captured together per
frame (not framing validated separately from accuracy).

**Framing / occlusion:**
| Metric | Value |
|---|---|
| Mean robot occupancy | 16.8% |
| Median robot occupancy | 16.6% |
| Max robot occupancy ever seen | 32.9% |
| Frames >50% robot | 0/300 (0.0%) |
| Frames >30% robot | 1/300 (0.3%) |
| Pill visible | 299/300 (99.7%) |
| Pill mean area when visible | 0.0134% (vs 0.0094% original, 0.0058% round-1 height-fix) |
| Box visible | 300/300 (100%) |
| Shelf visible | 300/300 (100%) |
| All three regions simultaneously visible | 299/300 (99.7%) |
| Frames with zero target regions visible | 0/300 (0.0%) |

**Empirical range** (actual measured depth under pill/box/shelf semantic
masks, not analytic corner estimates):
| Region | Nearest | Farthest | Mean-of-means |
|---|---|---|---|
| Pill | 0.680m | 0.909m | 0.793m |
| Box | 0.243m | 0.936m | 0.549m |
| Shelf | 0.725m | 0.986m | 0.859m |
| Overall | 0.243m | 0.986m | 0.734m |

**Estimated depth accuracy** (same quadratic model, applied to the ranges above):
| Point | Range | Est. error | vs. 5mm budget |
|---|---|---|---|
| Box nearest (closest point in the whole scene) | 0.243m | 0.59mm | OK |
| Box mean-of-means | 0.549m | 3.01mm | OK |
| Pill nearest | 0.680m | 4.63mm | OK |
| **Pill mean-of-means** | **0.793m** | **6.29mm** | Marginal (comparable to the 6.40mm original-baseline estimate) |
| Shelf nearest | 0.725m | 5.25mm | Marginal |
| Pill farthest | 0.909m | 8.26mm | Over budget |
| Shelf mean-of-means | 0.859m | 7.38mm | Over budget |
| Overall farthest ever seen (shelf) | 0.986m | 9.73mm | Over budget |

**Verdict on accuracy:** pill mean-of-means (6.29mm) is statistically
equivalent to the original 0.8m straight-down baseline (6.40mm) — this
candidate does not make the standing accuracy risk meaningfully worse, and
is dramatically better than round 1's 2.0m setup (9.2–14.5mm, 2-3x this
candidate's numbers). The oblique angle does introduce some per-object range
spread across the tray (0.68–0.91m for pills) that a perfectly straight-down
view wouldn't have, since objects at different points in the tray are no
longer equidistant from the camera — visible in the near/far spread above.

**Incidence angle** (central ray 47.5°; per-region-corner, analytic):
| Point | Incidence |
|---|---|
| Boxes near corner | 32.4° |
| Pills near corner | 33.9° |
| Pills far corner | 45.5° |
| Shelf front-opening bottom-left | 42.5° |
| Boxes far top corner | 56.9° |
| Shelf front-opening top-right | 61.0° |

All points are within or at the edge of the commonly-cited <60-70° safe
zone; the shelf's single farthest corner (61.0°) is marginally over the
tighter end of that range. Not flagged as a blocker, but worth watching if
shelf-compartment pose accuracy turns out to be a problem in Phase 4/8.

## Open items for Phase 4/8

- The <5mm translation-error budget is marginal-to-over-budget for pills
  under this conservative raw-sensor model across **every** geometry
  considered, including the original baseline. This should be verified
  empirically once the real perception pipeline exists (Phase 8's own
  pose-accuracy benchmark is the actual test) rather than assumed solved or
  assumed broken from this analysis alone.
- If Phase 8 benchmarking shows pill pose accuracy is in fact failing the
  5mm budget, candidates to investigate next: a shorter-range purpose-built
  sensor for the pill region specifically (e.g. D405-class, accurate to
  sub-mm under ~50cm but not usable at the box/shelf working distances used
  here), or a two-stage pipeline (wide-FOV coarse localization + a second,
  closer-range refinement capture).
