# RT-DETR ONNX I/O spec — as consumed by `isaac_ros_rtdetr`

Status: **load-bearing reference** for Phase 4A training. Extracted directly
from the shipped, installed package (`isaac_ros_rtdetr` 4.4.0, `/opt/ros/jazzy`)
— node headers and the default parameter values baked into
`isaac_ros_rtdetr.launch.py` / `isaac_ros_rtdetr_core.launch.py`. Not inferred
from documentation summaries; this is what the currently-installed pipeline
literally does, read from source on disk.

## Why this exists

TAO Toolkit doesn't support the RT-DETR architecture at all (confirmed by
NVIDIA staff, see `master_prompt_stage1.md`'s ARCHITECTURE DECISION section
and the forum evidence there) — the fine-tuning path is a standard PyTorch
RT-DETR implementation trained outside TAO. For a custom-trained model to
actually work with the decoder node shipped here, its ONNX export has to
match this exact I/O contract. A community forum thread confirms this is
achievable but only after working through preprocessing/output-format
mismatches with an NVIDIA engineer — this doc pins the contract down in
advance instead of discovering it by trial and error.

## The full pipeline (camera image → detections), as shipped

```
camera image (e.g. 1280x720, rgb8)
  -> resize_node        [ResizeNode]              resize to fit 640x640, keep_aspect_ratio=True, no padding here
  -> pad_node            [PadNode]                 pad to exactly 640x640, BOTTOM_RIGHT padding
  -> image_format_node   [ImageFormatConverterNode] confirm rgb8, 640x640
  -> image_to_tensor_node [ImageToTensorNode]       image -> tensor, tensor_name="image", scale=False
  -> interleave_to_planar_node [InterleavedToPlanarNode]  HWC [640,640,3] -> planar CHW
  -> reshape_node        [ReshapeNode]             CHW [3,640,640] -> NCHW [1,3,640,640], output_tensor_name="input_tensor"
  -> rtdetr_preprocessor_node [RtDetrPreprocessorNode]   adds "orig_target_sizes" tensor from the ORIGINAL
                                                          (pre-resize) camera resolution -- this is what lets
                                                          RT-DETR rescale predicted boxes back to original
                                                          image coordinates
  -> tensor_rt_node      [TensorRTNode]            runs the ONNX/TensorRT model
  -> rtdetr_decoder_node [RtDetrDecoderNode]        parses model output into vision_msgs/Detection2DArray
```

## The exact ONNX contract (default parameter values, as shipped)

**Model input size**: 640x640, 3 channels (`MODEL_INPUT_SIZE = 640`, hardcoded
constant in the launch file, not just a suggestion).

**Two named inputs** (`input_tensor_names` / `input_binding_names`, default):
| Name | Shape | Meaning |
|---|---|---|
| `images` | `[1, 3, 640, 640]` | preprocessed image, NCHW, RGB |
| `orig_target_sizes` | (small, per-image) | original (pre-resize) image height/width, used to rescale predicted boxes back to source-image coordinates |

**Three named outputs** (`output_tensor_names` / `output_binding_names`, default):
| Name | Meaning |
|---|---|
| `labels` | per-query predicted class index |
| `boxes` | per-query predicted bounding box |
| `scores` | per-query confidence score |

**Pixel normalization**: `image_to_tensor_node`'s `scale` parameter defaults
to **`False`** — pixel values are NOT rescaled to `[0,1]`, they stay in raw
`[0,255]` float range by default. (This is exactly the parameter the forum
thread's fix touched — they needed `scale=True` to match *their* custom
model's training-time normalization. Since we're training our own model from
scratch/fine-tuning rather than adapting to a fixed foreign checkpoint, the
simpler choice is to train with the SAME convention already in the default
pipeline config — i.e. don't normalize to `[0,1]` during training either —
so the shipped default launch files work unmodified. This is a design choice
to record, not a constraint to fight: it removes one whole class of the
mismatch the forum thread had to debug.)

**Confidence threshold**: decoder default is `0.9` (launch arg
`confidence_threshold`, tune after training based on validation performance).

## Label indexing: 0-indexed vs 1-indexed — CONFIRMED, decoder is agnostic

Checked the actual decoder source (`rtdetr_decoder_node.cpp`,
`NVIDIA-ISAAC-ROS/isaac_ros_object_detection`, `main` branch), not inferred
from its absence in the local headers/binary strings. The relevant lines,
quoted verbatim:

```cpp
auto labels = TensorToVector<int64_t>(msg, labels_tensor_name_, *cuda_stream_);
...
if (scores.at(i) <= confidence_threshold_) { continue; }
...
hyp.hypothesis.class_id = std::to_string(labels.at(i));
```

**The decoder applies zero offset, zero remapping, and has no hardcoded class
list or background-class handling.** It converts whatever raw integer is in
the `labels` tensor straight to a string with `std::to_string()`. It does not
"expect" 0-indexed or 1-indexed labels — it has no opinion at all; it
publishes exactly whatever the model outputs. (Locally, this is consistent
with what was already observed: no label-list yaml/json anywhere in the
installed package, and no `class_id`/label-string literals in the compiled
`.so`.)

**What this means in practice**: the indexing convention is entirely defined
by the trained model, not constrained by this ROS package.

**RESOLVED (verified against the actual training code, `rtdetrv2_pytorch`,
`lyuwenyu/RT-DETR`)**: `coco_annotations/{train,val}.json` now uses **0-indexed
`category_id`: 0=pill, 1=box** — NOT standard COCO's 1-indexed convention.
This was a deliberate change, not an oversight: checked
`src/data/dataset/coco_dataset.py`'s `ConvertCocoPolysToMask.__call__`, and
with `remap_mscoco_category=False` (the correct training-config setting for a
custom dataset — that flag's remap table, `mscoco_category2label`, is a
hardcoded literal dict of the standard 80-class MSCOCO category list, not a
generic custom-dataset remapper), the loader does exactly this:
```python
labels = [obj["category_id"] for obj in anno]  # raw passthrough, zero remap
```
So the COCO JSON's `category_id` values become the model's training labels
directly, with no remap step at all — which is why the JSON itself needed to
be 0-indexed rather than relying on the training pipeline to convert it.

Also verified this doesn't collide with an implicit background class during
**training**: checked `rtdetrv2_decoder.py` / `rtdetrv2_criterion.py` — the
denoising-query embedding and the classification loss use `num_classes + 1`
total slots, with the **last** index (`== num_classes`) reserved for the
no-object/background class (`nn.Embedding(num_classes+1, ...,
padding_idx=num_classes)`, loss one-hot-encodes to `num_classes+1` then drops
the last slot). This is real, but it's a training-time matching/loss detail —
see the cross-check below for what actually ships in the exported model.

## Cross-check against the DEPLOYED side, not just training: CONFIRMED, even stronger guarantee

The above (`num_classes+1`, background at the last index) describes the
*training loss*. What matters for `isaac_ros_rtdetr` is what the **exported,
deployed model** actually outputs — checked `RTDETRPostProcessor.forward()`
(`src/zoo/rtdetr/rtdetr_postprocessor.py`), the code that runs *inside* the
ONNX graph and produces exactly the `labels`/`boxes`/`scores` tensors the
decoder node consumes. Our config uses `use_focal_loss: True` (confirmed in
the smoke-test run's printed config), which takes this branch:

```python
if self.use_focal_loss:
    scores = F.sigmoid(logits)                          # logits shape: [..., num_classes] -- NOT num_classes+1
    scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
    labels = mod(index, self.num_classes)                # labels ALWAYS in [0, num_classes-1]
```

With focal loss, the classification head has exactly `num_classes` output
logits (2, for pill/box) — **there is no background logit in the exported
model's output at all**. `labels = index % num_classes` makes it
mathematically impossible for the exported model to ever emit anything
outside `{0, 1}` for this 2-class config; "no object" is handled implicitly
(all `num_classes` sigmoid scores are low, so the detection never makes the
top-k / never clears `confidence_threshold_` on the decoder side), not via a
competing background class index. This is a *stronger* guarantee than the
training-loss-based reasoning above, and it's the one that actually governs
what ships to the ROS node — the `num_classes+1`/last-index-reserved
mechanism never appears in the deployed output for a focal-loss model.

(For completeness, the non-focal-loss branch — not our config — does the
`num_classes+1` softmax and explicitly drops the last column:
`F.softmax(logits, dim=-1)[:, :, :-1]`. That branch's exported behavior does
match the training-loss description above. Ours doesn't take this branch.)

**Also confirmed**: `if self.deploy_mode: return labels, boxes, scores` runs
*before* the `remap_mscoco_category` block — the ONNX-exported graph
(`deploy_mode=True`, what `export_onnx.py` uses) never applies that remap at
all, regardless of the training-config setting. Whatever the raw 0-indexed
internal label space is, that's what ships to `isaac_ros_rtdetr`, full stop.

No bundled SyntheticaDETR class-list/labels asset was found locally to
cross-check against directly (the model itself was never downloaded in this
project, only its install script) — this finding rests on the actual
postprocessor/export source, which is authoritative regardless.

`num_classes: 2` in the training config, `remap_mscoco_category: False`.

**Downstream consequence, recorded so it doesn't become a Phase 5 mystery**:
Phase 5's grasp-logic node, when it reads `Detection2DArray`'s
`hypothesis.class_id` (a string, e.g. `"0"` or `"1"`), needs its own explicit
lookup table mapping `"0"->pill`, `"1"->box` — matching this same convention.
If Phase 5 is ever built assuming 1-indexed (`"1"->pill, "2"->box`) without
cross-checking this doc, it will silently pick the wrong grasp behavior for
every detection. Cross-reference this section from wherever Phase 5's
detection-to-grasp-behavior mapping gets implemented.

## Why this is a good sign for the "standard PyTorch, not TAO" path

The `images` + `orig_target_sizes` input pair and `labels` + `boxes` + `scores`
output triple are not an NVIDIA-invented convention — they're the standard
I/O contract used by the well-known open-source PyTorch RT-DETR
implementation and its own `export_onnx.py` script (the same family the
"custom RT-DETR model" forum thread referenced). SyntheticaDETR uses this
same convention. That means a standard PyTorch RT-DETR trained with the
matching num_classes (2: pill, box — see `coco_annotations/`) and exported
via that implementation's own standard ONNX export path should need
**minimal** custom glue to match this decoder's expectations, provided the
model is trained on `[0,255]`-range (unnormalized) input to match the
pipeline's default `scale=False`.

## Design decision vs. actual config: caught out of sync, now fixed

The `scale=False` decision above (pixel normalization) was recorded here but
not actually applied when the training config (`rtdetrv2_r18vd_pillbox.yml`)
was written — that config silently inherited the upstream `rtdetrv2_pytorch`
repo's stock `ConvertPILImage(scale=True)` default instead. Caught 2026-08-12
during an audit of documented-decision-vs-live-config drift (triggered by
finding this specific mismatch first). Every AP number from trial runs prior
to this date was produced under the wrong (`scale=True`) convention and is
void. Full writeup, root-cause analysis, and the fix: `docs/KNOWN_RISKS.md`
risk #4.

## Open items before an actual training run

- Confirm exact `num_classes` wiring — decoder itself is label-agnostic (no
  baked-in class list; `labels` tensor carries raw class indices), so class
  mapping (0=pill, 1=box, matching zero-indexed convention typical of this
  RT-DETR family, vs. COCO's 1-indexed convention used in
  `coco_annotations/*.json`) needs to be handled explicitly in the training
  config and re-confirmed against the decoder's actual label→string mapping
  (likely a ROS param or hardcoded string list not yet located).
- Verify the exact upstream PyTorch RT-DETR implementation/export script
  version before training, rather than assuming the most popular fork's
  current `export_onnx.py` still produces this exact tensor naming (open
  source projects drift).
