"""Convert the Phase 3 dataset's per-frame bounding_box_2d_tight (npy + labels
json + prim_paths json, from BasicWriter) into standard COCO-format detection
annotations for RT-DETR fine-tuning.

Only pill and box classes are included -- per master_prompt_stage1.md Phase 4A,
RT-DETR is fine-tuned on "pills + boxes" (shelf is environment structure, not
a 2D-detection target; it's handled separately by FoundationPose/CenterPose's
own pose pipeline, not RT-DETR's 2D bbox output).

Category IDs are 0-indexed (0=pill, 1=box), NOT standard COCO's 1-indexed
convention. Confirmed against the actual rtdetrv2_pytorch training code
(src/data/dataset/coco_dataset.py's ConvertCocoPolysToMask): with
remap_mscoco_category=False (the correct setting for a custom dataset --
that flag's remap table is hardcoded to the standard 80-class MSCOCO category
list, not a generic remapper), the raw category_id from this JSON is used
DIRECTLY as the training label with zero further remapping. Also confirmed
against rtdetrv2_decoder.py/rtdetrv2_criterion.py that index 0 is safe to use
for a real class -- this architecture reserves num_classes (i.e. the LAST
index, not index 0) for the implicit no-object/background class
(`nn.Embedding(num_classes+1, ..., padding_idx=num_classes)`), matching
standard DETR-family design, not older background-at-index-0 detectors. See
stage1_perception_arm/phase3_synthetic_data/logs/rtdetr_onnx_spec.md.

Pure post-processing of already-generated annotation files -- no Isaac Sim
needed, runs as plain python3.
"""
import glob
import json
import os
import re

import numpy as np

FR = "/home/vishal/poc/stage1_perception_arm/phase3_synthetic_data/full_run_50k/basic_writer"
OUT_DIR = "/home/vishal/poc/stage1_perception_arm/phase3_synthetic_data/coco_annotations"
IMG_W, IMG_H = 1280, 720
CATEGORIES = [{"id": 0, "name": "pill"}, {"id": 1, "name": "box"}]
CAT_ID = {"pill": 0, "box": 1}
VAL_EVERY_N = 20  # every 20th frame -> val split (5%), rest -> train (95%)

os.makedirs(OUT_DIR, exist_ok=True)

rgb_files = glob.glob(f"{FR}/rgb_*.png")
frame_re = re.compile(r"rgb_(\d+)\.png$")
indices = sorted(int(frame_re.search(f).group(1)) for f in rgb_files)
print(f"CONVERT_MARKER_0_FOUND n_frames={len(indices)} min={indices[0]} max={indices[-1]}", flush=True)

train_images, val_images = [], []
train_anns, val_anns = [], []
ann_id = 1
skipped_frames = 0

for i, idx in enumerate(indices):
    tag = f"{idx:04d}"
    bbox_path = f"{FR}/bounding_box_2d_tight_{tag}.npy"
    labels_path = f"{FR}/bounding_box_2d_tight_labels_{tag}.json"

    try:
        bbox = np.load(bbox_path)
        with open(labels_path) as f:
            labels = json.load(f)
    except FileNotFoundError:
        skipped_frames += 1
        continue

    is_val = (idx % VAL_EVERY_N == 0)
    image_id = idx
    image_entry = {"id": image_id, "file_name": f"rgb_{tag}.png", "width": IMG_W, "height": IMG_H}

    frame_anns = []
    for row in bbox:
        sid = str(row["semanticId"])
        cls = labels.get(sid, {}).get("class")
        if cls not in CAT_ID:
            continue
        x0, y0, x1, y1 = int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        frame_anns.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": CAT_ID[cls],
            "bbox": [x0, y0, w, h],
            "area": w * h,
            "iscrowd": 0,
        })
        ann_id += 1

    target = (val_images, val_anns) if is_val else (train_images, train_anns)
    target[0].append(image_entry)
    target[1].extend(frame_anns)

    if (i + 1) % 5000 == 0:
        print(f"CONVERT_MARKER_PROGRESS processed={i + 1}/{len(indices)}", flush=True)

train_coco = {"images": train_images, "annotations": train_anns, "categories": CATEGORIES}
val_coco = {"images": val_images, "annotations": val_anns, "categories": CATEGORIES}

with open(f"{OUT_DIR}/train.json", "w") as f:
    json.dump(train_coco, f)
with open(f"{OUT_DIR}/val.json", "w") as f:
    json.dump(val_coco, f)

print(f"CONVERT_MARKER_1_DONE train_images={len(train_images)} train_anns={len(train_anns)} "
      f"val_images={len(val_images)} val_anns={len(val_anns)} skipped_frames={skipped_frames}", flush=True)
