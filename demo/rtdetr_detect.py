#!/usr/bin/env python3
"""RT-DETR (trained on the Phase 3 SDG dataset) inference, standalone.

Runs in phase4_perception_training/rtdetr_venv (onnxruntime-gpu). Preprocessing
matches rtdetrv2_r18vd_pillbox.yml's val_dataloader exactly: plain Resize to
640x640, RGB, float32 in [0,255] (ConvertPILImage scale=False).

Usage: rtdetr_detect.py <image.png> <out.json> [--vis out.png] [--thresh 0.5]
Output JSON: {"detections": [{"label": "box"|"pill", "score": s, "xyxy": [x1,y1,x2,y2]}]}
"""
import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

MODEL = os.path.join(os.path.dirname(__file__), "..", "phase4_perception_training",
                     "exported_models", "rtdetrv2_r18vd_pillbox_epoch19.onnx")
CLASSES = {0: "pill", 1: "box"}


def load_session():
    return ort.InferenceSession(os.path.abspath(MODEL),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def detect(sess, bgr, thresh):
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    x = x.transpose(2, 0, 1)[None]
    sizes = np.array([[w, h]], dtype=np.int64)
    labels, boxes, scores = sess.run(None, {"images": x, "orig_target_sizes": sizes})
    out = []
    for lab, box, sc in zip(labels[0], boxes[0], scores[0]):
        if sc >= thresh:
            out.append({"label": CLASSES.get(int(lab), str(int(lab))), "score": float(sc),
                        "xyxy": [float(v) for v in box]})
    return out


def _nms(dets, iou_thr=0.3):
    def iou(a, b):
        iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = iw * ih
        return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    keep = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(iou(d["xyxy"], k["xyxy"]) < iou_thr for k in keep):
            keep.append(d)
    return keep


PILL_SCALES = (1.0, 0.75, 0.6)


def detect_multiscale(sess, bgr, thresh_box, thresh_pill):
    """Boxes from the full-resolution pass. Pills from a full pass plus a 0.75x pass:
    in the Phase 3 training frames a pill spans ~5 px, so a pill that renders larger
    (closer camera) is detected more confidently after downscaling to training scale.
    The downscaled frame is pasted onto a median-colored canvas of the original size
    so the model input geometry (1280x720 -> 640x640) matches training."""
    h, w = bgr.shape[:2]
    out = [d for d in detect(sess, bgr, thresh_box) if d["label"] == "box"]
    pills = []
    for s in PILL_SCALES:
        if s == 1.0:
            img = bgr
        else:
            small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            img = np.empty_like(bgr)
            img[:] = np.median(bgr.reshape(-1, 3), axis=0).astype(np.uint8)
            img[:small.shape[0], :small.shape[1]] = small
        for d in detect(sess, img, thresh_pill):
            if d["label"] != "pill":
                continue
            x1, y1, x2, y2 = (v / s for v in d["xyxy"])
            if x2 <= w and y2 <= h:
                pills.append({"label": "pill", "score": d["score"], "xyxy": [x1, y1, x2, y2], "scale": s})
    return out + _nms(pills)


def draw(bgr, dets):
    vis = bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        col = (0, 200, 255) if d["label"] == "box" else (255, 80, 200)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(vis, f'{d["label"]} {d["score"]:.2f}', (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return vis


def serve():
    """Persistent mode for the sim: one request per stdin line
    '<image> <out_json> <vis_png> <thresh>', one 'OK <n>' / 'ERR <msg>' line back."""
    import sys
    sess = load_session()
    print("READY " + ",".join(sess.get_providers()), flush=True)
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        try:
            image, out_json, vis, thresh = parts[0], parts[1], parts[2], float(parts[3])
            bgr = cv2.imread(image)
            if len(parts) > 4:   # '<...> <box_thresh> <pill_thresh>' -> multi-scale mode (v2 demo)
                dets = detect_multiscale(sess, bgr, thresh, float(parts[4]))
            else:
                dets = detect(sess, bgr, thresh)
            json.dump({"detections": dets}, open(out_json, "w"), indent=1)
            cv2.imwrite(vis, draw(bgr, dets))
            print(f"OK {len(dets)}", flush=True)
        except Exception as e:  # keep serving; the sim treats ERR as "no detections"
            print(f"ERR {e!r}", flush=True)


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        serve()
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out_json")
    ap.add_argument("--vis")
    ap.add_argument("--thresh", type=float, default=0.5)
    a = ap.parse_args()
    bgr = cv2.imread(a.image)
    dets = detect(load_session(), bgr, a.thresh)
    json.dump({"detections": dets}, open(a.out_json, "w"), indent=1)
    if a.vis:
        cv2.imwrite(a.vis, draw(bgr, dets))
    print(f"{len(dets)} detections: " + ", ".join(f'{d["label"]}:{d["score"]:.2f}' for d in dets))


if __name__ == "__main__":
    main()
