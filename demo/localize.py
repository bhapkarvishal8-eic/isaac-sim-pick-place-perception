"""RGB-D localization of RT-DETR box detections (numpy only; runs in Isaac Sim's python).

For each 2D box detection: back-project the aligned depth pixels inside the bbox
to world coordinates, keep the points above the table, take the top face
(highest points), and estimate
  * center (x, y) = centroid of the top face,
  * height      = top-face z (boxes rest on the table),
  * yaw         = rotation (mod 90 deg) minimizing the top face's bounding
                  rectangle area -- the angle a parallel gripper must match,
  * side length = top-face extent at that yaw (used to reject false positives).
"""

from __future__ import annotations

import numpy as np


def backproject(u, v, d, K):
    """Pixels (u, v) with depth d along the optical axis -> USD camera-local points (N, 3).
    USD cameras look down -Z with +Y up; ROS optical is +Z forward, +Y down."""
    xo = (u - K["cx"]) * d / K["fx"]
    yo = (v - K["cy"]) * d / K["fy"]
    return np.stack([xo, -yo, -d], axis=-1)


def to_world(p_cam, cam_to_world):
    """Row-vector convention (USD/Gf): p_world = [p, 1] @ M."""
    ph = np.concatenate([p_cam, np.ones((len(p_cam), 1))], axis=1)
    return (ph @ cam_to_world)[:, :3]


def _min_area_yaw(xy):
    """Yaw in [-45, 45) deg of the minimum-area bounding rectangle of 2D points."""
    c = xy - xy.mean(axis=0)
    best = (np.inf, 0.0, 0.0)
    for deg in np.arange(-45.0, 45.0, 0.5):
        a = np.radians(deg)
        rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        r = c @ rot.T
        ext = r.max(axis=0) - r.min(axis=0)
        area = ext[0] * ext[1]
        if area < best[0]:
            best = (area, deg, float(ext.mean()))
    return best[1], best[2]


def _largest_cluster(pts, cell=0.005):
    """Largest 8-connected cluster of points on a 5 mm xy grid (pure python/numpy)."""
    keys = np.floor(pts[:, :2] / cell).astype(int)
    cells = {}
    for i, k in enumerate(map(tuple, keys)):
        cells.setdefault(k, []).append(i)
    seen, best = set(), []
    for start in cells:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            c = stack.pop()
            comp.extend(cells[c])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in cells and n not in seen:
                        seen.add(n)
                        stack.append(n)
        if len(comp) > len(best):
            best = comp
    return pts[best]


def localize_box(det, depth, K, cam_to_world, table_z=0.0, shrink=0.08, side_range=(0.03, 0.09)):
    """Returns dict(center=(x,y,z), top_z, yaw_deg, side) or None if not box-like."""
    x1, y1, x2, y2 = det["xyxy"]
    w, h = x2 - x1, y2 - y1
    if w <= 4 or h <= 4 or w > 350 or h > 350:
        return None
    u0, u1 = int(x1 + shrink * w), int(x2 - shrink * w)
    # no shrink at the bbox top: the far edge of the top face lies there
    v0, v1 = int(y1), int(y2 - shrink * h)
    uu, vv = np.meshgrid(np.arange(u0, u1), np.arange(v0, v1))
    d = depth[v0:v1, u0:u1]
    ok = np.isfinite(d) & (d > 0.05) & (d < 5.0)
    if ok.sum() < 30:
        return None
    pts = to_world(backproject(uu[ok].astype(float), vv[ok].astype(float), d[ok], K), cam_to_world)
    above = pts[pts[:, 2] > table_z + 0.01]
    if len(above) < 30:
        return None
    top_z = float(np.percentile(above[:, 2], 98))
    top = above[above[:, 2] > top_z - 0.006]
    if len(top) < 20:
        return None
    top = _largest_cluster(top)       # a neighbouring box's top face can share the bbox
    if len(top) < 20:
        return None
    yaw, side = _min_area_yaw(top[:, :2])
    if not (side_range[0] < side < side_range[1]):
        return None
    cx, cy = top[:, :2].mean(axis=0)
    return {"center": (float(cx), float(cy), (top_z + table_z) / 2.0), "top_z": top_z,
            "yaw_deg": float(yaw), "side": side, "score": det["score"], "xyxy": det["xyxy"]}


def localize_pill(det, depth, K, cam_to_world, surface_z, pad=3, region=None):
    """A pill is a ~3 mm bump on a flat surface at height surface_z. Back-project the
    depth pixels around the detection, keep points 1.2-12 mm above the surface, and
    return their centroid (the pill top). None if there is no bump (false positive)
    or it lies outside `region` ((xmin, xmax), (ymin, ymax))."""
    x1, y1, x2, y2 = det["xyxy"]
    u0, u1 = max(0, int(x1) - pad), min(depth.shape[1], int(x2) + pad + 1)
    v0, v1 = max(0, int(y1) - pad), min(depth.shape[0], int(y2) + pad + 1)
    uu, vv = np.meshgrid(np.arange(u0, u1), np.arange(v0, v1))
    d = depth[v0:v1, u0:u1]
    ok = np.isfinite(d) & (d > 0.05) & (d < 5.0)
    if ok.sum() < 4:
        return None
    pts = to_world(backproject(uu[ok].astype(float), vv[ok].astype(float), d[ok], K), cam_to_world)
    bump = pts[(pts[:, 2] > surface_z + 0.0012) & (pts[:, 2] < surface_z + 0.012)]
    if len(bump) < 2:
        return None
    cx, cy = bump[:, :2].mean(axis=0)
    if region is not None:
        (xa, xb), (ya, yb) = region
        if not (xa < cx < xb and ya < cy < yb):
            return None
    return {"center": (float(cx), float(cy), float(bump[:, 2].max())), "top_z": float(bump[:, 2].max()),
            "score": det["score"], "xyxy": det["xyxy"], "n_pts": int(len(bump))}


def project(p_world, K, cam_to_world):
    """World point -> pixel (u, v). Inverse of backproject + to_world."""
    pc = np.append(np.asarray(p_world, float), 1.0) @ np.linalg.inv(cam_to_world)
    d = -pc[2]
    return K["cx"] + pc[0] * K["fx"] / d, K["cy"] - pc[1] * K["fy"] / d
