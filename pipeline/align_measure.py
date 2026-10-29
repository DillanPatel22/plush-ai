"""Align a raw 3D dog to the project frame and measure its side-view proportions.

Frame: X left/right (symmetry plane X=0), Y nose->tail (nose at -Y), Z up, ground Z=0, units mm.

Usage: python3 align_measure.py raw.glb out_dir [withers_mm]
Writes: aligned.ply, measures.json, side_silhouette.png (with landmarks drawn).
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def load_mesh(path):
    scene = trimesh.load(path, force="scene")
    mesh = trimesh.util.concatenate([g for g in scene.dump() if isinstance(g, trimesh.Trimesh)])
    return mesh


def align_matrix(mesh):
    """4x4 transform taking a Y-up glTF dog (as Tripo outputs it) into the project frame.

    Up = glTF +Y. Body axis = the longer horizontal axis. Nose end = the end with more
    surface area high up (the head), not the thin tail. Ground Z=0, centred on X, nose at Y=0.
    """
    c = mesh.vertices.mean(0)
    v = mesh.vertices - c
    up = np.array([0.0, 1.0, 0.0])
    ext = v.max(0) - v.min(0)
    fwd = np.array([1.0, 0.0, 0.0]) if ext[0] >= ext[2] else np.array([0.0, 0.0, 1.0])
    side = np.cross(fwd, up)              # right-handed: X = Y x Z
    R = np.stack([side, fwd, up])
    p = v @ R.T
    y = p[:, 1]; z = p[:, 2]
    ylo, yhi = y.min(), y.max(); span = yhi - ylo
    zhigh = z.min() + 0.6 * (z.max() - z.min())
    tri_c = p[mesh.faces].mean(1); area = mesh.area_faces
    a_lo = area[(tri_c[:, 1] < ylo + 0.25 * span) & (tri_c[:, 2] > zhigh)].sum()
    a_hi = area[(tri_c[:, 1] > yhi - 0.25 * span) & (tri_c[:, 2] > zhigh)].sum()
    if a_hi > a_lo:                       # head at +Y: turn 180 degrees about Z so the nose is at -Y
        R = np.stack([-side, -fwd, up]); p = v @ R.T
    t = -np.array([(p[:, 0].max() + p[:, 0].min()) / 2, p[:, 1].min(), p[:, 2].min()])
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t - R @ c
    return M


def align(mesh):
    out = mesh.copy(); out.apply_transform(align_matrix(mesh)); return out


def side_silhouette(mesh, px_per_mm):
    """Rasterise the side view (Y right, Z up) into a boolean mask."""
    v = mesh.vertices
    ymin, ymax = v[:, 1].min(), v[:, 1].max(); zmax = v[:, 2].max()
    W = int((ymax - ymin) * px_per_mm) + 20; H = int(zmax * px_per_mm) + 20
    img = Image.new("L", (W, H), 0); d = ImageDraw.Draw(img)
    yz = np.c_[(v[:, 1] - ymin) * px_per_mm + 10, H - 10 - v[:, 2] * px_per_mm]
    for f in mesh.faces:
        d.polygon([tuple(yz[i]) for i in f], fill=255)
    return np.array(img) > 0, ymin, H


def measure(mesh, px_per_mm=1.0):
    mask, ymin, H = side_silhouette(mesh, px_per_mm)
    return measure_mask(mask, px_per_mm, ymin, H)


def photo_mask(path, thresh=235):
    """Silhouette of a dog photographed on a plain white background (nose must face left)."""
    from scipy import ndimage
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    m = (a.min(2) < thresh)
    m = ndimage.binary_opening(m, iterations=2)
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    m = ndimage.binary_fill_holes(m)
    rows = np.where(m.any(1))[0]
    m = m[: rows.max() + 1]                      # crop below the paws so ground = last row
    pad = np.zeros((m.shape[0] + 10, m.shape[1]), bool); pad[: m.shape[0]] = m
    return pad


def measure_mask(mask, px_per_mm, ymin=0.0, H=None):
    rows, cols = mask.shape
    if H is None:
        H = rows
    to_mm_y = lambda c: (c - 10) / px_per_mm + ymin
    to_mm_z = lambda r: (H - 10 - r) / px_per_mm
    col_has = mask.any(0)
    top = np.where(col_has, mask.argmax(0), rows)                 # first filled row from top
    bottom = np.where(col_has, rows - 1 - mask[::-1].argmax(0), -1)
    ground_r = bottom.max()
    zmax_mm = to_mm_z(top[col_has].min())
    row_at = lambda z_mm: int(round(H - 10 - z_mm * px_per_mm))
    # Leg columns: filled in a mid-leg band (10-25 % of total height); the body never reaches that low.
    r_hi, r_lo = row_at(0.25 * zmax_mm), row_at(0.10 * zmax_mm)
    leg_cols = np.where(mask[r_hi:r_lo].any(0))[0]
    gaps = np.diff(leg_cols)
    split = np.argmax(gaps)                                       # front/rear legs split at the biggest gap
    front = leg_cols[: split + 1]; rear = leg_cols[split + 1:]
    gap = rear.min() - front.max()
    # Elbow/brisket: bottom of the body blob just behind the front leg (first 15 % of the gap).
    def body_bottom(c):
        col = mask[:, c]; r = np.where(col)[0]
        if not len(r): return np.nan
        # filled run containing the middle of the body height
        runs = np.split(r, np.where(np.diff(r) > 1)[0] + 1)
        mid = row_at(0.55 * zmax_mm)
        for run in runs:
            if run.min() <= mid <= run.max(): return run.max()
        return runs[0].max()
    cs = np.arange(front.max() + int(0.06 * gap), front.max() + max(int(0.06 * gap) + 3, int(0.22 * gap)))
    elbow_r = np.nanmax([body_bottom(c) for c in cs])
    elbow_c = int(cs[np.nanargmax([body_bottom(c) for c in cs])])
    # Withers: topline just behind the front leg (top of the shoulder blades sits over the elbow).
    withers_c = int(front.max() + 0.08 * gap)
    withers_r = top[withers_c]
    Hmm = to_mm_z(withers_r)
    E = to_mm_z(elbow_r)
    # Chest front (prosternum): most forward filled column between the elbow and 45 % up to the withers.
    zband = lambda z0, z1: (row_at(z1), row_at(z0))
    r0, r1 = zband(E, E + 0.45 * (Hmm - E))
    front_c = np.where(mask[r0:r1].any(0))[0].min()
    # Point of buttock: most rearward column in the 40-58 % height band (below the tail root).
    r0, r1 = zband(0.40 * Hmm, 0.58 * Hmm)
    rear_c = np.where(mask[r0:r1].any(0))[0].max()
    nose_c = np.where(col_has)[0].min()
    L = (rear_c - front_c) / px_per_mm
    res = {
        "withers_height_mm": round(Hmm, 1),
        "elbow_height_mm": round(E, 1),
        "body_length_mm": round(L, 1),
        "length_to_height": round(L / Hmm, 3),
        "elbow_over_height": round(E / Hmm, 3),
        "nose_to_chest_front_mm": round((front_c - nose_c) / px_per_mm, 1),
        "front_leg_y_mm": [round(to_mm_y(front.min()), 1), round(to_mm_y(front.max()), 1)],
        "rear_leg_y_mm": [round(to_mm_y(rear.min()), 1), round(to_mm_y(rear.max()), 1)],
        "landmarks_px": {"withers": [int(withers_c), int(withers_r)], "elbow": [elbow_c, int(elbow_r)],
                          "chest_front": int(front_c), "buttock": int(rear_c), "ground": int(ground_r)},
    }
    return res, mask


def draw(mask, res, path):
    im = Image.fromarray((~mask * 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im); lm = res["landmarks_px"]
    g = lm["ground"]
    d.line([(0, g), (im.width, g)], fill=(0, 160, 0), width=2)
    for name, col in [("withers", (220, 0, 0)), ("elbow", (0, 0, 220))]:
        c, r = lm[name]; d.ellipse([c - 6, r - 6, c + 6, r + 6], outline=col, width=3); d.text((c + 8, r - 14), name, fill=col)
        d.line([(c, r), (c, g)], fill=col, width=1)
    for name in ["chest_front", "buttock"]:
        c = lm[name]; d.line([(c, 0), (c, g)], fill=(200, 120, 0), width=2); d.text((c + 4, 4), name, fill=(200, 120, 0))
    im.save(path)


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    withers_target = float(sys.argv[3]) if len(sys.argv) > 3 else 560.0
    ref_photo = sys.argv[4] if len(sys.argv) > 4 else None
    out.mkdir(parents=True, exist_ok=True)
    raw = load_mesh(src)
    M = align_matrix(raw)
    m = raw.copy(); m.apply_transform(M)
    s = 1.0
    for ppm in (300.0 / m.vertices[:, 2].max(), 1.0 / s):          # coarse then fine pass
        res0, _ = measure(m, px_per_mm=ppm * s if ppm < 1 else 1.0) if False else measure(m, px_per_mm=ppm)
        k = withers_target / res0["withers_height_mm"]
        m.apply_scale(k); s *= k
    res, mask = measure(m, px_per_mm=1.0)
    S = np.eye(4) * s; S[3, 3] = 1.0
    T = S @ M                                                        # raw -> project frame (mm)
    np.savetxt(out / "raw_to_project.txt", T)
    m.export(out / "aligned.ply")
    scene = trimesh.load(src, force="scene"); scene.apply_transform(T); scene.export(out / "aligned_textured.glb")
    draw(mask, res, out / "side_silhouette.png")
    res["scale_applied"] = round(s, 6)
    res["vertices"] = int(len(m.vertices)); res["faces"] = int(len(m.faces))
    res["watertight"] = bool(m.is_watertight)
    res["bbox_mm"] = [round(x, 1) for x in (m.vertices.max(0) - m.vertices.min(0))]
    ok = True
    if ref_photo:
        ref, rmask = measure_mask(photo_mask(ref_photo), 1.0)
        draw(rmask, ref, out / "reference_silhouette.png")
        res["reference"] = {k: ref[k] for k in ["length_to_height", "elbow_over_height"]}
        res["reference"]["nose_to_chest_over_height"] = round(ref["nose_to_chest_front_mm"] / ref["withers_height_mm"], 3)
        res["nose_to_chest_over_height"] = round(res["nose_to_chest_front_mm"] / res["withers_height_mm"], 3)
        for k in ["length_to_height", "elbow_over_height", "nose_to_chest_over_height"]:
            d = res[k] / res["reference"][k] - 1; good = abs(d) <= 0.03; ok &= good
            print(f"{k}: mesh {res[k]} vs reference {res['reference'][k]} ({d*100:+.1f} %) {'PASS' if good else 'FAIL'} (limit 3 %)")
    json.dump(res, open(out / "measures.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("landmarks_px",)}, indent=1))
    print("RESULT:", "PASS" if ok else "FAIL")
