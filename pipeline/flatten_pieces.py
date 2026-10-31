"""Flatten every pattern piece at plush size and record distortion and boundary data for export.

- Pieces are scaled to the plush size first (k = plush nose-to-tail / template nose-to-tail).
- Left pieces are mirrors of right pieces (flattened once).
- The underbody gusset gets a centre slit from its rear edge forward to the crotch: the rear centre seam
  continues between the hind legs, which removes the saddle stretch there.

Usage: python3 flatten_pieces.py pieces_dir measures.json plush_nose_to_tail_mm out_dir
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).parent))
from flatten import flatten, flatten_injective, distortion, prescribe_boundary_lengths
from shapely.geometry import Polygon

pdir, meas, size_mm, out = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
ms = json.load(open(meas))
k = size_mm / (ms["nose_to_chest_front_mm"] + ms["body_length_mm"])
SLIT_Y_FROM_NOSE = 760.0                         # crotch point (real dog mm from the nose tip)
pieces = {"P01": None, "P02": "P03", "P04": None, "P05": "P06", "P07": "P14", "P15": "P16",
          "P08": "P09", "P10": "P11", "P12": "P13"}             # right/centre piece -> its mirrored left piece
SPLIT_GUSSET_Y_FROM_NOSE = 560.0                  # underbody gusset: front half / rear half (seam across the belly)


def centre_slit(V, F, y_tip):
    on = (np.abs(V[:, 0]) < 1e-6) & (V[:, 1] > y_tip)
    idx = np.where(on)[0]
    if len(idx) < 2: return V, F, 0
    tip = idx[np.argmin(V[idx, 1])]
    dup = [i for i in idx if i != tip]
    new_index = {i: len(V) + n for n, i in enumerate(dup)}
    V2 = np.vstack([V, V[dup]])
    cen = V[F].mean(1)
    F2 = F.copy()
    left = cen[:, 0] < 0
    for fi in np.where(left)[0]:
        for c in range(3):
            if F2[fi, c] in new_index: F2[fi, c] = new_index[F2[fi, c]]
    return V2, F2, len(dup)


report = {"scale": round(k, 4), "plush_nose_to_tail_mm": size_mm, "pieces": {}}
for pid, mirror in pieces.items():
    slit = 0
    if pid in ("P07", "P15"):
        # underbody gusset -> four pieces: front/rear halves (seam across the belly), each split on the centre
        # line (centre seam down the chest and between the legs). Simple shapes, low distortion.
        p = trimesh.load(pdir / "P07.obj", process=False)
        V = np.asarray(p.vertices, float); F = np.asarray(p.faces, np.int64)
        ysplit = ms.get("nose_y0", 0.0) + SPLIT_GUSSET_Y_FROM_NOSE
        cen = V[F].mean(1)
        keep = (cen[:, 1] < ysplit if pid == "P07" else cen[:, 1] >= ysplit) & (cen[:, 0] >= 0)
        sub = trimesh.Trimesh(V, F[keep], process=False); sub.remove_unreferenced_vertices()
        sub = max(sub.split(only_watertight=False), key=lambda c: c.area)
        V = np.asarray(sub.vertices, float); F = np.asarray(sub.faces, np.int64)
    else:
        p = trimesh.load(pdir / f"{pid}.obj", process=False)
        V = np.asarray(p.vertices, float); F = np.asarray(p.faces, np.int64)
    Vk = V * k
    uv, bnd = flatten(Vk, F, iters=150)
    method = "lscm+arap"
    def outline_ok(uv_, bnd_):
        # the export smooths the outline (~1.5 mm), so judge validity after the same smoothing
        from scipy.ndimage import gaussian_filter1d
        P2 = uv_[bnd_]; seg = np.linalg.norm(np.diff(P2, axis=0, append=P2[:1]), axis=1).mean()
        sg = max(1.0, 1.5 / seg)
        Ps = np.c_[gaussian_filter1d(P2[:, 0], sg, mode="wrap"), gaussian_filter1d(P2[:, 1], sg, mode="wrap")]
        Pg = Polygon(Ps)
        if Pg.is_valid: return True
        # tiny overlaps at sharp inner corners (armpits) are cleaned by the export; accept if negligible
        Pc = Pg.buffer(0)
        sa = 0.5 * abs(np.dot(Ps[:, 0], np.roll(Ps[:, 1], -1)) - np.dot(Ps[:, 1], np.roll(Ps[:, 0], -1)))
        return Pc.geom_type == "Polygon" and abs(Pc.area - sa) / sa < 0.01
    if not outline_ok(uv, bnd):                        # folded over itself: use the injective method
        uv, bnd = flatten_injective(Vk, F, iters=300); method = "tutte+slim"
    L3 = np.linalg.norm(Vk[np.roll(bnd, -1)] - Vk[bnd], axis=1).sum(); L2 = np.linalg.norm(np.roll(uv[bnd], -1, 0) - uv[bnd], axis=1).sum()
    # Forcing true 3D lengths on the boundary of a domed piece squeezes its interior (a dome's flat
    # perimeter is longer than its 3D rim). Without darts the honest result is ARAP's: good interior, rims
    # slightly long, eased in between matching notches. Darts (cone points) are the proper fix: v2.
    closure = None
    ar, lr, a3 = distortion(Vk, F, uv)
    w = a3 / a3.sum()
    m = trimesh.Trimesh(Vk, F, process=False)
    loops = len(trimesh.graph.connected_components(m.edges[trimesh.grouping.group_rows(m.edges_sorted, require_count=1)]))
    rec = {"faces": int(len(F)), "boundary_loops": int(loops), "slit_vertices": int(slit),
           "area_3d_cm2": round(a3.sum() / 100, 1), "area_flat_cm2": round(float((ar * a3).sum()) / 100, 1),
           "area_dev_p95": round(float(np.percentile(np.abs(ar - 1), 95)), 3),
           "len_dev_p95": round(float(np.percentile(np.abs(lr - 1), 95)), 3), "flipped": int((ar <= 0).sum()),
           "mirror": mirror, "method": method, "closure_change_mm": None if closure is None else round(closure, 3), "outline_valid": bool(outline_ok(uv, bnd)), "boundary_len_ratio": round(float(L2 / L3), 3)}
    report["pieces"][pid] = rec
    np.savez(out / f"{pid}.npz", V3=Vk, F=F, uv=uv, bnd=bnd, area_ratio=ar)
    print(pid, rec)
json.dump(report, open(out / "flatten_report.json", "w"), indent=1)
bad = [p for p, r in report["pieces"].items() if r["flipped"] or r["boundary_loops"] != 1 or not r["outline_valid"] or abs(r["boundary_len_ratio"] - 1) > 0.08]
print("RESULT:", "PASS" if not bad else f"FAIL {bad}")
