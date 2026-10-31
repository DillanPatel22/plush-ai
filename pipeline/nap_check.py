"""Fur-direction (nap) check: how far the fabric's pile direction, once sewn, is from a Labrador's coat direction.

Each piece is cut with its nap fixed (0 or 180 degrees on the roll), so on the finished plush the pile runs in a
single fabric direction per piece. Mapped back onto the 3D surface through the flattening, that direction is
compared with the dog's coat direction field:
  head and muzzle: nose -> back of skull;   neck and body sides: back and down (about 35 degrees below horizontal);
  chest front and legs: straight down;      underbody: back;      tail: towards the tip.
Reports, per piece and per body region, the share of area whose pile is more than 30 / 60 degrees off.

Usage: python3 nap_check.py flat_dir measures.json plush_nose_to_tail_mm out.json
"""
import json, math, sys
from pathlib import Path
import numpy as np

flat, ms, size = Path(sys.argv[1]), json.load(open(sys.argv[2])), float(sys.argv[3])
k = size / (ms["nose_to_chest_front_mm"] + ms["body_length_mm"])
leg_top_real = max(ms["elbow_height_mm"] * 1.05, 0.5 * ms["withers_height_mm"])


def export_fur(P):                     # the field export_pattern.py uses to choose each piece's rotation
    d = np.tile([0.0, 1.0, 0.0], (len(P), 1)); d[P[:, 2] < leg_top_real * k * 0.9] = [0, 0, -1.0]; return d


def coat(P, N, y0):
    """Labrador coat direction (unit vectors) and a region name per face."""
    y = (P[:, 1] - y0) / k; z = P[:, 2] / k
    d = np.tile([0.0, 0.82, -0.57], (len(P), 1)); reg = np.full(len(P), "body side", object)
    head = y < 230; d[head] = [0, 1, 0]; reg[head] = "head"
    chest = (y >= 200) & (y < 330) & (N[:, 1] < -0.3) & (z < 480); d[chest] = [0, 0, -1]; reg[chest] = "chest"
    under = (N[:, 2] < -0.5) & ~head; d[under] = [0, 1, 0]; reg[under] = "underbody"
    legs = z < leg_top_real * 0.9; d[legs] = [0, 0, -1]; reg[legs] = "legs"
    tail = (y > ms["nose_to_chest_front_mm"] + ms["body_length_mm"] + 30) & (z > 0.45 * ms["withers_height_mm"])
    d[tail] = [0, 0.9, 0.44]; reg[tail] = "tail"
    sole = (N[:, 2] < -0.85) & (z < 4); reg[sole] = "soles (pads, no pile)"
    return d, reg


def jac(V3, F, uv):
    e1 = V3[F[:, 1]] - V3[F[:, 0]]; e2 = V3[F[:, 2]] - V3[F[:, 0]]
    u1 = uv[F[:, 1]] - uv[F[:, 0]]; u2 = uv[F[:, 2]] - uv[F[:, 0]]
    return np.stack([u1, u2], 2) @ np.linalg.pinv(np.stack([e1, e2], 2))


def sa(P): return 0.5 * (np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1)))


y0 = None
files = sorted(flat.glob("P*.npz"))
for f in files:
    d = np.load(f); y0 = d["V3"][:, 1].min() if y0 is None else min(y0, d["V3"][:, 1].min())
rep = {"pieces": {}, "regions": {}}
reg_acc = {}
for f in files:
    d = np.load(f); V3, F, uv = d["V3"], d["F"], d["uv"].copy()
    if np.sign([sa(uv[t]) for t in F[:500]]).mean() < 0: uv[:, 0] *= -1
    J = jac(V3, F, uv); C = V3[F].mean(1)
    n = np.cross(V3[F[:, 1]] - V3[F[:, 0]], V3[F[:, 2]] - V3[F[:, 0]]); A = np.linalg.norm(n, axis=1); n /= A[:, None] + 1e-12
    def to2d(f3):
        f3 = f3 - (f3 * n).sum(1, keepdims=True) * n; return np.einsum("nij,nj->ni", J, f3)
    f2 = (to2d(export_fur(C)) * A[:, None]).sum(0)
    rot = -math.pi / 2 - math.atan2(f2[1], f2[0])                    # export: area-weighted fur points down
    nap = np.array([math.cos(-math.pi / 2 - rot), math.sin(-math.pi / 2 - rot)])   # pile direction in the piece's uv
    want, reg = coat(C, n, y0)
    w2 = to2d(want); w2 /= np.linalg.norm(w2, axis=1, keepdims=True) + 1e-12
    err = np.degrees(np.arccos(np.clip(w2 @ nap, -1, 1)))
    rep["pieces"][f.stem] = {"area_off_30deg_pct": round(100 * A[err > 30].sum() / A.sum(), 1),
                             "area_off_60deg_pct": round(100 * A[err > 60].sum() / A.sum(), 1),
                             "median_error_deg": round(float(np.median(err)), 0)}
    for r in np.unique(reg):
        s_ = reg == r; a = reg_acc.setdefault(r, [0.0, 0.0, 0.0])
        a[0] += A[s_].sum(); a[1] += A[s_ & (err > 30)].sum(); a[2] += A[s_ & (err > 60)].sum()
for r, (a, a30, a60) in reg_acc.items():
    rep["regions"][r] = {"area_off_30deg_pct": round(100 * a30 / a, 1), "area_off_60deg_pct": round(100 * a60 / a, 1)}
json.dump(rep, open(sys.argv[4], "w"), indent=1)
for p, v in rep["pieces"].items(): print(p, v)
for r, v in rep["regions"].items(): print(f"  {r:10s}", v)
bad = [r for r, v in rep["regions"].items() if v["area_off_30deg_pct"] > 20 and not r.startswith("soles")]
print("RESULT:", "PASS" if not bad else f"CHECK pile more than 30 deg off on over 20 % of {bad}")
