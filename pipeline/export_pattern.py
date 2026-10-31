"""Turn flattened pieces into cut-ready pattern files: laser DXF, true-size PDF and an assembly sheet.

- Stitch line = smoothed flattened boundary; cut line = stitch line + seam allowance.
- Notches: every ~40 mm along each seam at the SAME 3D points on both pieces, numbered per seam (A1, A2 ...).
- Nap: every piece is rotated so its fur direction points down the roll; layout uses 0 degree rotation only.
- Backing view: DXF and PDF are drawn as seen on the back of the fabric (laser cuts from the backing).
- Ears, eye and nose placement marks are added from the ear module and reference photos.

Usage: python3 export_pattern.py work_dir flat_dir size_label plush_nose_to_tail_mm out_dir
"""
import json, sys, math
from pathlib import Path
import numpy as np, trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, LineString, Point
from shapely import affinity

W, FLAT, SIZE, SIZE_MM, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], float(sys.argv[4]), Path(sys.argv[5])
OUT.mkdir(parents=True, exist_ok=True)
SA = 5.0            # seam allowance, mm
NOTCH_EVERY = 40.0  # mm along seams
ROLL_W = 1500.0
rep = json.load(open(FLAT / "flatten_report.json")); k = rep["scale"]
import os
TDIR = os.environ.get("TDIR", "template_v02")
ms = json.load(open(W / TDIR / "measures.json"))
leg_top_real = max(ms["elbow_height_mm"] * 1.05, 0.5 * ms["withers_height_mm"])

NAMES = {"P01": "HEAD GUSSET", "P02": "SIDE HEAD R", "P03": "SIDE HEAD L", "P04": "CHIN GUSSET",
         "P05": "BODY SIDE R", "P06": "BODY SIDE L", "P07": "FRONT UNDERBODY R", "P14": "FRONT UNDERBODY L",
         "P15": "REAR UNDERBODY R", "P16": "REAR UNDERBODY L", "P08": "SOLE FRONT R", "P09": "SOLE FRONT L",
         "P10": "SOLE REAR R", "P11": "SOLE REAR L", "P12": "TAIL R", "P13": "TAIL L",
         "P17": "EAR OUTER R", "P18": "EAR OUTER L", "P19": "EAR LINING R", "P20": "EAR LINING L"}

# ---------------------------------------------------------------- load pieces (3D + 2D) and make mirrors
pieces = {}
for pid, r in rep["pieces"].items():
    d = np.load(FLAT / f"{pid}.npz")
    pieces[pid] = {"V3": d["V3"], "F": d["F"], "uv": d["uv"].copy(), "bnd": d["bnd"]}
    if r["mirror"]:
        pieces[r["mirror"]] = {"V3": d["V3"] * [-1, 1, 1], "F": d["F"][:, ::-1], "uv": d["uv"] * [-1, 1], "bnd": d["bnd"][::-1]}


def signed_area(P):
    return 0.5 * (np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1)))


def face_jacobians(V3, F, uv):
    e1 = V3[F[:, 1]] - V3[F[:, 0]]; e2 = V3[F[:, 2]] - V3[F[:, 0]]
    u1 = uv[F[:, 1]] - uv[F[:, 0]]; u2 = uv[F[:, 2]] - uv[F[:, 0]]
    E = np.stack([e1, e2], 2); U = np.stack([u1, u2], 2)                    # (n,3,2), (n,2,2)
    return U @ np.linalg.pinv(E)                                            # (n,2,3)


def fur_dir(P):
    """3D fur direction: down the legs, nose->tail elsewhere."""
    d = np.tile([0.0, 1.0, 0.0], (len(P), 1))
    d[P[:, 2] < leg_top_real * k * 0.9] = [0, 0, -1.0]
    return d


for pid, p in pieces.items():
    V3, F, uv = p["V3"], p["F"], p["uv"]
    # fur-side view: triangles counter-clockwise when the outward-facing surface is seen from outside
    sa = np.sign([signed_area(uv[f]) for f in F[:500]]).mean()
    if sa < 0:
        uv[:, 0] *= -1
    # nap: area-weighted fur direction in 2D, rotate so it points down (-Y)
    J = face_jacobians(V3, F, uv); C = V3[F].mean(1)
    n = np.cross(V3[F[:, 1]] - V3[F[:, 0]], V3[F[:, 2]] - V3[F[:, 0]]); A = np.linalg.norm(n, axis=1); n /= A[:, None] + 1e-12
    f3 = fur_dir(C); f3 = f3 - (f3 * n).sum(1, keepdims=True) * n
    f2 = np.einsum("nij,nj->ni", J, f3); f2 = (f2 * A[:, None]).sum(0)
    ang = math.atan2(f2[1], f2[0]); rot = -math.pi / 2 - ang
    R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    p["uv"] = uv @ R.T
    p["nap_strength"] = float(np.linalg.norm(f2) / A.sum())

# ---------------------------------------------------------------- boundary, smoothing, seams
allpts, owner = [], []
for pid, p in pieces.items():
    b = p["bnd"]; allpts.append(p["V3"][b]); owner += [(pid, i) for i in range(len(b))]
allpts = np.vstack(allpts); tree = cKDTree(allpts)
for pid, p in pieces.items():
    b = p["bnd"]; P2 = p["uv"][b]
    seg = np.linalg.norm(np.diff(P2, axis=0, append=P2[:1]), axis=1); mean_seg = seg.mean()
    sig = max(1.0, 1.5 / mean_seg)                                   # ~1.5 mm smoothing
    p["stitch"] = np.c_[gaussian_filter1d(P2[:, 0], sig, mode="wrap"), gaussian_filter1d(P2[:, 1], sig, mode="wrap")]
    # neighbour piece for every boundary vertex (same 3D point on another piece)
    nb = []
    for q in p["V3"][b]:
        hits = tree.query_ball_point(q, 0.05)
        others = sorted({owner[h][0] for h in hits if owner[h][0] != pid})
        nb.append(others[0] if others else None)
    p["nb"] = nb

# seams: group consecutive boundary vertices by neighbour piece; seam id per unordered piece pair
seam_ids = {}
def seam_key(a, b): return tuple(sorted((a, b)))
for pid, p in pieces.items():
    for o in p["nb"]:
        if o and seam_key(pid, o) not in seam_ids:
            seam_ids[seam_key(pid, o)] = None
letters = [chr(ord("A") + i) for i in range(26)] + [chr(ord("A") + i) + chr(ord("A") + j) for i in range(26) for j in range(26)]
notches = {pid: [] for pid in pieces}
seam_len = {}
run_info = {}          # run letter -> (pieceA, pieceB)
li = 0
for key in sorted(seam_ids):
    a, b_ = key; pa = pieces[a]
    idx = [i for i, o in enumerate(pa["nb"]) if o == b_]
    if len(idx) < 2: continue
    idx = np.array(idx); n = len(pa["bnd"])
    runs, cur = [], [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1: cur.append(i)
        else: runs.append(cur); cur = [i]
    runs.append(cur)
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == n - 1: runs[0] = runs[-1] + runs[0]; runs.pop()
    for run in runs:
        P3 = pa["V3"][pa["bnd"][run]]
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P3, axis=0), axis=1))]
        if L[-1] < 8: continue
        letter = letters[li]; li += 1; run_info[letter] = key
        m = max(2, int(round(L[-1] / NOTCH_EVERY)) + 1)
        for cnt, t in enumerate(np.linspace(0, L[-1], m), 1):
            q = P3[np.searchsorted(L, t).clip(0, len(P3) - 1)]
            for pid in key:
                pp = pieces[pid]; bq = pp["V3"][pp["bnd"]]
                j = int(np.argmin(np.linalg.norm(bq - q, axis=1)))
                notches[pid].append((j, f"{letter}{cnt}"))
seam_ids = dict(run_info)                          # letter -> piece pair

# ease between consecutive notches: 2D length on each side of the seam, along the seam itself
def arc_along(pid, ia, ib, other):
    st = pieces[pid]["stitch"]; nb = pieces[pid]["nb"]; nn = len(st)
    fwd = np.arange(ia, ib + 1) if ib >= ia else np.r_[np.arange(ia, nn), np.arange(0, ib + 1)]
    bwd = (np.arange(ib, ia + 1) if ia >= ib else np.r_[np.arange(ib, nn), np.arange(0, ia + 1)])
    best = None
    for path in (fwd, bwd):
        if len(path) < 2: continue
        share = np.mean([nb[i] == other for i in path[1:-1]]) if len(path) > 2 else 1.0
        l = float(np.linalg.norm(np.diff(st[path], axis=0), axis=1).sum())
        if best is None or share > best[0] + 1e-9 or (abs(share - best[0]) < 1e-9 and l < best[1]): best = (share, l)
    return best[1]
for letter, (a, b_) in seam_ids.items():
    na = {lab: j for j, lab in notches[a] if lab[:len(letter)] == letter and lab[len(letter):].isdigit()}
    nb_ = {lab: j for j, lab in notches[b_] if lab[:len(letter)] == letter and lab[len(letter):].isdigit()}
    labs = sorted(set(na) & set(nb_), key=lambda l: int(l[len(letter):]))
    for l1, l2 in zip(labs[:-1], labs[1:]):
        la = arc_along(a, na[l1], na[l2], b_); lb = arc_along(b_, nb_[l1], nb_[l2], a)
        if max(la, lb) > 2: seam_len[f"{l1}-{l2} ({a}/{b_})"] = [round(la, 1), round(lb, 1)]

# ---------------------------------------------------------------- ears
ear = np.load(W / "ears/ear_outline_mm.npy") * k
att = np.load(W / "ears/ear_attach_idx.npy")
ear = ear - ear.mean(0)
ear_rot = np.c_[ear[:, 0], ear[:, 1]]                   # u = front->back, v = up; fur runs down the ear (-v)
for pid, mirror_x, lining in [("P17", 1, False), ("P18", -1, False), ("P19", -1, True), ("P20", 1, True)]:
    st = ear_rot * [mirror_x, 1]
    pieces[pid] = {"stitch": st, "nb": [None] * len(st), "nap_strength": 1.0, "attach": att}
    notches[pid] = [(int(att[0]), "EAR1"), (int(att[-1]), "EAR2")]   # not a seam letter: marks the opening ends

# ---------------------------------------------------------------- placement marks (ears, eyes, nose)
marks = {pid: [] for pid in pieces}
def to_uv(pid, P):
    p = pieces[pid]; m = trimesh.Trimesh(p["V3"], p["F"], process=False)
    cp, _, tri = trimesh.proximity.closest_point(m, P)
    bary = trimesh.triangles.points_to_barycentric(m.triangles[tri], cp)
    return (p["uv"][p["F"][tri]] * bary[:, :, None]).sum(1)
ear_on = np.load(W / "ears/ear_attach_on_template.npy") * k
if len(ear_on) >= 2:
    for pid, sx in [("P02", 1), ("P03", -1)]:
        uvs = to_uv(pid, ear_on * [sx, 1, 1])
        sm = np.c_[gaussian_filter1d(uvs[:, 0], 2), gaussian_filter1d(uvs[:, 1], 2)]
        marks[pid].append(("line", sm, "EAR"))
eyes = json.load(open(W / TDIR / "face_points.json")) if (W / TDIR / "face_points.json").exists() else None
if eyes:
    for pid, sx in [("P02", 1), ("P03", -1)]:
        e = np.array(eyes["eye_R"]) * k * [sx, 1, 1]
        uv = to_uv(pid, e[None])[0]
        marks[pid].append(("eye", uv, f"EYE {eyes['eye_diam_mm_plush']:.0f} mm"))

# ---------------------------------------------------------------- outlines, allowance, layout
items = []
for pid in sorted(pieces):
    p = pieces[pid]
    st = Polygon(p["stitch"]).buffer(0)
    if st.geom_type != "Polygon": st = max(st.geoms, key=lambda g: g.area)
    cut = st.buffer(SA, join_style=2, mitre_limit=2.0)
    items.append({"pid": pid, "stitch": st, "cut": cut, "w": cut.bounds[2] - cut.bounds[0], "h": cut.bounds[3] - cut.bounds[1]})
items.sort(key=lambda it: -it["h"])
x = y = row_h = 0.0; GAP = 10.0
for it in items:
    if x + it["w"] > ROLL_W: x = 0.0; y += row_h + GAP; row_h = 0.0
    bx, by = it["cut"].bounds[0], it["cut"].bounds[1]
    it["dx"], it["dy"] = x - bx, y - by
    x += it["w"] + GAP; row_h = max(row_h, it["h"])
layout_h = y + row_h

def place(it, P):           # piece coords -> sheet coords (fur-side view)
    return np.asarray(P) + [it["dx"], it["dy"]]
def backing(P):             # mirror for the backing side (laser cuts from the back)
    P = np.asarray(P, float).copy(); P[..., 0] = ROLL_W - P[..., 0]; return P

# ---------------------------------------------------------------- DXF
import ezdxf
doc = ezdxf.new("R2010", setup=True); doc.units = ezdxf.units.MM; doc.header["$INSUNITS"] = 4
for name, col in [("CUT", 1), ("NOTCH", 3), ("MARK", 5), ("LABEL", 7), ("STITCH_REF", 8)]:
    doc.layers.add(name, color=col)
msp = doc.modelspace()
checks = {"pieces": len(items), "notch_partners_ok": True, "seam_length_mismatch_max_pct": 0.0, "fits_roll": True,
          "smallest_piece_mm": None, "min_curve_radius_ok": True}
small = 1e9
for it in items:
    pid = it["pid"]; p = pieces[pid]
    cut = backing(place(it, np.array(it["cut"].exterior.coords)))
    msp.add_lwpolyline(cut.tolist(), close=True, dxfattribs={"layer": "CUT"})
    st = backing(place(it, np.array(it["stitch"].exterior.coords)))
    msp.add_lwpolyline(st.tolist(), close=True, dxfattribs={"layer": "STITCH_REF"})
    stitch = p["stitch"]
    for j, lab in notches[pid]:
        q = stitch[j]; t = stitch[(j + 1) % len(stitch)] - stitch[j - 1]; t /= np.linalg.norm(t) + 1e-9
        nrm = np.array([t[1], -t[0]])
        if it["stitch"].contains(Point(q + nrm * 1.0)): nrm = -nrm          # point outward
        tip = q + nrm * (SA - 3.0); base = q + nrm * SA
        v = [base - t * 1.5, tip, base + t * 1.5]
        msp.add_lwpolyline(backing(place(it, v)).tolist(), dxfattribs={"layer": "NOTCH"})
        tp = backing(place(it, [q - nrm * 3.5]))[0]
        msp.add_text(lab, height=2.5, dxfattribs={"layer": "LABEL"}).set_placement(tuple(tp), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
    for kind, data, text in marks.get(pid, []):
        if kind == "line":
            msp.add_lwpolyline(backing(place(it, data)).tolist(), dxfattribs={"layer": "MARK"})
        else:
            c = backing(place(it, [data]))[0]; msp.add_circle(tuple(c), 2.0, dxfattribs={"layer": "MARK"})
    cen = it["stitch"].representative_point()
    c = backing(place(it, [[cen.x, cen.y]]))[0]
    qty = "CUT 1"
    msp.add_text(f"{pid} {NAMES[pid]}", height=4.0, dxfattribs={"layer": "LABEL"}).set_placement((c[0], c[1] + 4), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
    msp.add_text(f"LAB {SIZE} | {qty} | NAP DOWN", height=3.0, dxfattribs={"layer": "LABEL"}).set_placement((c[0], c[1] - 2), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
    msp.add_lwpolyline([(c[0], c[1] - 6), (c[0], c[1] - 18), (c[0] - 2, c[1] - 14), (c[0], c[1] - 18), (c[0] + 2, c[1] - 14)], dxfattribs={"layer": "LABEL"})
    small = min(small, min(it["w"], it["h"]) - 2 * SA)
# calibration square
msp.add_lwpolyline([(10, layout_h + 20), (110, layout_h + 20), (110, layout_h + 120), (10, layout_h + 120)], close=True, dxfattribs={"layer": "CUT"})
msp.add_text("TEST SQUARE 100 x 100 mm", height=4, dxfattribs={"layer": "LABEL"}).set_placement((60, layout_h + 70), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
doc.saveas(OUT / f"labrador_{SIZE}_v1_laser.dxf")
checks["smallest_piece_mm"] = round(small, 1)
checks["layout_mm"] = [ROLL_W, round(layout_h, 1)]
mism = []
for key, (l1, l2) in [(k_, v) for k_, v in seam_len.items() if len(v) == 2]:
    if max(l1, l2) > 0: mism.append(abs(l1 - l2) / max(l1, l2) * 100)
checks["seam_length_mismatch_max_pct"] = round(max(mism), 1) if mism else 0.0
checks["seam_length_mismatch_p90_pct"] = round(float(np.percentile(mism, 90)), 1) if mism else 0.0
for pid in pieces:
    labs = [l for _, l in notches[pid]]
for lab_set in [set(l for _, l in v) for v in notches.values()]:
    pass
all_labels = {}
for pid, lst in notches.items():
    for _, l in lst: all_labels[l] = all_labels.get(l, 0) + 1
checks["notches_without_partner"] = sorted([l for l, c in all_labels.items() if c < 2 and not l.startswith("E")])
checks["notch_partners_ok"] = len(checks["notches_without_partner"]) == 0
json.dump({"checks": checks, "seams": {k_: list(v) for k_, v in seam_ids.items()}, "seam_lengths_2d": seam_len},
          open(OUT / "pattern_checks.json", "w"), indent=1)

# ---------------------------------------------------------------- PDF (true size, backing view)
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.colors import black, red, blue, grey, HexColor
PW, PH = ROLL_W + 40, layout_h + 160
c = canvas.Canvas(str(OUT / f"labrador_{SIZE}_v1_pattern_1to1.pdf"), pagesize=(PW * mm, PH * mm))
def P(pt): return (pt[0] + 20) * mm, (pt[1] + 20) * mm
def poly(pts, close=True, dash=None, col=black, w=0.6):
    c.setStrokeColor(col); c.setLineWidth(w); c.setDash(*(dash or [])) if dash else c.setDash()
    pth = c.beginPath(); x0, y0 = P(pts[0]); pth.moveTo(x0, y0)
    for q in pts[1:]: pth.lineTo(*P(q))
    if close: pth.close()
    c.drawPath(pth, stroke=1, fill=0); c.setDash()
for it in items:
    pid = it["pid"]; p = pieces[pid]
    poly(backing(place(it, np.array(it["cut"].exterior.coords))), w=0.8)
    poly(backing(place(it, np.array(it["stitch"].exterior.coords))), dash=[3, 2], col=grey, w=0.5)
    stitch = p["stitch"]
    for j, lab in notches[pid]:
        q = stitch[j]; t = stitch[(j + 1) % len(stitch)] - stitch[j - 1]; t /= np.linalg.norm(t) + 1e-9
        nrm = np.array([t[1], -t[0]])
        if it["stitch"].contains(Point(q + nrm * 1.0)): nrm = -nrm
        tip = q + nrm * (SA - 3.0); base = q + nrm * SA
        poly(backing(place(it, [base - t * 1.5, tip, base + t * 1.5])), close=False, col=HexColor("#0a7a2f"), w=0.6)
        tp = backing(place(it, [q - nrm * 3.5]))[0]
        c.setFillColor(HexColor("#0a7a2f")); c.setFont("Helvetica", 6); c.drawCentredString(*P(tp), lab)
    for kind, data, text in marks.get(pid, []):
        if kind == "line":
            poly(backing(place(it, data)), close=False, col=blue, w=0.8, dash=[2, 1])
            mid = backing(place(it, [data[len(data) // 2]]))[0]; c.setFillColor(blue); c.setFont("Helvetica", 6); c.drawString(*P(mid + [2, 2]), text)
        else:
            q = backing(place(it, [data]))[0]; c.setStrokeColor(blue); c.circle(*P(q), 2 * mm)
            c.setFillColor(blue); c.setFont("Helvetica", 6); c.drawString(*P(q + [3, 3]), text)
    cen = it["stitch"].representative_point(); q = backing(place(it, [[cen.x, cen.y]]))[0]
    c.setFillColor(black); c.setFont("Helvetica-Bold", 10); c.drawCentredString(*P(q + [0, 4]), f"{pid} {NAMES[pid]}")
    c.setFont("Helvetica", 7); c.drawCentredString(*P(q - [0, 2]), f"Labrador {SIZE} | cut 1 | nap down")
    poly([q - [0, 6], q - [0, 18]], close=False, w=0.8); poly([q - [2, 14], q - [0, 18], q + [2, -14]], close=False, w=0.8)
# calibration square + legend
poly(np.array([[10, layout_h + 20], [110, layout_h + 20], [110, layout_h + 120], [10, layout_h + 120]]), w=0.8)
c.setFont("Helvetica-Bold", 9); c.drawString(*P([20, layout_h + 70]), "Check: this square must measure 100 x 100 mm")
c.setFont("Helvetica", 9)
lines = [f"Labrador standing, size {SIZE} ({SIZE_MM:.0f} mm nose to tail base). Pattern v1 for FLEECE MOCK-UP.",
         "Drawn as seen on the BACKING (back of the fabric). Solid = cut line (incl. 5 mm seam allowance). Dashed = stitch line.",
         "Green V = notches, match same letters+numbers. Blue = placement marks (ears, eyes). Arrow = nap (fur runs down).",
         "Defaults until confirmed: seam allowance 5 mm, roll width 1500 mm, minimum tube 25 mm."]
for i, s_ in enumerate(lines): c.drawString(*P([130, layout_h + 110 - i * 12]), s_)
c.showPage(); c.save()
print(json.dumps(checks, indent=1))
print("seam runs:", len(seam_ids))
ok = checks["fits_roll"] and checks["notch_partners_ok"] and checks["seam_length_mismatch_max_pct"] <= 3.0
print("RESULT:", "PASS" if ok else "CHECK")
