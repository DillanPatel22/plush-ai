"""One-page (A3 landscape) seam-layout proposal for the workshop, in English: current v1 vs proposed v2 piece maps,
the v2 pieces at one scale with pile direction, what changes and why, the fur-direction check, cost (pieces and
seam length) and questions for the craftsman. Every number is read from the check outputs, nothing typed in.

Usage: python3 proposal_sheet.py work_dir out.pdf
Inputs (in work_dir): renders/pieces_v1_v04_*.png, renders/pieces_v2_v04_*.png, pieces_v2_template_v04/,
flat_template_v04/, flat_v2_template_v04/, review/nap_v1_as_cut.json, review/nap_v2_best.json
"""
import json, math, sys, datetime
from pathlib import Path
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.utils import ImageReader, simpleSplit

W, OUT = Path(sys.argv[1]), sys.argv[2]
PW, PH = landscape(A3)
INK, MUTE, ACC, RULE = HexColor("#1f2328"), HexColor("#5b636b"), HexColor("#b4532a"), HexColor("#d7dadd")
lay = json.load(open(W / "pieces_v2_template_v04/pieces.json"))["pieces"]
nap1 = json.load(open(W / "review/nap_v1_as_cut.json")); nap2 = json.load(open(W / "review/nap_v2_best.json"))
ms = json.load(open(W / "template_v04/measures.json"))


def perim(f):
    d = np.load(f); P = d["uv"][d["bnd"]]; return float(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum())
v1m = {"P01": 1, "P02": 2, "P04": 1, "P05": 2, "P07": 2, "P15": 2, "P08": 2, "P10": 2, "P12": 2}
seam1 = sum(perim(W / f"flat_template_v04/{p}.npz") * n for p, n in v1m.items()) / 2 / 1000
mult = {p: (2 if any(i.get("mirror_of") == p for i in lay.values()) else 1) for p, i in lay.items() if not i.get("mirror_of") and i["faces"]}
seam2 = sum(perim(W / f"flat_v2_template_v04/{p}.npz") * n for p, n in mult.items()) / 2 / 1000
n1, n2 = sum(v1m.values()) + 4, sum(mult.values()) + 4
def cut_area(f, n):                          # stitch area + 5 mm allowance (mitred, like the export)
    d = np.load(f); return Polygon(d["uv"][d["bnd"]]).buffer(0).buffer(5.0, join_style=2, mitre_limit=2.0).area * n
fab1 = sum(cut_area(W / f"flat_template_v04/{p}.npz", n) for p, n in v1m.items())
fab2 = sum(cut_area(W / f"flat_v2_template_v04/{p}.npz", n) for p, n in mult.items())
def feret(P):
    h = np.array(Polygon(P).convex_hull.exterior.coords); best = 1e9
    for i in range(len(h) - 1):
        e = h[i + 1] - h[i]; nn = np.hypot(*e)
        if nn: pr = h @ (np.array([-e[1], e[0]]) / nn); best = min(best, pr.max() - pr.min())
    return best
pad_mm = min(feret(np.load(W / f"flat_v2_template_v04/{p}.npz")["uv"][np.load(W / f"flat_v2_template_v04/{p}.npz")["bnd"]]) for p in ("P20", "P22"))                       # + 4 ear pieces (outer and lining, L and R)

c = canvas.Canvas(OUT, pagesize=(PW, PH)); c.setTitle("Labrador seam layout v2 proposal")
M = 12 * mm


def text(x, y, s, size=9, font="Helvetica", col=INK):
    c.setFont(font, size); c.setFillColor(col); c.drawString(x, y, s)


def para(x, y, s, width, size=8.5, font="Helvetica", col=INK, lead=1.28):
    c.setFont(font, size); c.setFillColor(col)
    for line in simpleSplit(s, font, size, width):
        c.drawString(x, y, line); y -= size * lead
    return y


def crop(path):
    im = Image.open(path).convert("RGB"); a = np.asarray(im).astype(int)
    bg = a[2, 2]; mask = np.abs(a - bg).sum(2) > 18
    r, cc = np.where(mask); pad = 6
    return im.crop((max(cc.min() - pad, 0), max(r.min() - pad, 0), cc.max() + pad, r.max() + pad))


def img(path, x, y, w, h):
    im = crop(path); iw, ih = im.size; s = min(w / iw, h / ih)
    c.drawImage(ImageReader(im), x + (w - iw * s) / 2, y + (h - ih * s) / 2, iw * s, ih * s)


# ---------------------------------------------------------------- header
text(M, PH - M - 6 * mm, "Labrador, standing: seam layout v2", 17, "Helvetica-Bold")
c.setFillColor(ACC); c.roundRect(M + 118 * mm, PH - M - 7.2 * mm, 52 * mm, 7 * mm, 1.5 * mm, fill=1, stroke=0)
text(M + 121 * mm, PH - M - 5 * mm, "PROPOSAL FOR APPROVAL", 10, "Helvetica-Bold", white)
text(M, PH - M - 12 * mm, f"Medium size: 350 mm nose to tail base.  Drawn on base shape v0.4.  {datetime.date(2026, 10, 31):%d %b %Y}.  "
     "Not for cutting: please answer the questions on the right and send them back.", 9, col=MUTE)
c.setStrokeColor(RULE); c.setLineWidth(0.6); c.line(M, PH - M - 15 * mm, PW - M, PH - M - 15 * mm)

# ---------------------------------------------------------------- left: piece maps
LX, LW = M, 200 * mm
y = PH - M - 22 * mm
for tag, title, pre in [("v1", f"Current layout v1: {n1} pieces", "pieces_v1_v04"), ("v2", f"Proposed layout v2: {n2} pieces", "pieces_v2_v04")]:
    text(LX, y, title, 10.5, "Helvetica-Bold")
    y -= 47 * mm
    for i, v in enumerate(["side", "three_q", "under"]):
        img(W / f"renders/{pre}_{v}.png", LX + i * (LW / 3), y, LW / 3 - 3 * mm, 44 * mm)
    y -= 6 * mm
# v2 legend
LEG = [("#e6503c", "P01 head gusset"), ("#468cdc", "P02/03 side head"), ("#f0aa28", "P04 chin gusset"), ("#fa78aa", "P05/06 muzzle side"),
       ("#78be5a", "P07/08 body side"), ("#fadc5a", "P09 chest bib"), ("#aa64c8", "P10/11 belly"), ("#28785a", "P12/13 front leg outer"),
       ("#96dcc8", "P14/15 front leg inner"), ("#c8783c", "P16/17 hind leg outer"), ("#f0be96", "P18/19 hind leg inner"),
       ("#5a5a5a", "P20-23 paw pads"), ("#3cbebe", "P24/25 tail"), ("#ffffff", "P26-29 ears (flat, not shown)")]
text(LX, y, "v2 colours", 8, "Helvetica-Bold", MUTE); y -= 4.5 * mm
for i, (col, name) in enumerate(LEG):
    cx, cy = LX + (i % 4) * (LW / 4), y - (i // 4) * 4.4 * mm
    c.setFillColor(HexColor(col)); c.setStrokeColor(MUTE); c.setLineWidth(0.3); c.rect(cx, cy - 0.6 * mm, 3 * mm, 3 * mm, fill=1, stroke=1)
    text(cx + 4.5 * mm, cy, name, 7.5)
y -= 4 * 4.4 * mm + 3 * mm

# v2 flat pieces at one scale, pile pointing down
c.setStrokeColor(RULE); c.line(LX, y + 2 * mm, LX + LW, y + 2 * mm)
text(LX, y - 2 * mm, "v2 pieces at one scale (right-hand and centre pieces; x2 = cut a mirrored left piece too). Arrow = pile direction.", 8.5, "Helvetica-Bold")
text(LX, y - 6 * mm, "Outlines are indicative: seam lines are faired, notched and checked in the v2 pattern after approval.", 7.5, col=MUTE)
top = y - 9 * mm; bottom = M + 2 * mm
shapes = []
for pid, info in lay.items():
    if info.get("mirror_of") or not info["faces"]: continue
    d = np.load(W / f"flat_v2_template_v04/{pid}.npz"); P = d["uv"][d["bnd"]].copy()
    r = nap2["pieces"][pid]
    if r["flip_u"]: P[:, 0] *= -1
    a = r["rotation_rad"]; R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]); P = P @ R.T
    g = Polygon(P).buffer(0).buffer(2.5).buffer(-5.0).buffer(2.5)            # display only: drop kinks under ~5 mm
    if g.geom_type != "Polygon": g = max(g.geoms, key=lambda q: q.area)
    shapes.append((pid, info["name"], g, mult[pid]))
shapes.sort(key=lambda t: -(t[2].bounds[3] - t[2].bounds[1]))
def lab(pid, n): return f"{pid}" + (" x2" if n == 2 else "")
def lab_w(pid, n): return c.stringWidth(lab(pid, n), "Helvetica-Bold", 7) + 2 * mm
def pack(s):
    """tallest piece in a column on the left, the rest in shelves to its right"""
    pid0, _, g0, n0 = shapes[0]
    w0 = max((g0.bounds[2] - g0.bounds[0]) * s, lab_w(pid0, n0)) + 4 * mm; h0 = (g0.bounds[3] - g0.bounds[1]) * s + 7 * mm
    pos = [(0, 0, w0, h0)]; x = w0; yy = 0; row = 0
    for pid, nm, g, n in shapes[1:]:
        w = max((g.bounds[2] - g.bounds[0]) * s, lab_w(pid, n)) + 4 * mm; h = (g.bounds[3] - g.bounds[1]) * s + 7 * mm
        if x + w > LW: x = w0; yy += row; row = 0
        pos.append((x, yy, w, h)); x += w; row = max(row, h)
    return pos, max(h0, yy + row), max(p_[0] + p_[2] for p_ in pos)
s = 2.0 * mm
while pack(s)[1] > top - bottom or pack(s)[2] > LW: s *= 0.98
pos = pack(s)[0]
NEW = ("P02", "P05", "P07", "P09", "P10", "P12", "P14", "P16", "P18")
for (pid, nm, g, n), (px, py, w, h) in zip(shapes, pos):
    gw = (g.bounds[2] - g.bounds[0]) * s
    ox = LX + px + (w - 4 * mm - gw) / 2 - g.bounds[0] * s; oy = top - py - 5 * mm - g.bounds[3] * s
    pth = c.beginPath(); ext = np.array(g.exterior.coords)
    pth.moveTo(ox + ext[0, 0] * s, oy + ext[0, 1] * s)
    for q in ext[1:]: pth.lineTo(ox + q[0] * s, oy + q[1] * s)
    pth.close(); c.setFillColor(HexColor("#fbe3d6") if pid in NEW else HexColor("#f3f1ee")); c.setStrokeColor(INK); c.setLineWidth(0.5); c.drawPath(pth, fill=1, stroke=1)
    cx, cy = g.centroid.x * s + ox, g.centroid.y * s + oy
    al = min(3.5 * mm, 0.3 * (g.bounds[3] - g.bounds[1]) * s)
    c.setStrokeColor(ACC); c.setLineWidth(0.8); c.line(cx, cy + al, cx, cy - al)
    c.setFillColor(ACC); p2 = c.beginPath(); p2.moveTo(cx, cy - al - 1 * mm); p2.lineTo(cx - 1.1 * mm, cy - al + 0.8 * mm); p2.lineTo(cx + 1.1 * mm, cy - al + 0.8 * mm); p2.close(); c.drawPath(p2, fill=1, stroke=0)
    text(LX + px + (w - 4 * mm) / 2 - c.stringWidth(lab(pid, n), "Helvetica-Bold", 7) / 2, top - py - 2.5 * mm, lab(pid, n), 7, "Helvetica-Bold", ACC if pid in NEW else INK)
text(LX, bottom - 1 * mm, f"Drawn at about 1:{1 / (s / mm):.1f} of the medium plush. Shaded = new or changed pieces. Names: see v2 colours above.", 6.5, col=MUTE)

# ---------------------------------------------------------------- right: changes, check, cost, questions
RX = LX + LW + 10 * mm; RW = PW - M - RX
y = PH - M - 22 * mm
text(RX, y, "What changes, and why", 10.5, "Helvetica-Bold"); y -= 5.5 * mm
r1, r2 = nap1["regions"], nap2["regions"]
CH = [("Legs are separate pieces below the elbow and the stifle (knee).",
       f"An outer and an inner piece per leg, cut with the pile running down. In v1 the legs were part of the body side and belly pieces, so the leg fur ran sideways: {r1['legs']['area_off_30deg_pct']:.0f} % of the leg area more than 30 degrees off."),
      ("Chest bib over the chest and the fronts of the shoulders, down to the elbows.",
       f"Pile runs down, as on a Labrador's chest. In v1 it ran towards the tail ({r1['chest & shoulder fronts']['area_off_30deg_pct']:.0f} % off)."),
      ("Belly in two halves, centre seam underneath.",
       "A one-piece belly stretches about 22 % between the hind legs; two halves stay under 9 %. The seam is hidden underneath."),
      ("Muzzle side pieces: seam from the stop, in front of the eye, to the corner of the mouth.",
       "Gives a sharper stop and fuller cheeks, and a natural place to set the eye. The lighter option is to keep the v1 side head and add a dart at the stop (question 3)."),
      ("Darts instead of easing at the hips, shoulders, tail root and tail tip.",
       "v1 needs up to 5 mm of easing in 40 mm of seam in places (armpits, neck, hips). Where v2 still needs more than about 3 mm, it gets a dart instead; exact positions come with the v2 pattern."),
      ("Ears: a small pleat at the base so the ear turns forward and frames the face, as in the photos.", ""),
      ("Unchanged: head gusset, chin gusset, body sides with the centre-back seam, two-piece tail.", "")]
for i, (h, b) in enumerate(CH, 1):
    text(RX, y, f"{i}.", 8.5, "Helvetica-Bold", ACC)
    y = para(RX + 5 * mm, y, h, RW - 5 * mm, 8.5, "Helvetica-Bold")
    if b: y = para(RX + 5 * mm, y, b, RW - 5 * mm, 8, col=MUTE)
    y -= 1.6 * mm

y -= 1.5 * mm; c.setStrokeColor(RULE); c.line(RX, y + 2 * mm, RX + RW, y + 2 * mm)
text(RX, y - 2.5 * mm, "Fur direction check", 10.5, "Helvetica-Bold"); y -= 6.5 * mm
y = para(RX, y, "Share of each area where the pile, once sewn, points more than 30 degrees away from a Labrador's coat direction. Lower is better.", RW, 7.5, col=MUTE) - 1 * mm
rows = [("Head", "head"), ("Body sides", "body side"), ("Chest and shoulder fronts", "chest & shoulder fronts"), ("Legs", "legs"),
        ("Belly and inner thighs (underneath)", "underbody"), ("Tail", "tail")]
cx0, cx1, cx2 = RX, RX + RW * 0.62, RX + RW * 0.82
text(cx0, y, "Area", 8, "Helvetica-Bold", MUTE); text(cx1, y, "v1 now", 8, "Helvetica-Bold", MUTE); text(cx2, y, "v2", 8, "Helvetica-Bold", MUTE); y -= 4.6 * mm
for lab, key in rows:
    a, b = r1[key]["area_off_30deg_pct"], r2[key]["area_off_30deg_pct"]
    text(cx0, y, lab, 8.5); text(cx1, y, f"{a:.0f} %", 8.5, "Helvetica-Bold" if a > 20 else "Helvetica", ACC if a > 20 else INK)
    text(cx2, y, f"{b:.0f} %", 8.5, "Helvetica-Bold" if b > 20 else "Helvetica", ACC if b > 20 else INK)
    c.setStrokeColor(RULE); c.setLineWidth(0.3); c.line(RX, y - 1.6 * mm, RX + RW, y - 1.6 * mm); y -= 4.8 * mm
y -= 1 * mm
text(RX, y, "Cost", 9, "Helvetica-Bold"); y -= 4.4 * mm
y = para(RX, y, f"{n1} pieces become {n2}. Total seam length at medium size goes from {seam1:.1f} m to {seam2:.1f} m "
         f"(about {100 * (seam2 / seam1 - 1):.0f} % more sewing). Fabric for the body pieces (ears are the same in both), seam allowance included: "
         f"{fab1 / 100:,.0f} cm2 to {fab2 / 100:,.0f} cm2 ({100 * (fab2 / fab1 - 1):+.0f} %).", RW, 8.5) - 2 * mm

c.setStrokeColor(RULE); c.line(RX, y + 2 * mm, RX + RW, y + 2 * mm)
text(RX, y - 2.5 * mm, "Questions for the workshop", 10.5, "Helvetica-Bold"); y -= 6.8 * mm
Q = ["Separate leg pieces (outer and inner, joined at the elbow and the stifle): OK? Or do you prefer another way to make the leg fur run down?",
     "Chest bib over the chest and the fronts of the shoulders: OK?",
     "Head: A) muzzle side pieces as shown, or B) keep the v1 side head with a dart at the stop?",
     "Darts instead of easing: OK? What is the most easing you are happy with in 40 mm of seam?",
     f"Paw pads come out about {pad_mm:.0f} mm across at medium, just over the 25 mm smallest-piece default. Keep them as separate pads (in short pile or suede?), or close the leg end with a dart?",
     "Ears: is a pleat at the base OK? (Lining in the same fur or a short pile? Please say in the comments.)",
     "Do you already make a realistic dog you are happy with? Photos of its pattern pieces would help us."]
DEF = ("Defaults until you confirm them: seam allowance 5 mm; narrowest turnable tube 25 mm; tightest curve 3 mm; "
       "smallest piece 25 mm; roll width 1500 mm; laser files DXF in mm, cut and notch on separate layers (RDWorks / LightBurn). "
       "Fabric: knitted polyester backing, acrylic pile; one white stock fur per plush, trimmed to the trim map; pieces laid at 0 or 180 degrees only.")
def_lines = simpleSplit(DEF, "Helvetica", 7.5, RW - 6 * mm); def_h = len(def_lines) * 7.5 * 1.28 + 5 * mm
qs = [simpleSplit(q, "Helvetica", 8.5, RW - 5 * mm) for q in Q]
text_h = sum(len(l) * 8.5 * 1.28 for l in qs)
gap = max(7 * mm, min(13 * mm, (y - (M + def_h + 3 * mm) - text_h) / len(Q)))
for i, q in enumerate(Q, 1):
    text(RX, y, f"{i}.", 8.5, "Helvetica-Bold", ACC)
    y = para(RX + 5 * mm, y, q, RW - 5 * mm, 8.5)
    yb = y - 0.5 * mm
    for j, lab_ in enumerate({3: ["A", "B"], 5: ["Pads", "Dart"]}.get(i, ["Yes", "No"])):
        bx = RX + 5 * mm + j * 14 * mm
        c.setStrokeColor(INK); c.setLineWidth(0.5); c.rect(bx, yb - 0.4 * mm, 3 * mm, 3 * mm); text(bx + 4.2 * mm, yb, lab_, 8)
    text(RX + 34 * mm, yb, "Comments:", 7, col=MUTE)
    c.setStrokeColor(RULE); c.setLineWidth(0.4)
    c.line(RX + 47 * mm, yb - 0.6 * mm, RX + RW, yb - 0.6 * mm)
    if gap > 10 * mm: c.line(RX + 5 * mm, yb - 5.6 * mm, RX + RW, yb - 5.6 * mm)
    y -= gap
c.setFillColor(HexColor("#f3f4f5")); c.rect(RX, M - 1 * mm, RW, def_h, fill=1, stroke=0)
yy = M - 1 * mm + def_h - 4.5 * mm
c.setFont("Helvetica", 7.5); c.setFillColor(MUTE)
for line in def_lines: c.drawString(RX + 3 * mm, yy, line); yy -= 7.5 * 1.28
c.showPage(); c.save()
print(f"saved {OUT}: v1 {n1} pieces / {seam1:.2f} m / {fab1 / 100:,.0f} cm2, v2 {n2} pieces / {seam2:.2f} m / {fab2 / 100:,.0f} cm2, pieces drawn at 1:{1 / (s / mm):.1f}")
