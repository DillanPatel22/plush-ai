"""Two-page (A3 landscape) English proposal for the workshop: seam layout v2.1 (competitor-style legs) and the finish
plan that makes the plush look like the dog (colour and markings from the photos, fur trim zones, size).
Every number is read from the check outputs.

Usage: python3 proposal_sheet_v21.py work_dir out.pdf
"""
import json, math, sys, datetime
from pathlib import Path
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader, simpleSplit

W, OUT = Path(sys.argv[1]), sys.argv[2]
PW, PH = landscape(A3); M = 12 * mm
INK, MUTE, ACC, RULE = HexColor("#1f2328"), HexColor("#5b636b"), HexColor("#b4532a"), HexColor("#d7dadd")
lay = json.load(open(W / "pieces_v21_template_v04/pieces.json"))["pieces"]
nap1 = json.load(open(W / "review/nap_v1_as_cut.json")); nap2 = json.load(open(W / "review/nap_v21_visible.json"))
flat2 = json.load(open(W / "flat_v21_template_v04/flatten_report.json"))["pieces"]
zones = json.load(open(W / "review/labrador_v21_colour_zones.json")); spec = json.load(open(W / "finish_spec.json"))
thin = json.load(open(W / "review/thin_canonical_350.json"))
DATE = datetime.date(2026, 11, 3)


def perim(f):
    d = np.load(f); P = d["uv"][d["bnd"]]; return float(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum())
def cut_area(f, n):
    d = np.load(f); return Polygon(d["uv"][d["bnd"]]).buffer(0).buffer(5.0, join_style=2, mitre_limit=2.0).area * n
v1m = {"P01": 1, "P02": 2, "P04": 1, "P05": 2, "P07": 2, "P15": 2, "P08": 2, "P10": 2, "P12": 2}
mult = {p: (2 if any(i.get("mirror_of") == p for i in lay.values()) else 1) for p, i in lay.items() if not i.get("mirror_of") and i["faces"]}
seam1 = sum(perim(W / f"flat_template_v04/{p}.npz") * n for p, n in v1m.items()) / 2000
seam2 = sum(perim(W / f"flat_v21_template_v04/{p}.npz") * n for p, n in mult.items()) / 2000
fab1 = sum(cut_area(W / f"flat_template_v04/{p}.npz", n) for p, n in v1m.items()) / 100
fab2 = sum(cut_area(W / f"flat_v21_template_v04/{p}.npz", n) for p, n in mult.items()) / 100
n1, n2 = sum(v1m.values()) + 4, sum(mult.values()) + 4
max_stretch = max(r["area_dev_p95"] for p, r in flat2.items() if not p.startswith(("P13", "P14", "P15", "P16", "P17", "P18")))
legs_w = min(s["plush_width_mm"] for s in thin["legs"]); tail_w = min(s["plush_width_mm"] for s in thin["tail"])
size_true = math.ceil(350 * 25 / min(legs_w, tail_w) / 10) * 10

c = canvas.Canvas(OUT, pagesize=(PW, PH)); c.setTitle("Labrador seam layout v2.1 and finish plan")


def text(x, y, s, size=9, font="Helvetica", col=INK):
    c.setFont(font, size); c.setFillColor(col); c.drawString(x, y, s)
def para(x, y, s, width, size=8.5, font="Helvetica", col=INK, lead=1.28):
    c.setFont(font, size); c.setFillColor(col)
    for line in simpleSplit(s, font, size, width): c.drawString(x, y, line); y -= size * lead
    return y
def crop(path, bgtol=18):
    im = Image.open(path).convert("RGB"); a = np.asarray(im).astype(int); bg = a[2, 2]
    r, cc = np.where(np.abs(a - bg).sum(2) > bgtol); pad = 6
    return im.crop((max(cc.min() - pad, 0), max(r.min() - pad, 0), cc.max() + pad, r.max() + pad))
def img(path, x, y, w, h, do_crop=True):
    im = crop(path) if do_crop else Image.open(path).convert("RGB"); iw, ih = im.size; s = min(w / iw, h / ih)
    c.drawImage(ImageReader(im), x + (w - iw * s) / 2, y + (h - ih * s) / 2, iw * s, ih * s)
def header(title, sub):
    text(M, PH - M - 6 * mm, title, 17, "Helvetica-Bold")
    tw = c.stringWidth(title, "Helvetica-Bold", 17)
    c.setFillColor(ACC); c.roundRect(M + tw + 6 * mm, PH - M - 7.2 * mm, 52 * mm, 7 * mm, 1.5 * mm, fill=1, stroke=0)
    text(M + tw + 9 * mm, PH - M - 5 * mm, "PROPOSAL FOR APPROVAL", 10, "Helvetica-Bold", white)
    text(M, PH - M - 12 * mm, sub, 9, col=MUTE)
    c.setStrokeColor(RULE); c.setLineWidth(0.6); c.line(M, PH - M - 15 * mm, PW - M, PH - M - 15 * mm)
def questions(Q, x, y, width, start=1, choices=None, gap=10 * mm):
    for i, q in enumerate(Q, start):
        text(x, y, f"{i}.", 8.5, "Helvetica-Bold", ACC); y = para(x + 5 * mm, y, q, width - 5 * mm, 8.5); yb = y - 0.5 * mm
        for j, lab_ in enumerate((choices or {}).get(i, ["Yes", "No"])):
            bx = x + 5 * mm + j * 15 * mm
            c.setStrokeColor(INK); c.setLineWidth(0.5); c.rect(bx, yb - 0.4 * mm, 3 * mm, 3 * mm); text(bx + 4.2 * mm, yb, lab_, 8)
        text(x + 36 * mm, yb, "Comments:", 7, col=MUTE); c.setStrokeColor(RULE); c.setLineWidth(0.4); c.line(x + 49 * mm, yb - 0.6 * mm, x + width, yb - 0.6 * mm)
        y -= gap
    return y

SUB = (f"Medium size: {spec['plush_size_mm']} mm nose to tail base. Drawn on base shape v0.4. {DATE:%d %b %Y}. "
       "Not for cutting: please answer the questions and send them back.")
# =============================================================== page 1: layout
header("Labrador, standing: seam layout v2.1", SUB)
LX, LW = M, 200 * mm; y = PH - M - 22 * mm
for title, pre in [(f"Current layout v1: {n1} pieces", "pieces_v1_v04"), (f"Proposed layout v2.1: {n2} pieces (legs built like Cuddle Clones and Dragonmoss)", "pieces_v21_v04")]:
    text(LX, y, title, 10.5, "Helvetica-Bold"); y -= 47 * mm
    for i, v in enumerate(["side", "three_q", "under"]): img(W / f"renders/{pre}_{v}.png", LX + i * (LW / 3), y, LW / 3 - 3 * mm, 44 * mm)
    y -= 6 * mm
LEG = [("#e6503c", "P01 head gusset"), ("#468cdc", "P02/03 side head"), ("#f0aa28", "P04 chin gusset"), ("#fa78aa", "P05/06 muzzle side"),
       ("#78be5a", "P07/08 body side, with outside of legs"), ("#fadc5a", "P09/10 chest halves"), ("#aa64c8", "P11/12 underbody, with inside of legs"),
       ("#5a5a5a", "P13-16 paw pads"), ("#3cbebe", "P17/18 tail"), ("#ffffff", "P19-22 ears (flat, not shown)")]
text(LX, y, "v2.1 colours", 8, "Helvetica-Bold", MUTE); y -= 4.5 * mm
for i, (col, name) in enumerate(LEG):
    cx, cy = LX + (i % 3) * (LW / 3), y - (i // 3) * 4.4 * mm
    c.setFillColor(HexColor(col)); c.setStrokeColor(MUTE); c.setLineWidth(0.3); c.rect(cx, cy - 0.6 * mm, 3 * mm, 3 * mm, fill=1, stroke=1)
    text(cx + 4.5 * mm, cy, name, 7.5)
y -= 4 * 4.4 * mm + 2 * mm
c.setStrokeColor(RULE); c.line(LX, y + 2 * mm, LX + LW, y + 2 * mm)
text(LX, y - 2 * mm, "What the finished plush should look like, with the v2.1 seams drawn in (on the plush they sit under the fur)", 8.5, "Helvetica-Bold")
yy = y - 5 * mm; hh = yy - (M + 2 * mm)
ims_ = [crop(W / f"renders/preview_v21/{v}.png") for v in ["side", "three_quarter", "front", "back"]]
ar = [im.size[0] / im.size[1] for im in ims_]; hrow = min(hh, (LW - 6 * mm) / sum(ar)); xx_ = LX
for im, a_ in zip(ims_, ar):
    c.drawImage(ImageReader(im), xx_, M + 2 * mm + (hh - hrow) / 2, a_ * hrow, hrow); xx_ += a_ * hrow + 2 * mm

RX = LX + LW + 10 * mm; RW = PW - M - RX; y = PH - M - 22 * mm
text(RX, y, "What changes from v1, and why", 10.5, "Helvetica-Bold"); y -= 5.5 * mm
r1, r2 = nap1["regions"], nap2["regions"]
CH = [("Legs stay part of the body sides (outside) and the underbody (inside), as the competitors make them.",
       "No ring seam at the elbow or knee, so the legs read as plush legs. Leg fur is trimmed short and brushed down."),
      ("Chest in two halves with a centre seam, pile running down.",
       f"Like the competitors' short-coated dogs. In v1 the chest pile ran towards the tail ({r1['chest & shoulder fronts']['area_off_30deg_pct']:.0f} % off); now {r2['chest & shoulder fronts']['area_off_30deg_pct']:.0f} %."),
      ("Muzzle side pieces: seam from the stop, in front of the eye, to the corner of the mouth.",
       "Both Cuddle Clones and Dragonmoss do this: sharper stop, fuller cheeks, a clean line for the short muzzle fur and the eye."),
      ("Underbody in two halves with one centre seam underneath (v1 had four belly pieces).",
       f"Every piece stays within {max_stretch * 100:.0f} % stretch when flattened (paw pads and tail apart)."),
      ("Darts instead of easing where a seam still needs more than about 3 mm of easing in 40 mm.", "Positions come with the v2.1 pattern."),
      ("Ears: a small pleat at the base so the ear turns forward and frames the face.", ""),
      ("Unchanged: head gusset, chin gusset, centre-back seam, two-piece tail, paw pads.", "")]
for i, (h, b) in enumerate(CH, 1):
    text(RX, y, f"{i}.", 8.5, "Helvetica-Bold", ACC); y = para(RX + 5 * mm, y, h, RW - 5 * mm, 8.5, "Helvetica-Bold")
    if b: y = para(RX + 5 * mm, y, b, RW - 5 * mm, 8, col=MUTE)
    y -= 1.5 * mm
y -= 1 * mm; c.setStrokeColor(RULE); c.line(RX, y + 2 * mm, RX + RW, y + 2 * mm)
text(RX, y - 2.5 * mm, "Fur direction check", 10.5, "Helvetica-Bold"); y -= 6.5 * mm
y = para(RX, y, "Share of each area where the pile, once sewn, points more than 30 degrees away from a Labrador's coat direction. "
         "Legs and underside follow the competitor-style compromise; everything seen from the front and sides is close.", RW, 7.5, col=MUTE) - 1 * mm
cx0, cx1, cx2 = RX, RX + RW * 0.62, RX + RW * 0.82
text(cx0, y, "Area", 8, "Helvetica-Bold", MUTE); text(cx1, y, "v1 now", 8, "Helvetica-Bold", MUTE); text(cx2, y, "v2.1", 8, "Helvetica-Bold", MUTE); y -= 4.6 * mm
for lab_, key in [("Head", "head"), ("Body sides", "body side"), ("Chest and shoulder fronts", "chest & shoulder fronts"), ("Legs", "legs"),
                  ("Belly and inner thighs (underneath)", "underbody"), ("Tail", "tail")]:
    a, b = r1[key]["area_off_30deg_pct"], r2[key]["area_off_30deg_pct"]
    text(cx0, y, lab_, 8.5)
    for xx_, v in [(cx1, a), (cx2, b)]: text(xx_, y, f"{v:.0f} %", 8.5, "Helvetica-Bold" if v > 20 else "Helvetica", ACC if v > 20 else INK)
    c.setStrokeColor(RULE); c.setLineWidth(0.3); c.line(RX, y - 1.6 * mm, RX + RW, y - 1.6 * mm); y -= 4.8 * mm
y -= 1 * mm; text(RX, y, "Cost", 9, "Helvetica-Bold"); y -= 4.4 * mm
y = para(RX, y, f"{n1} pieces become {n2}. Seam length at medium size {seam1:.1f} m to {seam2:.1f} m ({100 * (seam2 / seam1 - 1):+.0f} %). "
         f"Fabric for the body pieces, seam allowance included: {fab1:,.0f} cm2 to {fab2:,.0f} cm2 ({100 * (fab2 / fab1 - 1):+.0f} %).", RW, 8.5) - 2 * mm
c.setStrokeColor(RULE); c.line(RX, y + 2 * mm, RX + RW, y + 2 * mm)
text(RX, y - 2.5 * mm, "Questions on the layout", 10.5, "Helvetica-Bold"); y -= 6.8 * mm
Q1 = ["Legs as part of the body sides and underbody, like the competitors: OK?",
      "Chest in two halves with a centre seam, pile running down: OK?",
      "Head: A) muzzle side pieces as shown, or B) keep the v1 side head with a dart at the stop?",
      "Darts instead of easing: OK? What is the most easing you are happy with in 40 mm of seam?",
      "Paw pads come out about 26 mm across at medium. Separate pads (short pile or suede?), or close the leg end with a dart?",
      "Ears: is a pleat at the base OK? (Lining in the same fur or a short pile? Please say in the comments.)",
      "Do you already make a realistic dog you are happy with? Photos of its pattern pieces would help us."]
questions(Q1, RX, y, RW, 1, {3: ["A", "B"], 5: ["Pads", "Dart"]}, gap=8.6 * mm)
c.showPage()

# =============================================================== page 2: finish (colour, markings, fur, size)
header("Making it look like the dog: colour, markings, fur and size", SUB)
y = PH - M - 22 * mm; CW = 132 * mm
text(M, y, "Colour and markings, read from the customer's photos", 10.5, "Helvetica-Bold"); y -= 4.5 * mm
y = para(M, y, "The coat colours were transferred from the photo-textured 3D model onto the plush shape. The white stock fur is airbrushed "
         "or dyed to match. A yellow Labrador's shading is darker on the ears, back and head and lighter underneath, as the breed standard describes.",
         CW, 8, col=MUTE) - 1 * mm
for i, v in enumerate(["side", "front"]):
    img(W / f"renders/preview_v21_head/{v}.png", M + i * (CW / 2), y - 58 * mm, CW / 2 - 2 * mm, 56 * mm)
y -= 62 * mm
text(M, y, "Paint colours (four tones plus ears), share of the body and where they go", 8.5, "Helvetica-Bold"); y -= 5 * mm
for z in zones["zones"]:
    c.setFillColor(HexColor(z["hex"])); c.setStrokeColor(MUTE); c.rect(M, y - 1.5 * mm, 12 * mm, 6 * mm, fill=1, stroke=1)
    where = ", ".join(w.split(" ", 1)[1].split(" (")[0] for w in z["mostly_on"][:3])
    text(M + 15 * mm, y + 1 * mm, f"{z['hex'].upper()}   {z['area_share'] * 100:.0f} %   mostly on: {where}", 8)
    y -= 7.5 * mm
er = zones["ear_rgb"]; c.setFillColor(HexColor("#%02x%02x%02x" % tuple(er))); c.rect(M, y - 1.5 * mm, 12 * mm, 6 * mm, fill=1, stroke=1)
text(M + 15 * mm, y + 1 * mm, f"#{er[0]:02X}{er[1]:02X}{er[2]:02X}   ears (outer pieces)", 8); y -= 7.5 * mm
pr = zones["pad_rgb"]; c.setFillColor(HexColor("#%02x%02x%02x" % tuple(pr))); c.rect(M, y - 1.5 * mm, 12 * mm, 6 * mm, fill=1, stroke=1)
text(M + 15 * mm, y + 1 * mm, f"#{pr[0]:02X}{pr[1]:02X}{pr[2]:02X}   paw pads (separate pad fabric, to the dog's pads)", 8); y -= 9 * mm
y = para(M, y, "Markings rule: " + spec["markings_rule"].replace("'", "") + ".", CW, 8) - 1 * mm
y = para(M, y, "Baked-in ground shadow on the paws and belly in the 3D model has been lifted, so the feet are coat-coloured.", CW, 7.5, col=MUTE)

MX = M + CW + 8 * mm; MW = 150 * mm; y = PH - M - 22 * mm
text(MX, y, "Colour guide on every piece (cutting orientation, pile down)", 10.5, "Helvetica-Bold"); y -= 3 * mm
img(W / "review/labrador_v21_colour_pieces.png", MX, y - 118 * mm, MW, 116 * mm); y -= 124 * mm
text(MX, y, "Fur trim zones", 10.5, "Helvetica-Bold"); y -= 3 * mm
for i, v in enumerate(["side", "front"]):
    img(W / f"renders/trim_v21/{v}.png", MX + i * (MW / 2), y - 70 * mm, MW / 2 - 2 * mm, 68 * mm)
y -= 74 * mm

RX = MX + MW + 8 * mm; RW = PW - M - RX; y = PH - M - 22 * mm
text(RX, y, "Fur type and trim lengths (medium size)", 10.5, "Helvetica-Bold"); y -= 5 * mm
y = para(RX, y, "Stock fur: " + spec["stock_fur"] + ". Breed standard: coat " + spec["sources"]["coat"].split(": ", 1)[1].replace("'", "") + ".", RW, 8) - 2 * mm
for z in spec["trim_zones"]:
    c.setFillColor(HexColor("#%02x%02x%02x" % tuple(z["colour"]))); c.rect(RX, y - 1.2 * mm, 4 * mm, 4 * mm, fill=1, stroke=0)
    text(RX + 6 * mm, y, z["zone"], 8.5, "Helvetica-Bold"); text(RX + 38 * mm, y, f"{z['length_mm']} mm", 8.5, "Helvetica-Bold")
    y = para(RX + 52 * mm, y, z["note"], RW - 52 * mm, 7.5, col=MUTE) - 1.8 * mm
y = para(RX, y, "Blend each zone into the next over about 10 mm. Lengths are starting values for the first sample.", RW, 7.5, col=MUTE) - 3 * mm
c.setStrokeColor(RULE); c.line(RX, y + 2 * mm, RX + RW, y + 2 * mm)
text(RX, y - 2.5 * mm, "Size", 10.5, "Helvetica-Bold"); y -= 7 * mm
y = para(RX, y, f"The plush is the dog at one scale; fur lengths scale with it. At {spec['plush_size_mm']} mm the thinnest lower-leg and tail sections "
         f"({legs_w:.0f} and {tail_w:.0f} mm at true proportions) are raised to the 25 mm tube minimum, which gives the slightly fuller plush legs "
         f"the competitors have. From about {size_true} mm the dog keeps its true proportions everywhere.", RW, 8) - 3 * mm
c.setStrokeColor(RULE); c.line(RX, y + 2 * mm, RX + RW, y + 2 * mm)
text(RX, y - 2.5 * mm, "Questions on the finish", 10.5, "Helvetica-Bold"); y -= 6.8 * mm
Q2 = ["Can you source a white, straight, dense (not silky) acrylic pile of at least 12 mm? Which pile lengths do you stock?",
      "Can you trim to these zone lengths (4 to 10 mm)? Which clipper guards do you use?",
      "Can you airbrush or dye the white fur to these four tones and the shading on the colour guide?",
      "When a dog has a crisp marking (white bib, blaze, socks), is it OK to move the nearest seam onto the marking edge?"]
y = questions(Q2, RX, y, RW, 8, {}, gap=9.5 * mm)
c.setFillColor(HexColor("#f3f4f5")); lines = simpleSplit("Defaults until you confirm them: seam allowance 5 mm; narrowest turnable tube 25 mm; tightest curve 3 mm; "
    "smallest piece 25 mm; roll width 1500 mm; laser files DXF in mm, cut and notch on separate layers (RDWorks / LightBurn). "
    "Pieces laid at 0 or 180 degrees only.", "Helvetica", 7.5, RW - 6 * mm)
hb = len(lines) * 7.5 * 1.28 + 5 * mm; c.rect(RX, M - 1 * mm, RW, hb, fill=1, stroke=0); yy = M - 1 * mm + hb - 4.5 * mm
c.setFont("Helvetica", 7.5); c.setFillColor(MUTE)
for line in lines: c.drawString(RX + 3 * mm, yy, line); yy -= 7.5 * 1.28
c.showPage(); c.save()
print(f"saved {OUT}: v1 {n1} pieces/{seam1:.2f} m/{fab1:,.0f} cm2; v2.1 {n2} pieces/{seam2:.2f} m/{fab2:,.0f} cm2; max stretch {max_stretch:.3f}; true-proportion size {size_true} mm")
