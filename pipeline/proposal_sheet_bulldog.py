"""Two-page (A3 landscape) English proposal for the workshop: the English bulldog on the standing layout v2.1
(the Labrador's pieces, reshaped) with the pronounced-muzzle module, and the finish plan (colour and markings from the
photos, wrinkle map, fur trim zones, size). Every number is read from the check outputs.

Usage: python3 proposal_sheet_bulldog.py work_dir labrador_work_dir out.pdf
"""
import json, sys, datetime
from pathlib import Path
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader, simpleSplit

W, LW_, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
PW, PH = landscape(A3); M = 12 * mm
INK, MUTE, ACC, RULE = HexColor("#1f2328"), HexColor("#5b636b"), HexColor("#b4532a"), HexColor("#d7dadd")
lay = json.load(open(W / "pieces_v21/pieces.json"))["pieces"]
napB = json.load(open(W / "review/nap_v21_visible.json"))["regions"]; napL = json.load(open(LW_ / "review/nap_v21_visible.json"))["regions"]
flatB = json.load(open(W / "flat_v21/flatten_report.json"))["pieces"]; darts = json.load(open(W / "flat_v21/darts.json"))
zones = json.load(open(W / "review/bulldog_colour_zones.json")); spec = json.load(open(W / "finish_spec.json"))
thin = json.load(open(W / "template/thin_after.json")); wr = json.load(open(W / "review/bulldog_wrinkles.json"))
mk = json.load(open(W / "review/bulldog_markings.json")); rs = json.load(open(W / "template/template_reshape.json"))
ms = json.load(open(W / "template/measures.json")); tail = json.load(open(W / "plush/tail_stub.json"))
DATE = datetime.date(2026, 11, 3)


def perim(f):
    d = np.load(f); P = d["uv"][d["bnd"]]; return float(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum())
def cut_area(f, n):
    d = np.load(f); return Polygon(d["uv"][d["bnd"]]).buffer(0).buffer(5.0, join_style=2, mitre_limit=2.0).area * n
def totals(L, fdir):
    mult = {p: (2 if any(i.get("mirror_of") == p for i in L.values()) else 1) for p, i in L.items() if not i.get("mirror_of") and i["faces"]}
    return (sum(mult.values()) + 4, sum(perim(fdir / f"{p}.npz") * n for p, n in mult.items()) / 2000,
            sum(cut_area(fdir / f"{p}.npz", n) for p, n in mult.items()) / 100, mult)
layL = json.load(open(LW_ / "pieces_v21_template_v04/pieces.json"))["pieces"]
nL, seamL, fabL, _ = totals(layL, LW_ / "flat_v21_template_v04"); nB, seamB, fabB, multB = totals(lay, W / "flat_v21")
after = {p: (darts[p]["after"] if p in darts else flatB[p]) for p in flatB}
ndarts = sum(len(darts[p]["darts"]) * multB[p] for p in darts if darts[p]["darts"])
body_max = max(after[p]["area_dev_p95"] for p in ("P07", "P09", "P11"))
legs_w = min(s["plush_width_mm"] for s in thin["legs"]); tail_w = min(s["plush_width_mm"] for s in thin["tail"])
k = 350 / (ms["nose_to_chest_front_mm"] + ms["body_length_mm"])

c = canvas.Canvas(OUT, pagesize=(PW, PH)); c.setTitle("English bulldog seam layout and finish plan")


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
def row(paths, x, y, w, h, gap=2 * mm):
    ims_ = [crop(p) for p in paths]; ar = [im.size[0] / im.size[1] for im in ims_]
    hrow = min(h, (w - gap * (len(ims_) - 1)) / sum(ar)); xx_ = x
    for im, a_ in zip(ims_, ar):
        c.drawImage(ImageReader(im), xx_, y + (h - hrow) / 2, a_ * hrow, hrow); xx_ += a_ * hrow + gap
def header(title, sub):
    text(M, PH - M - 6 * mm, title, 17, "Helvetica-Bold")
    tw = c.stringWidth(title, "Helvetica-Bold", 17)
    c.setFillColor(ACC); c.roundRect(M + tw + 6 * mm, PH - M - 7.2 * mm, 52 * mm, 7 * mm, 1.5 * mm, fill=1, stroke=0)
    text(M + tw + 9 * mm, PH - M - 5 * mm, "PROPOSAL FOR APPROVAL", 10, "Helvetica-Bold", white)
    text(M, PH - M - 12 * mm, sub, 9, col=MUTE)
    c.setStrokeColor(RULE); c.setLineWidth(0.6); c.line(M, PH - M - 15 * mm, PW - M, PH - M - 15 * mm)
def rule(x, y, w):
    c.setStrokeColor(RULE); c.setLineWidth(0.6); c.line(x, y, x + w, y)
def questions(Q, x, y, width, start=1, choices=None, gap=10 * mm):
    for i, q in enumerate(Q, start):
        text(x, y, f"{i}.", 8.5, "Helvetica-Bold", ACC); y = para(x + 5 * mm, y, q, width - 5 * mm, 8.5); yb = y - 0.5 * mm
        for j, lab_ in enumerate((choices or {}).get(i, ["Yes", "No"])):
            bx = x + 5 * mm + j * 15 * mm
            c.setStrokeColor(INK); c.setLineWidth(0.5); c.rect(bx, yb - 0.4 * mm, 3 * mm, 3 * mm); text(bx + 4.2 * mm, yb, lab_, 8)
        text(x + 36 * mm, yb, "Comments:", 7, col=MUTE); c.setStrokeColor(RULE); c.setLineWidth(0.4); c.line(x + 49 * mm, yb - 0.6 * mm, x + width, yb - 0.6 * mm)
        y -= gap
    return y
def bullets(items, x, y, width, size=8.2):
    for i, (h, b) in enumerate(items, 1):
        text(x, y, f"{i}.", size, "Helvetica-Bold", ACC); y = para(x + 5 * mm, y, h, width - 5 * mm, size, "Helvetica-Bold")
        if b: y = para(x + 5 * mm, y, b, width - 5 * mm, size - 0.4, col=MUTE)
        y -= 1.3 * mm
    return y

SUB = (f"Medium size: {spec['plush_size_mm']} mm nose to tail base. The dog base shape v0.4 reshaped onto a brindle-and-white English bulldog. "
       f"{DATE:%d %b %Y}. Not for cutting: please answer the questions and send them back.")
# =============================================================== page 1: layout
header("English bulldog, standing: seam layout", SUB)
LX, LW = M, 200 * mm; y = PH - M - 22 * mm
text(LX, y, f"The Labrador's layout v2.1 reshaped for the bulldog (same piece numbers), plus a pair of jowls: {nB} pieces", 10.5, "Helvetica-Bold"); y -= 47 * mm
row([W / f"renders/pieces_v21_{v}.png" for v in ("side", "three_q", "under")], LX, y, LW, 44 * mm, 3 * mm); y -= 6 * mm
text(LX, y, "Bulldog face: jowls, muzzle tops and the undershot chin", 10.5, "Helvetica-Bold"); y -= 47 * mm
row([W / f"renders/pieces_v21_head/{v}.png" for v in ("front", "three_quarter", "side")], LX, y, LW, 44 * mm, 3 * mm); y -= 5 * mm
LEG = [("#e6503c", "P01 head gusset"), ("#468cdc", "P02/03 side head (1 dart each)"), ("#f0aa28", "P04 chin / lower lip"),
       ("#fa78aa", "P05/06 muzzle top"), ("#be328c", "P23/24 jowls (3 darts each)"), ("#78be5a", "P07/08 body side, with outside of legs"),
       ("#fadc5a", "P09/10 chest halves"), ("#aa64c8", "P11/12 underbody, with inside of legs"), ("#5a5a5a", "P13-16 paw pads"),
       ("#3cbebe", "P17/18 tail stub"), ("#ffffff", "P19-22 rose ears (flat, not shown)")]
for i, (col, name) in enumerate(LEG):
    cx, cy = LX + (i % 3) * (LW / 3), y - (i // 3) * 4.4 * mm
    c.setFillColor(HexColor(col)); c.setStrokeColor(MUTE); c.setLineWidth(0.3); c.rect(cx, cy - 0.6 * mm, 3 * mm, 3 * mm, fill=1, stroke=1)
    text(cx + 4.5 * mm, cy, name, 7.5)
y -= 4 * 4.4 * mm + 1 * mm
rule(LX, y + 2 * mm, LW)
text(LX, y - 2 * mm, "What the finished plush should look like: seams in brown, needle-sculpted wrinkles in red (under the fur on the plush)", 8.5, "Helvetica-Bold")
row([W / f"renders/preview_v21/{v}.png" for v in ("side", "three_quarter", "front", "back")], LX, M + 2 * mm, LW, y - 5 * mm - (M + 2 * mm))

RX = LX + LW + 10 * mm; RW = PW - M - RX; y = PH - M - 22 * mm
text(RX, y, "What the competitors do on bulldogs", 10.5, "Helvetica-Bold"); y -= 5 * mm
y = bullets([("Jowls are their own stuffed lobes; the lower lip / chin sits below and between them.",
              "Cuddle Clones customer photos, a pattern designer's bulldog (jowls, lips, chin pieces), Dragonmoss ('undershot lower lip')."),
             ("The nose rope is either sewn in as a folded strip (pattern designer, Dragonmoss) or sculpted and shaded (Cuddle Clones).", ""),
             ("Legs belong to the body; chest has a centre seam; brindle and white are airbrushed on one fur.", ""),
             ("Their customers' main complaint on bulldogs: wrinkles not deep or dark enough.", "Two Cuddle Clones reviews ask for deeper, darker head wrinkles.")],
            RX, y, RW) - 1 * mm
rule(RX, y + 2 * mm, RW); text(RX, y - 2.5 * mm, "What is new for the bulldog, and why", 10.5, "Helvetica-Bold"); y -= 7 * mm
d02 = darts.get("P02", {"darts": []}); d23 = darts.get("P23", {"darts": []})
y = bullets([("Same pieces as the Labrador (frozen layout), reshaped onto the bulldog: the pattern and the job card read the same for every dog.",
              f"The base shape was fitted to the bulldog with every vertex kept (surface within {rs['to_breed_p95_mm']:.1f} mm, 95 % of points)."),
             ("Face module for flat-faced breeds: the muzzle side splits along the jowl crease into a small muzzle top and a jowl.",
              f"The jowls meet under the nose (philtrum seam; the nose covers the meeting point). Each jowl has {len(d23['darts'])} small darts along its lower edge, where it curls under: "
              f"stretch {d23['before']['area_dev_p95'] * 100:.0f} % without, {d23['after']['area_dev_p95'] * 100:.0f} % with."),
             ("The chin gusset becomes the chin / lower lip and shows between the jowls at the front, as on an undershot bulldog.", ""),
             ("Side head: one dart each on top of the skull, from the head gusset seam out over the brow (it can hide in a forehead wrinkle).",
              f"Stretch {d02['before']['area_dev_p95'] * 100:.0f} % without, {d02['after']['area_dev_p95'] * 100:.0f} % with."),
             ("Tail: a short thick stub set on the rump, carried back and down (breed standard), in two halves.",
              f"{tail_w:.0f} mm wide at medium. The rounded tip needs a little gathering."),
             ("Rose ears: small flat ear pieces, folded back and tacked by hand after sewing.", "")], RX, y, RW) - 1 * mm
rule(RX, y + 2 * mm, RW); text(RX, y - 2.5 * mm, "Fur direction check", 10.5, "Helvetica-Bold"); y -= 6.5 * mm
y = para(RX, y, "Share of each area where the pile, once sewn, points more than 30 degrees away from the dog's coat direction.", RW, 7.5, col=MUTE) - 1 * mm
cx0, cx1, cx2 = RX, RX + RW * 0.62, RX + RW * 0.82
text(cx0, y, "Area", 8, "Helvetica-Bold", MUTE); text(cx1, y, "Labrador", 8, "Helvetica-Bold", MUTE); text(cx2, y, "Bulldog", 8, "Helvetica-Bold", MUTE); y -= 4.6 * mm
for lab_, key in [("Head", "head"), ("Body sides", "body side"), ("Chest and shoulder fronts", "chest & shoulder fronts"), ("Legs", "legs"),
                  ("Belly and inner thighs (underneath)", "underbody"), ("Tail", "tail")]:
    a, b = napL[key]["area_off_30deg_pct"], napB[key]["area_off_30deg_pct"]
    text(cx0, y, lab_ + (" (short stub)" if key == "tail" else ""), 8.5)
    for xx_, v in [(cx1, a), (cx2, b)]: text(xx_, y, f"{v:.0f} %", 8.5, "Helvetica-Bold" if v > 20 else "Helvetica", ACC if v > 20 else INK)
    c.setStrokeColor(RULE); c.setLineWidth(0.3); c.line(RX, y - 1.6 * mm, RX + RW, y - 1.6 * mm); y -= 4.6 * mm
y -= 1 * mm; text(RX, y, "Cost and stretch", 9, "Helvetica-Bold"); y -= 4.4 * mm
y = para(RX, y, f"Labrador {nL} pieces, bulldog {nB} ({ndarts} darts). Seam length at medium {seamL:.1f} m vs {seamB:.1f} m; fabric for the body "
         f"pieces, seam allowance included, {fabL:,.0f} cm2 vs {fabB:,.0f} cm2 (a bulldog is shorter but much wider). "
         f"The barrel chest and heavy thighs stretch the body side, chest and underbody up to {body_max * 100:.0f} % at the flank and inner thigh "
         f"(Labrador 9 %).", RW, 8.2) - 2 * mm
rule(RX, y + 2 * mm, RW); text(RX, y - 2.5 * mm, "Questions on the layout", 10.5, "Helvetica-Bold"); y -= 6.8 * mm
Q1 = ["Same layout as the Labrador, reshaped for the bulldog (same piece numbers): OK?",
      "Jowls as their own pieces with small darts, meeting under the nose: OK?",
      "Nose rope: A) needle-sculpted and shaded (as shown), or B) a folded strip sewn into the seam above the jowls?",
      f"Up to {body_max * 100:.0f} % stretch at the flank and inner thigh: OK with this backing, or would you add a dart there?",
      "Tail stub in two halves with the tip gathered: OK? (A curled screw tail is possible for dogs that have one.)",
      "Rose ears folded back and tacked by hand: OK? Lining in short pile or the same fur?"]
questions(Q1, RX, y, RW, 1, {3: ["A", "B"]}, gap=8.4 * mm)
c.showPage()

# =============================================================== page 2: finish
header("Making it look like the dog: colour, markings, wrinkles, fur and size", SUB)
y = PH - M - 22 * mm; CW = 132 * mm
text(M, y, "Colour and markings, read from the customer's photos", 10.5, "Helvetica-Bold"); y -= 4.5 * mm
y = para(M, y, "Colours transferred from the photo-textured 3D model onto the plush shape. The white stock fur stays white where the dog is white; "
         "the brindle is airbrushed in stripes following the colour guide.", CW, 8, col=MUTE) - 1 * mm
row([W / f"renders/preview_v21_head/{v}.png" for v in ("side", "front")], M, y - 58 * mm, CW, 56 * mm); y -= 62 * mm
text(M, y, "Paint tones (share of the body and where they go)", 8.5, "Helvetica-Bold"); y -= 5 * mm
for i, z in enumerate(zones["zones"]):
    c.setFillColor(HexColor(z["hex"])); c.setStrokeColor(MUTE); c.rect(M, y - 1.5 * mm, 12 * mm, 6 * mm, fill=1, stroke=1)
    where = ", ".join(w.split(" ", 1)[1].split(" (")[0] for w in z["mostly_on"][:3])
    lbl = "white: leave the fur unpainted" if i == 0 else f"mostly on: {where}"
    text(M + 15 * mm, y + 1 * mm, f"{z['hex'].upper()}   {z['area_share'] * 100:.0f} %   {lbl}", 8); y -= 7.2 * mm
er = zones["ear_rgb"]; c.setFillColor(HexColor("#%02x%02x%02x" % tuple(er))); c.rect(M, y - 1.5 * mm, 12 * mm, 6 * mm, fill=1, stroke=1)
text(M + 15 * mm, y + 1 * mm, f"#{er[0]:02X}{er[1]:02X}{er[2]:02X}   ears (outer pieces)", 8); y -= 7.2 * mm
pr = zones["pad_rgb"]; c.setFillColor(HexColor("#%02x%02x%02x" % tuple(pr))); c.rect(M, y - 1.5 * mm, 12 * mm, 6 * mm, fill=1, stroke=1)
text(M + 15 * mm, y + 1 * mm, f"#{pr[0]:02X}{pr[1]:02X}{pr[2]:02X}   paw pads (separate pad fabric)", 8); y -= 9 * mm
text(M, y, "Markings rule, on this dog", 9, "Helvetica-Bold"); y -= 4.5 * mm
gb = [r for r in mk["head_gusset_vs_blaze"] if r["y_from_nose_mm"] <= 100]; cb = [r for r in mk["chest_vs_bib"] if 170 <= r["z_mm"] <= 250]
y = para(M, y, f"Head gusset: over the forehead its edges already sit on the white blaze (within {max(abs(r['offset_plush_mm']) for r in gb):.0f} mm). "
         f"Behind the eyes the blaze narrows and stops, so the gusset there is airbrushed. Chest: the brindle starts "
         f"{min(-r['offset_plush_mm'] for r in cb):.0f}-{max(-r['offset_plush_mm'] for r in cb):.0f} mm in front of the chest seam. Moving that seam forward onto the "
         "bib edge would hide the colour change in the seam (or the brindle is airbrushed on the outer strip of the chest pieces).", CW, 8) - 4 * mm
c.setFillColor(HexColor("#f3f4f5")); notes = simpleSplit("To check on the first sample: eye and nose positions (read from the photos), how full the jowls "
    "stand, the rose-ear fold, and the gathered tail tip. The automatic face check against the photos was not used for this dog: its white blaze, "
    "muzzle and chest blend into the white studio background, so the photo outline is unreliable there; the 3D model built from all four views is used instead.",
    "Helvetica", 7.5, CW - 6 * mm)
hb = len(notes) * 7.5 * 1.28 + 5 * mm; c.rect(M, y - hb, CW, hb, fill=1, stroke=0); yy = y - 4.5 * mm
c.setFont("Helvetica", 7.5); c.setFillColor(MUTE)
for line in notes: c.drawString(M + 3 * mm, yy, line); yy -= 7.5 * 1.28

MX = M + CW + 8 * mm; MW = 150 * mm; y = PH - M - 22 * mm
text(MX, y, "Colour guide on every piece (cutting orientation, pile down)", 10.5, "Helvetica-Bold"); y -= 3 * mm
img(W / "review/bulldog_colour_pieces.png", MX, y - 104 * mm, MW, 102 * mm); y -= 108 * mm
text(MX, y, "Wrinkle map (needle-sculpted; the shape itself stays smooth)", 10.5, "Helvetica-Bold"); y -= 3 * mm
row([W / f"renders/wrinkles/{v}.png" for v in ("front", "side")], MX, y - 52 * mm, MW, 50 * mm); y -= 56 * mm
groups = {}
for w in wr["wrinkles"]: groups.setdefault(w["name"], []).append(w)
text(MX, y, "Wrinkle", 8, "Helvetica-Bold", MUTE); text(MX + 50 * mm, y, "Length", 8, "Helvetica-Bold", MUTE)
text(MX + 72 * mm, y, "Depth", 8, "Helvetica-Bold", MUTE); text(MX + 92 * mm, y, "Extra fabric across it", 8, "Helvetica-Bold", MUTE); y -= 4.4 * mm
for name, ws in groups.items():
    L_ = np.mean([w["length_plush_mm"] for w in ws]); w0 = ws[0]
    text(MX, y, f"{name} (x{len(ws)})" if len(ws) > 1 else name, 8); text(MX + 50 * mm, y, f"{L_:.0f} mm", 8)
    text(MX + 72 * mm, y, f"{w0['sculpt_depth_plush_mm']} mm", 8); text(MX + 92 * mm, y, f"{w0['fabric_allowance_plush_mm']:.0f} mm, {w0['allowance_where']}", 7.5)
    c.setStrokeColor(RULE); c.setLineWidth(0.3); c.line(MX, y - 1.5 * mm, MX + MW, y - 1.5 * mm); y -= 4.4 * mm
y = para(MX, y - 1 * mm, "Shade each crease darker with the airbrush as well (competitor reviews ask for deeper, darker wrinkles). Depths and extra fabric are starting values.", MW, 7.5, col=MUTE)

RX = MX + MW + 8 * mm; RW = PW - M - RX; y = PH - M - 22 * mm
text(RX, y, "Fur type and trim lengths (medium size)", 10.5, "Helvetica-Bold"); y -= 5 * mm
y = para(RX, y, "Stock fur: " + spec["stock_fur"] + ". Breed standard: coat short, close and smooth, fine in texture.", RW, 8) - 2 * mm
for z in spec["trim_zones"]:
    c.setFillColor(HexColor("#%02x%02x%02x" % tuple(z["colour"]))); c.rect(RX, y - 1.2 * mm, 4 * mm, 4 * mm, fill=1, stroke=0)
    text(RX + 6 * mm, y, z["zone"], 8.5, "Helvetica-Bold"); text(RX + 38 * mm, y, f"{z['length_mm']} mm", 8.5, "Helvetica-Bold")
    y = para(RX + 52 * mm, y, z["note"], RW - 52 * mm, 7.5, col=MUTE) - 1.8 * mm
row([W / f"renders/trim_v21/{v}.png" for v in ("side", "front")], RX, y - 40 * mm, RW, 38 * mm); y -= 44 * mm
rule(RX, y + 2 * mm, RW); text(RX, y - 2.5 * mm, "Size", 10.5, "Helvetica-Bold"); y -= 7 * mm
y = para(RX, y, f"At {spec['plush_size_mm']} mm every leg is already wider than the 25 mm tube minimum (narrowest {legs_w:.0f} mm), so the bulldog keeps its "
         f"true proportions at every size. Only the tail stub is set at the minimum ({tail_w:.0f} mm wide). Fur lengths scale with the plush size.", RW, 8) - 3 * mm
rule(RX, y + 2 * mm, RW); text(RX, y - 2.5 * mm, "Questions on the finish", 10.5, "Helvetica-Bold"); y -= 6.8 * mm
Q2 = ["Can you source a white, short, dense, smooth acrylic pile of at least 8 mm? Which pile lengths do you stock?",
      "Can you trim to these zone lengths (3 to 6 mm)?",
      "Can you airbrush brindle stripes from the colour guide, and leave the white areas unpainted?",
      "Needle-sculpting: are these wrinkle depths and the extra fabric right? Do you add the extra fabric to the pattern yourself?",
      "On this dog, move the chest seam forward onto the edge of the white bib: OK?"]
y = questions(Q2, RX, y, RW, 7, {}, gap=9.2 * mm)
c.setFillColor(HexColor("#f3f4f5")); lines = simpleSplit("Defaults until you confirm them: seam allowance 5 mm; narrowest turnable tube 25 mm; tightest curve 3 mm; "
    "smallest piece 25 mm (the front paw pads come out 24 mm: pad fabric); roll width 1500 mm; laser files DXF in mm, cut and notch on separate layers "
    "(RDWorks / LightBurn). Pieces laid at 0 or 180 degrees only.", "Helvetica", 7.5, RW - 6 * mm)
hb = len(lines) * 7.5 * 1.28 + 5 * mm; c.rect(RX, M - 1 * mm, RW, hb, fill=1, stroke=0); yy = M - 1 * mm + hb - 4.5 * mm
c.setFont("Helvetica", 7.5); c.setFillColor(MUTE)
for line in lines: c.drawString(RX + 3 * mm, yy, line); yy -= 7.5 * 1.28
c.showPage(); c.save()
print(f"saved {OUT}: Labrador {nL} pieces/{seamL:.2f} m/{fabL:,.0f} cm2; bulldog {nB} pieces/{seamB:.2f} m/{fabB:,.0f} cm2, {ndarts} darts; "
      f"body stretch max {body_max:.3f}; legs min {legs_w} mm, tail {tail_w} mm")
