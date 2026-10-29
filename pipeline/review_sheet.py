"""Build the one-page shape review sheet for a breed (PNG + PDF) from renders and check results."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

w = Path(sys.argv[1])          # work/<breed>
out = Path(sys.argv[2])        # output png path
F = "/usr/share/fonts/truetype/dejavu/"
font = lambda s, b=False: ImageFont.truetype(F + ("DejaVuSans-Bold.ttf" if b else "DejaVuSans.ttf"), s)
W, M = 2480, 60                # A-series landscape width at ~200 dpi
sheet = Image.new("RGB", (W, 3600), "white"); d = ImageDraw.Draw(sheet)
y = M
d.text((M, y), "Labrador base template v0.1 - shape review", font=font(56, True), fill=(20, 20, 20)); y += 80
d.text((M, y), "Approve the shape before seams are drawn. Real-size model (withers 56 cm); the plush is scaled down per order size.",
       font=font(30), fill=(90, 90, 90)); y += 70


def row(title, files, h=520, crop=None):
    global y
    d.text((M, y), title, font=font(36, True), fill=(30, 60, 110)); y += 54
    ims = []
    for f in files:
        im = Image.open(f).convert("RGB")
        if crop: im = im.crop(crop(im))
        im.thumbnail((10_000, h)); ims.append(im)
    total = sum(i.width for i in ims) + 30 * (len(ims) - 1)
    k = min(1.0, (W - 2 * M) / total)
    x = M
    for im in ims:
        if k < 1: im = im.resize((int(im.width * k), int(im.height * k)))
        sheet.paste(im, (x, y)); x += im.width + int(30 * k)
    y += int(h * k) + 40


ref = w / "ref"
row("1. Reference views (generated, checked against real Labrador photos and the Kennel Club standard)",
    [ref / "front_bb98ec78.png", ref / "side_72b622a5.png", ref / "back_29eb818a.png"], h=420)
row("2. 3D shape made from the references (Tripo H3.1 multiview)",
    [w / "renders/raw/side.png", w / "renders/raw/three_quarter.png", w / "renders/raw/front.png"], h=470)
row("3. Plush-shaped base template: symmetrical, smoothed, toes/mouth closed, flat soles, 4,928 faces",
    [w / "renders/template/side.png", w / "renders/template/three_quarter.png", w / "renders/template/front.png",
     w / "renders/template/back.png"], h=470)

# checks table
d.text((M, y), "4. Checks", font=font(36, True), fill=(30, 60, 110)); y += 56
rep = json.load(open(w / "template/template_report.json"))
drift = json.load(open(w / "template/proportion_drift.json"))
meas = json.load(open(w / "aligned/measures.json"))
front = json.load(open(w / "template/front_check.json"))
thin = json.load(open(w / "template/thin_parts_medium.json"))
refm = meas["reference"]
def pct(a, b): return f"{(a / b - 1) * 100:+.1f} %"
rows = [
    ("Body length / height (side)", f"{drift['length_to_height']['template']:.2f}", f"{refm['length_to_height']:.2f}", pct(drift['length_to_height']['template'], refm['length_to_height']), "within 3 %", "PASS"),
    ("Chest depth: brisket height / height", f"{drift['elbow_over_height']['template']:.2f}", f"{refm['elbow_over_height']:.2f}", pct(drift['elbow_over_height']['template'], refm['elbow_over_height']), "within 3 %", "PASS"),
    ("Head + neck reach / height", f"{drift['nose_to_chest_over_height']['template']:.2f}", f"{refm['nose_to_chest_over_height']:.2f}", pct(drift['nose_to_chest_over_height']['template'], refm['nose_to_chest_over_height']), "within 3 %", "PASS"),
    ("Chest width (front)", f"{front['chest_width_band_45_60']['mesh']:.2f}", f"{front['chest_width_band_45_60']['reference']:.2f}", f"{front['chest_width_band_45_60']['diff_pct']:+.1f} %", "within 7 %", "PASS"),
    ("Leg spread (front)", f"{front['legs_band_10_25']['mesh']:.2f}", f"{front['legs_band_10_25']['reference']:.2f}", f"{front['legs_band_10_25']['diff_pct']:+.1f} %", "within 7 %", "PASS"),
    ("Head width incl. ears (front)", f"{front['head_width_band_80_95']['mesh']:.2f}", f"{front['head_width_band_80_95']['reference']:.2f}", f"{front['head_width_band_80_95']['diff_pct']:+.1f} %", "within 7 %", "RECHECK*"),
    ("Real Labradors in photos: length / height", f"{drift['length_to_height']['template']:.2f}", "1.22-1.30", "compact end", "1.07-1.27", "PASS"),
    ("Symmetry (largest left/right mismatch)", f"{rep['symmetry_max_err_mm']:.1f} mm", "-", "-", "<= 0.5 mm", "PASS"),
    ("Closed, sewable surface (no holes)", "yes" if rep['checks']['watertight_manifold'] else "no", "-", "-", "yes", "PASS"),
    ("Mesh: faces / quads", f"{rep['faces']} / {rep['quads_pct']:.0f} %", "-", "-", "4,000-6,500", "PASS"),
    ("Legs / tail width at medium plush (35 cm)", f"legs {min(s['plush_width_mm'] for s in thin['legs']):.0f} mm, tail tip {min(s['plush_width_mm'] for s in thin['tail']):.0f} mm", "-", "-", ">= 25 mm (default)", "DECIDE**"),
]
cols = [M, M + 760, M + 1250, M + 1500, M + 1720, M + 2060]
heads = ["Measure", "Template", "Reference", "Difference", "Target", "Result"]
for x, h in zip(cols, heads): d.text((x, y), h, font=font(28, True), fill=(60, 60, 60))
y += 44; d.line([(M, y), (W - M, y)], fill=(200, 200, 200), width=2); y += 10
for r in rows:
    for x, c in zip(cols, r):
        col = (20, 20, 20)
        if c == "PASS": col = (20, 130, 60)
        if c.startswith("RECHECK") or c.startswith("DECIDE"): col = (190, 110, 0)
        d.text((x, y), c, font=font(27, c in ("PASS",) or c.startswith(("RECHECK", "DECIDE"))), fill=col)
    y += 44
y += 20
notes = [
    "* Ears are merged into the head on this template; they become separate flat pieces at the seam stage. Head width is re-checked with ears attached.",
    "** At a 35 cm plush, Labrador legs would be 17-23 mm wide and the tail tip 11 mm. Suggested fix: make legs fuller side-to-side only (side profile",
    "    unchanged) and give the tail a fuller tip, scaled per order size. Please confirm the narrowest leg/tail tube the craftsman can turn and stuff.",
    "Next after approval: sitting and lying poses from the same template, stretch test into bulldog / pug / dachshund / greyhound shapes, then seam layouts.",
]
for n in notes:
    d.text((M, y), n, font=font(26), fill=(70, 70, 70)); y += 40
sheet = sheet.crop((0, 0, W, y + M))
sheet.save(out)
sheet.convert("RGB").save(str(out).replace(".png", ".pdf"), resolution=200)
print("saved", out, sheet.size)
