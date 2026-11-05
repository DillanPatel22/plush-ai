"""Review figure for a sew-and-stuff simulation: target, sewn as cut, and after pattern correction, side by side, in
the side and front views, coloured by how far the stuffed surface moves (red = bulges out, blue = sinks in).

Usage: python3 sim_figure.py out.png title target_dir as_cut_dir [corrected_dir] [--lim 4]
Each *_dir holds the render_views.py output (side.png, front.png) of target.ply / sim.ply / sim_comp.ply.
Acceptance: every panel found; the figure is written.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

out, title = sys.argv[1], sys.argv[2]
dirs = [Path(a) for a in sys.argv[3:] if not a.startswith("--") and not a.replace(".", "").isdigit()]
LIM = float(sys.argv[sys.argv.index("--lim") + 1]) if "--lim" in sys.argv else 4.0
heads = ["Designed shape (target)", "Sewn and stuffed, pieces as cut", "After pattern correction"][:len(dirs)]
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
fb, fr, fs = ImageFont.truetype(FB, 30), ImageFont.truetype(FR, 24), ImageFont.truetype(FR, 21)


def crop(im):
    """trim the uniform background around the model, keep a margin."""
    g = im.convert("L"); bg = g.getpixel((2, 2))
    mask = g.point(lambda v: 255 if abs(v - bg) > 6 else 0)
    bb = mask.getbbox(); m = 30
    return im.crop((max(bb[0] - m, 0), max(bb[1] - m, 0), min(bb[2] + m, im.width), min(bb[3] + m, im.height)))


views = ["side", "front"]
panels = [[crop(Image.open(d / f"{v}.png").convert("RGB")) for v in views] for d in dirs]
cw = 620; rows_h = []
for vi in range(len(views)):
    rows_h.append(max(int(p[vi].height * cw / p[vi].width) if p[vi].width / p[vi].height > 1 else 420 for p in panels))
W = 40 + len(dirs) * (cw + 30); H = 150 + sum(rows_h) + 30 * len(views) + 120
fig = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(fig)
dr.text((40, 30), title, font=fb, fill=(20, 20, 20))
for i, h in enumerate(heads):
    dr.text((40 + i * (cw + 30), 95), h, font=fr, fill=(40, 40, 40))
y = 140
for vi, v in enumerate(views):
    for i, p in enumerate(panels):
        im = p[vi]; s = min(cw / im.width, rows_h[vi] / im.height); im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        x0 = 40 + i * (cw + 30) + (cw - im.width) // 2
        fig.paste(im, (x0, y + (rows_h[vi] - im.height) // 2))
    y += rows_h[vi] + 30
# legend: blue - white - red bar
lx, ly, lw = 40, y + 10, 360
for k in range(lw):
    t = 2 * k / (lw - 1) - 1
    c = (1.0 if t > 0 else 1 + t, 1 - abs(t), 1.0 if t < 0 else 1 - t)
    c = tuple(int(255 * (0.82 * q + 0.1)) for q in c)
    dr.line([(lx + k, ly), (lx + k, ly + 22)], fill=c)
dr.text((lx, ly + 30), f"sinks in {LIM:g} mm", font=fs, fill=(60, 60, 60))
dr.text((lx + lw - 150, ly + 30), f"bulges {LIM:g} mm", font=fs, fill=(60, 60, 60))
dr.text((lx + lw + 40, ly), "Plush size 350 mm nose to tail. Overall size growth removed so shapes compare.", font=fs, fill=(60, 60, 60))
dr.text((lx + lw + 40, ly + 30), "Uncalibrated fabric and stuffing values: directions are reliable, sizes are estimates.", font=fs, fill=(60, 60, 60))
fig.save(out)
print("RESULT: PASS", out, fig.size)
