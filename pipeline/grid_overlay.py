"""Overlay a labelled pixel grid on a photo so landmarks can be read off precisely."""
import sys
from PIL import Image, ImageDraw
src, dst = sys.argv[1], sys.argv[2]
step = int(sys.argv[3]) if len(sys.argv) > 3 else 50
im = Image.open(src).convert("RGB")
d = ImageDraw.Draw(im)
w, h = im.size
for x in range(0, w, step):
    d.line([(x, 0), (x, h)], fill=(255, 0, 0) if x % (step*4) == 0 else (255, 160, 160), width=1)
    if x % (step*2) == 0: d.text((x+2, 2), str(x), fill=(255, 0, 0))
for y in range(0, h, step):
    d.line([(0, y), (w, y)], fill=(0, 0, 255) if y % (step*4) == 0 else (160, 160, 255), width=1)
    if y % (step*2) == 0: d.text((2, y+2), str(y), fill=(0, 0, 255))
im.save(dst, quality=92)
