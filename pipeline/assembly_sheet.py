"""Bilingual (Chinese first) assembly sheet for a pattern: piece map, piece list, sewing order with seam letters.

Usage: python3 assembly_sheet.py work_dir pattern_dir renders_prefix size_label face_points.json out.pdf
"""
import json, sys
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

W, PD, RP, SIZE, FP, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
pdfmetrics.registerFont(TTFont("ZH", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", subfontIndex=0))   # embedded, works on any viewer
chk = json.load(open(PD / "pattern_checks.json")); seams = chk["seams"]
fp = json.load(open(FP))
def L(*pairs):
    out = []
    for a, b in pairs:
        out += [k for k, v in seams.items() if sorted(v) == sorted([a, b])]
    return ", ".join(sorted(set(out), key=lambda s: (len(s), s))) or "-"

PIECES = [("P01", "头顶条", "Head gusset"), ("P02", "头侧片 右", "Side head R"), ("P03", "头侧片 左", "Side head L"),
          ("P04", "下巴条", "Chin gusset"), ("P05", "身侧片 右", "Body side R"), ("P06", "身侧片 左", "Body side L"),
          ("P07", "前腹片 右", "Front underbody R"), ("P14", "前腹片 左", "Front underbody L"),
          ("P15", "后腹片 右", "Rear underbody R"), ("P16", "后腹片 左", "Rear underbody L"),
          ("P08", "前脚底 右", "Front sole R"), ("P09", "前脚底 左", "Front sole L"), ("P10", "后脚底 右", "Rear sole R"),
          ("P11", "后脚底 左", "Rear sole L"), ("P12", "尾巴 右", "Tail R"), ("P13", "尾巴 左", "Tail L"),
          ("P17", "耳朵外片 右", "Ear outer R"), ("P18", "耳朵外片 左", "Ear outer L"), ("P19", "耳朵里片 右", "Ear lining R"),
          ("P20", "耳朵里片 左", "Ear lining L")]
STEPS = [
    ("耳朵：外片与里片正面相对，沿边缝合，上边 EAR1–EAR2 留口，翻正。", "Ears: outer + lining right sides together, sew round, leave the top edge EAR1–EAR2 open, turn."),
    (f"头部：头顶条缝左右头侧片（{L(('P01','P02'),('P01','P03'))}）；下巴条缝头侧片（{L(('P02','P04'),('P03','P04'))}）。",
     f"Head: gusset to side heads ({L(('P01','P02'),('P01','P03'))}); chin gusset to side heads ({L(('P02','P04'),('P03','P04'))})."),
    (f"腹部：前腹片左右缝合（{L(('P07','P14'))}），后腹片左右缝合（{L(('P15','P16'))}），再缝前后腹片（{L(('P07','P15'),('P14','P16'))}）。",
     f"Underbody: front halves ({L(('P07','P14'))}), rear halves ({L(('P15','P16'))}), then front to rear ({L(('P07','P15'),('P14','P16'))})."),
    (f"尾巴：左右尾片缝合（{L(('P12','P13'))}），尾根留口。", f"Tail: halves together ({L(('P12','P13'))}), leave the root open."),
    (f"身体：左右身侧片沿背中线缝合（{L(('P05','P06'))}）。", f"Body: body sides together along the back and rear ({L(('P05','P06'))})."),
    (f"腹部缝到身侧片（{L(('P05','P07'),('P06','P14'),('P05','P15'),('P06','P16'))}），腹部中间留约 6 厘米开口。",
     f"Underbody to body sides ({L(('P05','P07'),('P06','P14'),('P05','P15'),('P06','P16'))}); leave a 6 cm opening mid-belly."),
    (f"缝脚底（{L(('P05','P08'),('P07','P08'),('P06','P09'),('P09','P14'),('P05','P10'),('P10','P15'),('P06','P11'),('P11','P16'))}）。",
     f"Sew in the soles ({L(('P05','P08'),('P07','P08'),('P06','P09'),('P09','P14'),('P05','P10'),('P10','P15'),('P06','P11'),('P11','P16'))})."),
    (f"尾巴缝到身体（{L(('P05','P12'),('P06','P13'))}）。", f"Tail to body ({L(('P05','P12'),('P06','P13'))})."),
    (f"头与身体缝合（颈圈 {L(('P01','P05'),('P01','P06'),('P02','P05'),('P03','P06'))}）。",
     f"Head to body at the neck ring ({L(('P01','P05'),('P01','P06'),('P02','P05'),('P03','P06'))})."),
    (f"翻面前装眼睛和鼻子：按蓝色标记，眼睛约 {fp['eye_diam_mm_plush']:.0f} 毫米，鼻子宽约 {fp['nose_width_mm_plush']:.0f} 毫米，装在头顶条前端。",
     f"Before turning: eyes at the blue marks (about {fp['eye_diam_mm_plush']:.0f} mm), nose about {fp['nose_width_mm_plush']:.0f} mm wide at the front of the head gusset."),
    ("翻面，填充（记录填充克数），藏针缝合开口。", "Turn, stuff (record the fill weight in grams), ladder-stitch the opening."),
    ("耳朵按头侧蓝色 EAR 线缝上。", "Sew the ears on along the blue EAR lines."),
    ("拍六个角度照片，测量体长、身高、头宽、口吻长度，标出起皱或拉紧的位置。", "Photograph 6 angles; measure length, height, head width, muzzle length; mark any puckers or strain."),
]
NOTES = [("缝份 5 毫米已含在裁剪线内。图纸为布背面视图（激光从布背面切割）。", "5 mm seam allowance is included. Drawings show the BACK of the fabric (laser cuts from the back)."),
         ("相同字母数字的刀口对齐；刀口之间较长一边吃进（最多约 5 毫米）。", "Match notches with the same letter+number; ease the longer side in between notches (up to about 5 mm)."),
         ("此版为摇粒绒试样：未做填充补偿、未加省道；尾根和臀部可能起皱，请在照片中标出。", "v1 fleece mock-up: no stuffing compensation or darts yet; the tail root and hips may pucker, please mark them.")]

c = canvas.Canvas(OUT, pagesize=landscape(A3)); PW, PH = landscape(A3)
c.setFont("ZH", 18); c.drawString(15 * mm, PH - 18 * mm, f"拉布拉多 站姿 {SIZE} 码 纸样 v1（摇粒绒试样） / Labrador standing, size {SIZE}, pattern v1 (fleece mock-up)")
# piece map renders
imgs = [f"{RP}_side.png", f"{RP}_three_q.png", f"{RP}_under.png", f"{RP}_front.png"]
x0, y0, wimg = 15 * mm, PH - 30 * mm, 95 * mm
for i, f in enumerate(imgs):
    ir = ImageReader(f); iw, ih = ir.getSize(); h = wimg * ih / iw
    c.drawImage(ir, x0 + (i % 2) * (wimg + 5 * mm), y0 - (i // 2 + 1) * (h + 4 * mm), wimg, h)
# legend
leg = [("#e6503c", "头顶条 Head gusset P01"), ("#468cdc", "头侧片 Side head P02/P03"), ("#f0aa28", "下巴条 Chin P04"),
       ("#78be5a", "身侧片 Body side P05/P06"), ("#aa64c8", "腹片 Underbody P07/P14/P15/P16"), ("#5a5a5a", "脚底 Soles P08-P11"),
       ("#3cbebe", "尾巴 Tail P12/P13")]
ly = y0 - 2 * (wimg * 650 / 900 + 4 * mm) - 6 * mm
c.setFont("ZH", 9)
for i, (col, t) in enumerate(leg):
    c.setFillColor(HexColor(col)); c.rect(x0 + (i % 2) * 100 * mm, ly - (i // 2) * 6 * mm, 4 * mm, 4 * mm, fill=1, stroke=0)
    c.setFillColor(black); c.drawString(x0 + (i % 2) * 100 * mm + 6 * mm, ly - (i // 2) * 6 * mm + 1 * mm, t)
# piece list
tx = 220 * mm; ty = PH - 32 * mm
c.setFont("ZH", 11); c.drawString(tx, ty, "裁片 Pieces（每片裁 1 / cut 1 each）"); ty -= 6 * mm
c.setFont("ZH", 8.5)
for i, (pid, zh, en) in enumerate(PIECES):
    col = i // 10; row = i % 10
    c.drawString(tx + col * 95 * mm, ty - row * 4.6 * mm, f"{pid}  {zh}  {en}")
ty -= 10 * 4.6 * mm + 6 * mm
c.setFont("ZH", 11); c.drawString(tx, ty, "缝制顺序 Sewing order（字母 = 刀口组 / letters = notch groups）"); ty -= 6 * mm
for i, (zh, en) in enumerate(STEPS, 1):
    c.setFont("ZH", 9); c.drawString(tx, ty, f"{i}. {zh}"); ty -= 4.3 * mm
    c.setFont("Helvetica", 7.5); c.setFillColor(HexColor("#444444")); c.drawString(tx + 4 * mm, ty, en); c.setFillColor(black); ty -= 5.2 * mm
ty -= 2 * mm
c.setFont("ZH", 11); c.drawString(tx, ty, "注意 Notes"); ty -= 5.5 * mm
for zh, en in NOTES:
    c.setFont("ZH", 9); c.drawString(tx, ty, "• " + zh); ty -= 4.3 * mm
    c.setFont("Helvetica", 7.5); c.setFillColor(HexColor("#444444")); c.drawString(tx + 4 * mm, ty, en); c.setFillColor(black); ty -= 5.2 * mm
c.setFont("Helvetica", 7); c.drawString(15 * mm, 8 * mm, f"Seam letters: " + "  ".join(f"{k}={'/'.join(v)}" for k, v in seams.items()))
c.showPage(); c.save(); print("saved", OUT)
