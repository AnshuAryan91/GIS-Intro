from PIL import Image, ImageDraw, ImageFont
import os

out_dir = 'sample'
os.makedirs(out_dir, exist_ok=True)

width, height = 1800, 1200
img = Image.new('RGB', (width, height), '#eaf2ea')
draw = ImageDraw.Draw(img)

# Draw road network
draw.rectangle([0, 0, width, height], fill='#edf3ef')
road_color = '#9aa3a1'
for x in range(0, width, 280):
    draw.rectangle([x, 0, x + 22, height], fill=road_color)
for y in range(0, height, 220):
    draw.rectangle([0, y, width, y + 18], fill=road_color)

# Parcel layout: regular rectangular lots
dark = '#3a3a3a'
parcel_colors = ['#c8e6c9', '#d7f0ff', '#f8d7b5', '#f3e4a9', '#d9d7f4', '#f7c7d9']

parcels = [
    (60, 60, 420, 260),
    (470, 60, 820, 260),
    (880, 60, 1230, 260),
    (1290, 60, 1640, 260),
    (60, 310, 420, 510),
    (470, 310, 820, 510),
    (880, 310, 1230, 510),
    (1290, 310, 1640, 510),
    (60, 560, 420, 760),
    (470, 560, 820, 760),
    (880, 560, 1230, 760),
    (1290, 560, 1640, 760),
    (60, 810, 420, 1010),
    (470, 810, 820, 1010),
    (880, 810, 1230, 1010),
    (1290, 810, 1640, 1010),
]

for i, rect in enumerate(parcels):
    x1, y1, x2, y2 = rect
    color = parcel_colors[i % len(parcel_colors)]
    draw.rectangle([x1, y1, x2, y2], fill=color, outline=dark, width=4)

# Extra building blocks inside some parcels
after = [
    (90, 90, 220, 180),
    (520, 95, 700, 200),
    (915, 110, 1075, 200),
    (1360, 110, 1560, 210),
    (110, 360, 250, 470),
    (510, 350, 700, 470),
    (940, 360, 1100, 470),
    (1385, 350, 1585, 470),
]
for rect in after:
    x1, y1, x2, y2 = rect
    draw.rectangle([x1, y1, x2, y2], fill='#c6c6c6', outline='#444444', width=2)

# Add parcel labels
try:
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 28)
except Exception:
    font = ImageFont.load_default()

for idx, rect in enumerate(parcels, start=1):
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    label = f'P{idx}'
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    draw.text((cx - text_w / 2, cy - text_h / 2), label, fill=dark, font=font)

# Add small map title
try:
    title_font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 42)
except Exception:
    title_font = font

draw.text((60, 1080), 'Urban Parcel Map Sample', fill='#1f2937', font=title_font)

out_path = os.path.join(out_dir, 'parcel_sample_map.png')
img.save(out_path)
print(f'Created sample parcel map: {out_path}')
