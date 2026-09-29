from PIL import Image, ImageDraw
import json
import numpy as np
import math

p = r"recordings\ls20-9607627b.arcagi3solver.af963ffd-6d86-41aa-a715-9b62c514a8d0.recording.jsonl"

rows = [json.loads(x) for x in open(p, encoding="utf-8")]

frames = []

for r in rows:
    frame = r.get("data", {}).get("frame")

    if isinstance(frame, list) and frame:
        grid = np.array(frame[0], dtype=np.uint8)

        if grid.shape == (64, 64):
            frames.append(grid)


# ARC-style visualization palette.
palette = {
    0: (0, 0, 0),
    1: (255, 255, 255),
    2: (204, 204, 204),
    3: (80, 80, 80),
    4: (30, 30, 30),
    5: (255, 0, 0),
    6: (255, 165, 0),
    7: (255, 255, 0),
    8: (0, 255, 0),
    9: (0, 200, 255),
    10: (0, 0, 255),
    11: (180, 0, 255),
    12: (255, 0, 255),
}

selected = [
    0, 5, 10, 15, 20,
    25, 30, 35, 40, 41,
    42, 43, 44, 50, 60,
    70, 75, 79,
]

scale = 6
cell_w = 64 * scale
cell_h = 64 * scale
label_h = 28

cols = 5
rows_count = math.ceil(len(selected) / cols)

sheet = Image.new(
    "RGB",
    (cols * cell_w, rows_count * (cell_h + label_h)),
    "white"
)

draw = ImageDraw.Draw(sheet)

for n, frame_index in enumerate(selected):

    if frame_index >= len(frames):
        continue

    grid = frames[frame_index]

    img = Image.new("RGB", (64, 64), "black")
    pixels = img.load()

    for y in range(64):
        for x in range(64):
            value = int(grid[y, x])
            pixels[x, y] = palette.get(
                value,
                (128, 128, 128)
            )

    img = img.resize(
        (cell_w, cell_h),
        Image.Resampling.NEAREST
    )

    x = (n % cols) * cell_w
    y = (n // cols) * (cell_h + label_h)

    sheet.paste(img, (x, y))

    draw.text(
        (x + 5, y + 5),
        f"Frame {frame_index}",
        fill="yellow"
    )

output = r"ls20_contact_sheet.png"
sheet.save(output)

print("Saved:", output)
print("Frames:", len(frames))
