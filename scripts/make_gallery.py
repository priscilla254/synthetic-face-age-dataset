"""Build docs/gallery/overview.png: one example per ethnicity label x age group.

Pass the folder holding the final images (img0001.png ...). Uses a seeded pick so
the sheet is reproducible. Images are synthetic (no real people).
"""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from facedata.dataset.age_bins import AGE_GROUPS

ROOT = Path(__file__).resolve().parents[1]
image_dir = Path(sys.argv[1])
meta = pd.read_csv(ROOT / "data/metadata/final_dataset_original_labels.csv")
T, LABEL_W, HEAD_H = 150, 150, 24
ethnicities = sorted(meta["ethnicity"].dropna().unique())
sheet = Image.new("RGB", (LABEL_W + T * len(AGE_GROUPS), HEAD_H + T * len(ethnicities)), "white")
draw = ImageDraw.Draw(sheet)
for j, group in enumerate(AGE_GROUPS):
    draw.text((LABEL_W + j * T + 6, 6), group, fill="black")
for i, eth in enumerate(ethnicities):
    draw.text((6, HEAD_H + i * T + T // 2 - 5), eth, fill="black")
    for j, group in enumerate(AGE_GROUPS):
        pool = meta[(meta.ethnicity == eth) & (meta.age_group == group)]
        if pool.empty:
            continue
        name = pool.sample(1, random_state=7).iloc[0]["filename"]
        with Image.open(image_dir / name) as im:
            tile = im.convert("RGB").resize((T, T))
        sheet.paste(tile, (LABEL_W + j * T, HEAD_H + i * T))
out = ROOT / "docs/gallery/overview.jpg"
out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out, quality=85)
print("wrote", out, sheet.size)
