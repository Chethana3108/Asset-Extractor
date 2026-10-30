"""Tune thresholds locally without Content Hub: python local_test.py shot.jpg out/
Writes each cutout PNG plus a preview with boxes drawn."""
import os
import sys

from PIL import Image, ImageDraw

os.environ.setdefault("CH_BASE_URL", "http://unused")
os.environ.setdefault("CH_CLIENT_ID", "x")
os.environ.setdefault("CH_CLIENT_SECRET", "x")
os.environ.setdefault("SB_CONNECTION", "x")

from config import Settings          # noqa: E402
from extractor import ObjectExtractor  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
ex = ObjectExtractor(Settings())
data = open(src, "rb").read()
cutouts = ex.extract(data)

preview = Image.open(src).convert("RGB")
d = ImageDraw.Draw(preview)
for i, c in enumerate(cutouts, 1):
    open(os.path.join(out, f"{i:02d}_{c.label}_{c.kind}.png"), "wb").write(c.png)
    d.rectangle(c.bbox, outline="red", width=4)
    d.text((c.bbox[0] + 6, c.bbox[1] + 6), f"{c.label} {c.score:.2f}", fill="red")
preview.save(os.path.join(out, "_preview.jpg"))
print(f"{len(cutouts)} outputs -> {out}")
