"""Tune thresholds locally without Content Hub: python local_test.py shot.jpg output/
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

if len(sys.argv) < 2:
    print("Usage: python local_test.py <image_path> [output_dir]")
    print("Example: python local_test.py test_product.jpg output")
    sys.exit(1)

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "output"

if not os.path.exists(src):
    print(f"Error: Image file not found: {src}")
    sys.exit(1)

os.makedirs(out, exist_ok=True)
print(f"Loading models and extracting objects from '{src}'...")
ex = ObjectExtractor(Settings())
data = open(src, "rb").read()
cutouts = ex.extract(data)

print(f"\nExtraction complete! Found {len(cutouts)} objects:")
preview = Image.open(src).convert("RGB")
d = ImageDraw.Draw(preview)

for i, c in enumerate(cutouts, 1):
    filename = f"{i:02d}_{c.label}_{c.kind}.png"
    filepath = os.path.join(out, filename)
    open(filepath, "wb").write(c.png)
    d.rectangle(c.bbox, outline="red", width=4)
    d.text((c.bbox[0] + 6, c.bbox[1] + 6), f"{c.label} {c.score:.2f}", fill="red")
    print(f"  [{i:02d}] {c.label:15s} (score: {c.score:.2f}) -> {filename}  bbox: {c.bbox}")

preview_path = os.path.join(out, "_preview.jpg")
preview.save(preview_path)
print(f"\nPreview saved to: {preview_path}")
print(f"All {len(cutouts)} cutouts saved to directory: '{out}/'\n")
