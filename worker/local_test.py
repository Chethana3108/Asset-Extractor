"""Test Object Extraction locally: python local_test.py <image_path> [output_dir]
Uses GPT-4o Vision for image understanding + SAM for pixel-precise masks.
Writes isolated transparent PNG cutouts and an annotated preview."""
import os
import sys

from PIL import Image, ImageDraw

os.environ.setdefault("CH_BASE_URL", "http://unused")
os.environ.setdefault("CH_CLIENT_ID", "x")
os.environ.setdefault("CH_CLIENT_SECRET", "x")
os.environ.setdefault("SB_CONNECTION", "x")

# OPENAI_API_KEY must be set in environment or will be read from config
from config import Settings          # noqa: E402
from extractor import ObjectExtractor  # noqa: E402

if len(sys.argv) < 2:
    print("Usage: python local_test.py <image_path> [output_dir]")
    print("  Ensure OPENAI_API_KEY is set in your environment.")
    print("  Example: OPENAI_API_KEY=sk-... python local_test.py test_product.jpg output")
    sys.exit(1)

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "output"

if not os.path.exists(src):
    print(f"Error: Image file not found: {src}")
    sys.exit(1)

os.makedirs(out, exist_ok=True)
print(f"Loading GPT Vision + SAM pipeline and extracting objects from '{src}'...")
settings = Settings()
ex = ObjectExtractor(settings)
data = open(src, "rb").read()
cutouts, diag = ex.extract(data, return_diagnostics=True)

print(f"\n{'='*60}")
print(f"GPT VISION + SAM EXTRACTION SUMMARY:")
print(f"Objects Identified by GPT: {diag.get('dynamic_labels_identified', [])}")
print(f"Total Layers Extracted:    {len(cutouts)}")
print(f"Review Needed:             {diag.get('review_needed', False)}")
print(f"{'='*60}\n")

preview = Image.open(src).convert("RGB")
d = ImageDraw.Draw(preview)

for i, c in enumerate(cutouts, 1):
    filename = f"{i:02d}_{c.label.replace(' ', '_')}_{c.confidence_status}_{c.kind}.png"
    filepath = os.path.join(out, filename)
    open(filepath, "wb").write(c.png)
    color = "#00E676" if c.confidence_status == "confirmed" else "#FF9100"
    d.rectangle(c.bbox, outline=color, width=4)
    tag = f"{c.label} {c.score:.2f} [{c.confidence_status}]"
    d.text((c.bbox[0] + 6, max(0, c.bbox[1] - 16)), tag, fill=color)
    print(f"  [{i:02d}] {c.label:20s} (score: {c.score:.2f}, status: {c.confidence_status:12s}, "
          f"area: {c.area_ratio:.3f}) -> {filename}")

preview_path = os.path.join(out, "_preview.jpg")
preview.save(preview_path)
print(f"\nPreview saved to: {preview_path}")
print(f"All {len(cutouts)} layers saved to directory: '{out}/'\n")
