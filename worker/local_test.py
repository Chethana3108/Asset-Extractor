"""Test Universal AI Object Extraction locally: python local_test.py <image_path> [output_dir]
Understands the image with Florence-2, localizes with Grounding DINO, segments with SAM,
and writes isolated transparent PNG cutouts and annotated preview."""
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
print(f"Loading Universal 3-Model Pipeline and extracting objects from '{src}'...")
settings = Settings()
ex = ObjectExtractor(settings)
data = open(src, "rb").read()
cutouts, diag = ex.extract(data, return_diagnostics=True)

print(f"\n==================================================")
print(f"UNIVERSAL AI EXTRACTION SUMMARY:")
print(f"Dynamic Labels Identified: {diag.get('dynamic_labels_identified', [])}")
print(f"Total Objects Extracted:   {len(cutouts)}")
print(f"Review Needed:             {diag.get('review_needed', False)}")
print(f"Diagnostics:               {diag.get('dino_raw_boxes_count', 0)} raw DINO boxes, "
      f"{diag.get('containment_suppressed_count', 0)} sub-parts suppressed")
print(f"==================================================\n")

preview = Image.open(src).convert("RGB")
d = ImageDraw.Draw(preview)

for i, c in enumerate(cutouts, 1):
    completed_tag = "completed" if c.is_completed else "raw"
    filename = f"{i:02d}_{c.label}_{c.confidence_status}_{completed_tag}_{c.kind}.png"
    filepath = os.path.join(out, filename)
    open(filepath, "wb").write(c.png)
    color = "#00E676" if c.confidence_status == "confirmed" else "#FF9100"
    d.rectangle(c.bbox, outline=color, width=4)
    tag = f"{c.label} {c.score:.2f} [{c.confidence_status}]"
    if c.is_completed:
        tag += f" (inpainted occ:{c.occlusion_ratio:.2f})"
    d.text((c.bbox[0] + 6, max(0, c.bbox[1] - 16)), tag, fill=color)
    print(f"  [{i:02d}] {c.label:12s} (score: {c.score:.2f}, status: {c.confidence_status:12s}, area: {c.area_ratio:.3f}, completed: {str(c.is_completed):5s}, occ_ratio: {c.occlusion_ratio:.3f}) -> {filename}")

preview_path = os.path.join(out, "_preview.jpg")
preview.save(preview_path)
print(f"\nPreview saved to: {preview_path}")
print(f"All {len(cutouts)} cutouts saved to directory: '{out}/'\n")
