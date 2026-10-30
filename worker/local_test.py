"""Test Universal AI Object Extraction locally: python local_test.py <image_path> [output_dir]

Runs the 4-Stage Architecture:
1. GPT Vision Semantic Analysis -> Clean Object Inventory
2. Concept Segmentation (Grounded-SAM / SAM 3.1)
3. Python Mask Validation & Quality Assurance
4. Sub-pixel Edge Refinement & Transparent Layer Generation
"""
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
    print("Example: python local_test.py testing.png output")
    sys.exit(1)

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "output"

if not os.path.exists(src):
    print(f"Error: Image file not found: {src}")
    sys.exit(1)

os.makedirs(out, exist_ok=True)
print(f"Initializing 4-Stage Extraction Pipeline and analyzing '{src}'...")
settings = Settings()
ex = ObjectExtractor(settings)
data = open(src, "rb").read()
cutouts, diag = ex.extract(data, return_diagnostics=True)

print(f"\n{'='*65}")
print(f"UNIVERSAL AI OBJECT EXTRACTION SUMMARY:")
print(f"GPT Semantic Inventory:   {diag.get('semantic_inventory', [])}")
print(f"Raw Candidates Detected: {diag.get('raw_candidates_count', 0)}")
print(f"Validated Layers Extracted: {len(cutouts)}")
print(f"Review Needed:           {diag.get('review_needed', False)}")
print(f"{'='*65}\n")

preview = Image.open(src).convert("RGB")
d = ImageDraw.Draw(preview)

for i, c in enumerate(cutouts, 1):
    safe_name = c.label.lower().replace(" ", "_").replace("/", "-")
    filename = f"{i:02d}_{safe_name}_{c.confidence_status}.png"
    filepath = os.path.join(out, filename)
    open(filepath, "wb").write(c.png)

    # Color code: Green = confirmed, Amber = needs_review
    color = "#00E676" if c.confidence_status == "confirmed" else "#FF9100"
    d.rectangle(c.bbox, outline=color, width=4)
    tag = f"{c.label} ({c.score:.2f}) [{c.confidence_status}]"
    d.text((c.bbox[0] + 6, max(0, c.bbox[1] - 16)), tag, fill=color)

    print(f"  [{i:02d}] {c.label:24s} (score: {c.score:.2f}, status: {c.confidence_status:12s}, "
          f"area: {c.area_ratio:.3f}, cat: {c.category:10s}) -> {filename}")

preview_path = os.path.join(out, "_preview.jpg")
preview.save(preview_path, quality=92)
print(f"\nPreview saved to: {preview_path}")
print(f"All {len(cutouts)} transparent layers saved to: '{out}/'\n")
