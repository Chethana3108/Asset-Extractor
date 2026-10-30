"""Test Universal AI Object Extraction locally: python local_test.py <image_path> [output_dir]

Strict Verification Rules:
1. Every extracted PNG must have EXACTLY the same width, height and coordinate system as the original image.
2. The extracted object stays in its original position; all pixels outside its mask become transparent.
3. No fragments or sub-components (caps, flaps) are extracted as independent layers.
4. Grouped items (scattered tablets/capsules) are unified into a single group layer.
"""
import io
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
source = Image.open(src)
original_size = source.size
print(f"Loaded source image '{src}' with original dimensions: {original_size} (WxH)")

settings = Settings()
ex = ObjectExtractor(settings)
data = open(src, "rb").read()
cutouts, diag = ex.extract(data, return_diagnostics=True)

print(f"\n{'='*70}")
print(f"UNIVERSAL AI OBJECT EXTRACTION SUMMARY:")
print(f"Planned Inventory:       {diag.get('inventory_planned', [])}")
print(f"Total Layers Extracted:  {len(cutouts)}")
print(f"Review Needed:           {diag.get('review_needed', False)}")
print(f"{'='*70}\n")

# Annotated preview image
preview = source.convert("RGB")
d = ImageDraw.Draw(preview)

for i, cutout in enumerate(cutouts, 1):
    output_image = Image.open(io.BytesIO(cutout.png)).convert("RGBA")

    # Strict assertion: full canvas dimensions must match original
    assert output_image.size == original_size, (
        f"Layer {i} has incorrect dimensions: {output_image.size}, expected {original_size}"
    )

    alpha = output_image.getchannel("A")
    non_transparent_box = alpha.getbbox()

    safe_name = cutout.label.lower().replace(" ", "_").replace("/", "-")
    filename = f"{i:02d}_{safe_name}.png"
    filepath = os.path.join(out, filename)
    open(filepath, "wb").write(cutout.png)

    # Annotate on preview
    color = "#00E676" if cutout.confidence_status == "confirmed" else "#FF9100"
    d.rectangle(cutout.bbox, outline=color, width=4)
    tag = f"{cutout.label} ({cutout.score:.2f}) [{cutout.role}]"
    d.text((cutout.bbox[0] + 6, max(0, cutout.bbox[1] - 16)), tag, fill=color)

    print(f"  [{i:02d}] {cutout.label:28s} | size={output_image.size} | "
          f"object_bbox={non_transparent_box} | score={cutout.score:.2f} | role={cutout.role} -> {filename}")

preview_path = os.path.join(out, "_preview.jpg")
preview.save(preview_path, quality=92)
print(f"\nPreview comparison saved to: {preview_path}")
print(f"All {len(cutouts)} full-canvas transparent layers saved to: '{out}/'\n")
