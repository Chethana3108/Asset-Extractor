"""FastAPI server for Universal Object Extraction Engine.

Features:
- GPT Asset Decomposition Planning
- SAM 3 Text-Promptable Concept Segmentation
- Python Mask QA & Containment Validation
- Full-Canvas Transparent Layer Export
"""
import base64
import io
import os
from typing import Optional

os.environ.setdefault("CH_BASE_URL", "http://localhost")
os.environ.setdefault("CH_CLIENT_ID", "test-client")
os.environ.setdefault("CH_CLIENT_SECRET", "test-secret")
os.environ.setdefault("SB_CONNECTION", "Endpoint=sb://localhost/;SharedAccessKeyName=x;SharedAccessKey=x")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageDraw

from config import Settings
from extractor import ObjectExtractor

app = FastAPI(
    title="Universal AI Object Extraction Engine",
    description="Full-Canvas Layer Extraction: GPT Planner + SAM 3 Concept Segmentation + Full-Canvas Layer Export",
    version="7.0.0"
)

settings: Optional[Settings] = None
extractor: Optional[ObjectExtractor] = None


@app.on_event("startup")
def load_models():
    global settings, extractor
    settings = Settings()
    extractor = ObjectExtractor(settings)


@app.get("/health")
def health():
    import torch
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return {
        "status": "healthy",
        "version": "7.0-full-canvas-sam3",
        "cuda_available": cuda_available,
        "device": device_name,
        "models": {
            "gpt_model": settings.gpt_model if settings else "gpt-4o",
            "sam_model": settings.sam_model if settings else "facebook/sam3",
        },
    }


@app.post("/extract")
async def extract_objects(
    file: UploadFile = File(...),
    return_base64: bool = Form(default=True),
):
    """Upload an image file and extract isolated full-canvas product layers."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    try:
        cutouts, diagnostics = extractor.extract(
            image_bytes,
            return_diagnostics=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Layer extraction failed: {str(e)}")

    results = []
    for i, c in enumerate(cutouts, 1):
        item = {
            "index": i,
            "label": c.label,
            "category": c.category,
            "role": c.role,
            "score": round(float(c.score), 4),
            "confidence_status": c.confidence_status,
            "area_ratio": c.area_ratio,
            "canvas_width": c.canvas_width,
            "canvas_height": c.canvas_height,
            "bbox": {
                "x0": c.bbox[0],
                "y0": c.bbox[1],
                "x1": c.bbox[2],
                "y1": c.bbox[3]
            },
            "kind": c.kind
        }
        if return_base64:
            item["png_base64"] = base64.b64encode(c.png).decode("utf-8")
        results.append(item)

    return {
        "filename": file.filename,
        "total_extracted": len(cutouts),
        "review_needed": diagnostics.get("review_needed", False),
        "canvas_dimensions": diagnostics.get("canvas_dimensions", {}),
        "inventory_planned": diagnostics.get("inventory_planned", []),
        "layers": results
    }


@app.post("/extract/preview")
async def extract_preview(
    file: UploadFile = File(...),
):
    """Upload an image file and receive the annotated comparison sheet with color-coded bounding boxes."""
    image_bytes = await file.read()
    try:
        cutouts, diagnostics = extractor.extract(
            image_bytes,
            return_diagnostics=True
        )
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        for c in cutouts:
            color = "#00E676" if c.confidence_status == "confirmed" else "#FF9100"
            draw.rectangle(c.bbox, outline=color, width=4)
            tag = f"{c.label} ({c.score:.2f}) [{c.role}]"
            draw.text((c.bbox[0] + 6, max(0, c.bbox[1] - 16)), tag, fill=color)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)

        headers = {
            "X-Total-Extracted": str(len(cutouts)),
            "X-Review-Needed": str(diagnostics.get("review_needed", False)),
        }
        return Response(content=buf.getvalue(), media_type="image/jpeg", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 6006))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
