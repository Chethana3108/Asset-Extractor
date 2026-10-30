"""FastAPI server for Universal Object Extraction Engine.

4-Stage Architecture:
1. GPT Vision Semantic Analysis
2. Concept Segmentation (Grounded-SAM / SAM 3.1)
3. Python Mask Validation & Quality Assurance
4. Sub-pixel Edge Refinement & Transparent Layer Generation
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
    description="4-Stage Layer Extraction: GPT Vision + Concept Segmentation + Mask QA Validation + Sub-pixel Refinement",
    version="6.0.0"
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
        "version": "6.0-universal-4stage",
        "cuda_available": cuda_available,
        "device": device_name,
        "sam3_available": extractor.sam3_available if extractor else False,
        "models": {
            "gpt_model": settings.gpt_model if settings else "gpt-4o",
            "detector_model": settings.detector_model if settings else "IDEA-Research/grounding-dino-base",
            "sam_model": settings.sam_model if settings else "facebook/sam-vit-large",
        },
    }


def _parse_labels(labels_str: Optional[str]):
    if not labels_str:
        return None
    return [l.strip() for l in labels_str.split(",") if l.strip()]


@app.post("/extract")
async def extract_objects(
    file: UploadFile = File(...),
    return_base64: bool = Form(default=True),
    labels: Optional[str] = Form(default=None),
):
    """Upload any image file and extract isolated product cutouts with 4-stage pipeline."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    try:
        custom_labels = _parse_labels(labels)
        cutouts, diagnostics = extractor.extract(
            image_bytes,
            labels=custom_labels,
            return_diagnostics=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    results = []
    for i, c in enumerate(cutouts, 1):
        item = {
            "index": i,
            "label": c.label,
            "category": c.category,
            "score": round(float(c.score), 4),
            "confidence_status": c.confidence_status,
            "area_ratio": c.area_ratio,
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
        "semantic_inventory": diagnostics.get("semantic_inventory", []),
        "diagnostics": {
            "raw_candidates_count": diagnostics.get("raw_candidates_count", 0),
            "validated_objects_count": diagnostics.get("validated_objects_count", 0),
        },
        "objects": results
    }


@app.post("/extract/preview")
async def extract_preview(
    file: UploadFile = File(...),
    labels: Optional[str] = Form(default=None),
):
    """Upload an image file and receive the annotated image with color-coded bounding boxes."""
    image_bytes = await file.read()
    try:
        custom_labels = _parse_labels(labels)
        cutouts, diagnostics = extractor.extract(
            image_bytes,
            labels=custom_labels,
            return_diagnostics=True
        )
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        for c in cutouts:
            color = "#00E676" if c.confidence_status == "confirmed" else "#FF9100"
            draw.rectangle(c.bbox, outline=color, width=4)
            tag = f"{c.label} ({c.score:.2f}) [{c.confidence_status}]"
            draw.text((c.bbox[0] + 6, max(0, c.bbox[1] - 16)), tag, fill=color)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)

        sem_inv = ",".join(diagnostics.get("semantic_inventory", []))
        headers = {
            "X-Total-Extracted": str(len(cutouts)),
            "X-Review-Needed": str(diagnostics.get("review_needed", False)),
            "X-Semantic-Inventory": sem_inv[:200],
        }
        return Response(content=buf.getvalue(), media_type="image/jpeg", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 6006))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
