"""FastAPI server for interactive testing via Postman / cURL and external HTTP calls."""
import base64
import io
import os
from typing import Optional

# Set fallback defaults before config is imported so testing doesn't crash without Content Hub env vars
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
    title="Asset Extractor API",
    description="Zero-shot object detection (Grounding DINO) and segmentation (SAM) for asset cutout extraction",
    version="1.0.0"
)

# Initialize settings and extractor
settings = None
extractor = None


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
        "cuda_available": cuda_available,
        "device": device_name,
        "labels": settings.labels if settings else []
    }


@app.post("/extract")
async def extract_objects(
    file: UploadFile = File(...),
    return_base64: bool = Form(default=True)
):
    """Upload an image file and receive detected objects, bounding boxes, confidence scores, and cutout PNGs."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    try:
        cutouts = extractor.extract(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    results = []
    for i, c in enumerate(cutouts, 1):
        item = {
            "index": i,
            "label": c.label,
            "score": round(float(c.score), 4),
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
        "objects": results
    }


@app.post("/extract/preview")
async def extract_preview(file: UploadFile = File(...)):
    """Upload an image file and receive the annotated image (JPEG) with bounding boxes and labels drawn."""
    image_bytes = await file.read()
    try:
        cutouts = extractor.extract(image_bytes)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        for c in cutouts:
            draw.rectangle(c.bbox, outline="red", width=4)
            draw.text((c.bbox[0] + 6, c.bbox[1] + 6), f"{c.label} {c.score:.2f}", fill="red")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
