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
        "version": "3.0-multiscale",
        "cuda_available": cuda_available,
        "device": device_name,
        "use_multi_pass": settings.use_multi_pass if settings else True,
        "labels": settings.labels if settings else []
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
    box_threshold: Optional[float] = Form(default=None),
    text_threshold: Optional[float] = Form(default=None),
    min_area_ratio: Optional[float] = Form(default=None),
    nms_iou: Optional[float] = Form(default=None),
    max_objects: Optional[int] = Form(default=None),
    min_objects_to_split: Optional[int] = Form(default=None),
    use_multi_pass: bool = Form(default=True),
):
    """Upload an image file and receive detected objects, bounding boxes, confidence scores, and cutout PNGs."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    try:
        custom_labels = _parse_labels(labels)
        cutouts, diagnostics = extractor.extract(
            image_bytes,
            labels=custom_labels,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            min_area_ratio=min_area_ratio,
            nms_iou=nms_iou,
            max_objects=max_objects,
            min_objects_to_split=min_objects_to_split,
            use_multi_pass=use_multi_pass,
            return_diagnostics=True
        )
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
        "diagnostics": {
            "raw_candidates_by_category": diagnostics.get("raw_by_category", {}),
            "kept_by_category": diagnostics.get("kept_by_category", {}),
            "total_candidate_boxes": len(diagnostics.get("all_raw_boxes", [])),
        },
        "objects": results
    }


@app.post("/extract/preview")
async def extract_preview(
    file: UploadFile = File(...),
    labels: Optional[str] = Form(default=None),
    box_threshold: Optional[float] = Form(default=None),
    text_threshold: Optional[float] = Form(default=None),
    min_area_ratio: Optional[float] = Form(default=None),
    nms_iou: Optional[float] = Form(default=None),
    max_objects: Optional[int] = Form(default=None),
    min_objects_to_split: Optional[int] = Form(default=None),
    use_multi_pass: bool = Form(default=True),
):
    """Upload an image file and receive the annotated image (JPEG) with bounding boxes and labels drawn."""
    image_bytes = await file.read()
    try:
        custom_labels = _parse_labels(labels)
        cutouts, diagnostics = extractor.extract(
            image_bytes,
            labels=custom_labels,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            min_area_ratio=min_area_ratio,
            nms_iou=nms_iou,
            max_objects=max_objects,
            min_objects_to_split=min_objects_to_split,
            use_multi_pass=use_multi_pass,
            return_diagnostics=True
        )
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw detected cutout bounding boxes
        for c in cutouts:
            draw.rectangle(c.bbox, outline="red", width=4)
            draw.text((c.bbox[0] + 6, c.bbox[1] + 6), f"{c.label} {c.score:.2f}", fill="red")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)

        raw_count = len(diagnostics.get("all_raw_boxes", []))
        headers = {
            "X-Raw-Detections-Count": str(raw_count),
            "X-Cutouts-Count": str(len(cutouts)),
            "X-Multi-Pass": str(use_multi_pass),
        }
        return Response(content=buf.getvalue(), media_type="image/jpeg", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 6006))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
