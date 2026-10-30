"""FastAPI server for Universal Object Extraction Engine.

Features:
- Dynamic Image Understanding via Florence-2 (no hardcoded category rules).
- Grounding DINO open-vocabulary localization.
- SAM pixel-precise transparent cutouts.
- Confidence validation (confirmed vs needs_review).
- Interactive endpoints for Postman / cURL testing and DAM integration.
"""
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
    title="Universal AI Object Extraction Engine",
    description="Universal 3-Model Pipeline: Florence-2 (Image Understanding) + Grounding DINO (Localization) + SAM (Segmentation)",
    version="4.0.0"
)

# Initialize settings and extractor
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
        "version": "4.0-universal-engine",
        "cuda_available": cuda_available,
        "device": device_name,
        "models": {
            "vlm_model": settings.vlm_model if settings else "microsoft/Florence-2-base",
            "detector_model": settings.detector_model if settings else "IDEA-Research/grounding-dino-base",
            "sam_model": settings.sam_model if settings else "facebook/sam-vit-large",
        },
        "use_dynamic_vlm": settings.use_dynamic_vlm if settings else True,
        "labels": settings.labels if settings else [],
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
    max_area_ratio: Optional[float] = Form(default=None),
    containment_threshold: Optional[float] = Form(default=None),
    confidence_review_threshold: Optional[float] = Form(default=None),
    nms_iou: Optional[float] = Form(default=None),
    max_objects: Optional[int] = Form(default=None),
    min_objects_to_split: Optional[int] = Form(default=None),
    use_dynamic_vlm: bool = Form(default=True),
    use_multi_pass: Optional[bool] = Form(default=None),
):
    """Upload any image file and extract isolated product cutouts with dynamic AI understanding."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    try:
        custom_labels = _parse_labels(labels)
        dynamic_flag = use_dynamic_vlm if use_multi_pass is None else use_multi_pass
        cutouts, diagnostics = extractor.extract(
            image_bytes,
            labels=custom_labels,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            containment_threshold=containment_threshold,
            confidence_review_threshold=confidence_review_threshold,
            nms_iou=nms_iou,
            max_objects=max_objects,
            min_objects_to_split=min_objects_to_split,
            use_dynamic_vlm=dynamic_flag,
            return_diagnostics=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Universal extraction failed: {str(e)}")

    results = []
    for i, c in enumerate(cutouts, 1):
        item = {
            "index": i,
            "label": c.label,
            "score": round(float(c.score), 4),
            "confidence_status": c.confidence_status,
            "area_ratio": c.area_ratio,
            "is_completed": c.is_completed,
            "occlusion_ratio": c.occlusion_ratio,
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
        "dynamic_labels_identified": diagnostics.get("dynamic_labels_identified", []),
        "diagnostics": {
            "vlm_raw_boxes_count": diagnostics.get("vlm_raw_boxes_count", 0),
            "dino_raw_boxes_count": diagnostics.get("dino_raw_boxes_count", 0),
            "area_filtered_count": diagnostics.get("area_filtered_count", 0),
            "containment_suppressed_count": diagnostics.get("containment_suppressed_count", 0),
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
    max_area_ratio: Optional[float] = Form(default=None),
    containment_threshold: Optional[float] = Form(default=None),
    confidence_review_threshold: Optional[float] = Form(default=None),
    nms_iou: Optional[float] = Form(default=None),
    max_objects: Optional[int] = Form(default=None),
    min_objects_to_split: Optional[int] = Form(default=None),
    use_dynamic_vlm: bool = Form(default=True),
    use_multi_pass: Optional[bool] = Form(default=None),
):
    """Upload an image file and receive the annotated image with color-coded bounding boxes and confidence flags."""
    image_bytes = await file.read()
    try:
        custom_labels = _parse_labels(labels)
        dynamic_flag = use_dynamic_vlm if use_multi_pass is None else use_multi_pass
        cutouts, diagnostics = extractor.extract(
            image_bytes,
            labels=custom_labels,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            containment_threshold=containment_threshold,
            confidence_review_threshold=confidence_review_threshold,
            nms_iou=nms_iou,
            max_objects=max_objects,
            min_objects_to_split=min_objects_to_split,
            use_dynamic_vlm=dynamic_flag,
            return_diagnostics=True
        )
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw detected cutout bounding boxes
        # Green = confirmed, Amber = needs_review
        for c in cutouts:
            color = "#00E676" if c.confidence_status == "confirmed" else "#FF9100"
            draw.rectangle(c.bbox, outline=color, width=4)
            tag = f"{c.label} ({c.score:.2f}) [{c.confidence_status}]"
            draw.text((c.bbox[0] + 6, max(0, c.bbox[1] - 16)), tag, fill=color)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)

        dyn_labels = ",".join(diagnostics.get("dynamic_labels_identified", []))
        headers = {
            "X-Total-Extracted": str(len(cutouts)),
            "X-Review-Needed": str(diagnostics.get("review_needed", False)),
            "X-Dynamic-Labels": dyn_labels[:200],
        }
        return Response(content=buf.getvalue(), media_type="image/jpeg", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 6006))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
