"""Universal AI Object Extraction Engine — GPT Vision + SAM Architecture.

Architecture:
1. GPT-4o Vision: Analyzes the image intelligently — identifies every distinct physical
   object with bounding boxes. No hardcoded labels, no detection rules.
2. SAM (Segment Anything Model): Generates pixel-precise masks from GPT's bounding boxes.
3. Post-processing: Refines mask edges and produces clean transparent RGBA PNG cutouts.

This replaces the old Florence-2 + Grounding DINO + LaMa pipeline entirely.
"""
from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from openai import OpenAI
from PIL import Image, ImageOps
from transformers import SamModel, SamProcessor

from config import Settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class Cutout:
    label: str
    score: float
    bbox: tuple[int, int, int, int]           # x0, y0, x1, y1 in ORIGINAL image pixels
    png: bytes                                 # RGBA cutout (or RGB crop in 'crop' mode)
    kind: str                                  # 'cutout' | 'crop'
    confidence_status: str = "confirmed"       # 'confirmed' | 'needs_review'
    area_ratio: float = 0.0                    # Fraction of image area occupied
    is_completed: bool = False                 # Kept for backward compat (always False now)
    occlusion_ratio: float = 0.0               # Kept for backward compat (always 0 now)


# ---------------------------------------------------------------------------
# GPT Prompts
# ---------------------------------------------------------------------------
GPT_SYSTEM_PROMPT = """You are a precision visual element extraction expert. Your task is to identify every distinct physical object in an image so each can be extracted as an independent transparent layer.

STRICT RULES:
1. Identify every physically separate, distinct object visible in the image.
2. Each object must appear exactly ONCE — absolutely no duplicates.
3. Do NOT list sub-parts of objects separately (e.g., do NOT list a "bottle cap" if you already list the "bottle").
4. Do NOT include: background/surface, shadows, reflections, pure text labels, decorative patterns.
5. Bounding boxes must tightly enclose each individual object.
6. Even partially occluded objects should be listed with their visible bounding box.
7. Small objects (individual pills, capsules, etc.) that form a scattered group can be listed as one group item.

Return ONLY valid JSON — no markdown, no code fences, no explanation."""

GPT_USER_PROMPT = """Analyze this image carefully. The image dimensions are {width} x {height} pixels.

Identify EVERY distinct physical object in the image and provide:
- "label": short descriptive name (1-3 words, e.g. "open box", "pill bottle", "blister pack", "foil sachet")
- "bbox": tight bounding box as [x_min, y_min, x_max, y_max] in pixel coordinates where (0,0) is top-left
- "confidence": your confidence (0.0 to 1.0) that this is a real, distinct object worth extracting

Return JSON: {{"objects": [{{"label": "...", "bbox": [x1, y1, x2, y2], "confidence": 0.95}}, ...]}}"""


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------
class ObjectExtractor:
    """GPT Vision + SAM Extraction Pipeline."""

    def __init__(self, s: Settings):
        self.s = s
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        log.info("Initializing ObjectExtractor (GPT Vision + SAM) on %s (dtype=%s)",
                 self.device, self.dtype)

        # 1. OpenAI GPT Vision client
        self.gpt = OpenAI(api_key=s.openai_api_key)
        log.info("OpenAI client ready (model: %s)", s.gpt_model)

        # 2. SAM — Segment Anything Model for pixel-precise masks
        log.info("Loading SAM Model: %s", s.sam_model)
        self.sam_proc = SamProcessor.from_pretrained(s.sam_model)
        self.sam = SamModel.from_pretrained(
            s.sam_model, torch_dtype=self.dtype
        ).to(self.device).eval()
        log.info("Models loaded successfully.")

    # ------------------------------------------------------------------
    # Image Loading
    # ------------------------------------------------------------------
    def _load(self, data: bytes) -> tuple[Image.Image, Image.Image, float]:
        """Load image, handle EXIF rotation, optionally downscale for SAM."""
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        rgb = img.convert("RGB")
        scale = 1.0
        m = max(rgb.size)
        if m > self.s.max_working_side:
            scale = self.s.max_working_side / m
            rgb = rgb.resize(
                (round(rgb.width * scale), round(rgb.height * scale)),
                Image.LANCZOS,
            )
        return img.convert("RGB"), rgb, scale

    # ------------------------------------------------------------------
    # Step 1: GPT Vision — Intelligent Image Understanding
    # ------------------------------------------------------------------
    def _analyze_with_gpt(self, image_data: bytes, W: int, H: int) -> list[dict]:
        """Send image to GPT-4o Vision, receive structured object list with bounding boxes."""
        b64 = base64.b64encode(image_data).decode("utf-8")
        mime = "image/png" if image_data[:4] == b"\x89PNG" else "image/jpeg"

        try:
            response = self.gpt.chat.completions.create(
                model=self.s.gpt_model,
                messages=[
                    {"role": "system", "content": GPT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": GPT_USER_PROMPT.format(width=W, height=H),
                            },
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=2000,
                temperature=0.1,
            )

            raw = response.choices[0].message.content
            result = json.loads(raw)
            objects = result.get("objects", [])

            # Validate and clamp bounding boxes
            validated: list[dict] = []
            img_area = float(W * H)

            for obj in objects:
                bbox = obj.get("bbox", [])
                if len(bbox) != 4:
                    log.warning("Skipping object with invalid bbox: %s", obj)
                    continue

                x1, y1, x2, y2 = [float(v) for v in bbox]

                # Clamp to image bounds
                x1 = max(0.0, min(x1, float(W)))
                y1 = max(0.0, min(y1, float(H)))
                x2 = max(0.0, min(x2, float(W)))
                y2 = max(0.0, min(y2, float(H)))

                # Ensure valid box (non-zero area)
                if x2 <= x1 + 5 or y2 <= y1 + 5:
                    log.warning("Skipping tiny/invalid box for '%s': [%.0f,%.0f,%.0f,%.0f]",
                                obj.get("label", "?"), x1, y1, x2, y2)
                    continue

                area_ratio = ((x2 - x1) * (y2 - y1)) / img_area
                if area_ratio < self.s.min_area_ratio:
                    continue
                if area_ratio > self.s.max_area_ratio:
                    continue

                validated.append({
                    "label": str(obj.get("label", "object")).strip(),
                    "bbox": [x1, y1, x2, y2],
                    "confidence": min(1.0, max(0.0, float(obj.get("confidence", 0.9)))),
                    "area_ratio": round(area_ratio, 4),
                })

            # Respect max_objects limit
            validated = validated[: self.s.max_objects]

            log.info("GPT Vision identified %d objects: %s",
                     len(validated), [v["label"] for v in validated])
            return validated

        except Exception as e:
            log.error("GPT Vision analysis failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Step 2: SAM — Pixel-Precise Mask Generation
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _segment(self, img: Image.Image, boxes: list[list[float]]) -> np.ndarray:
        """Generate pixel-precise masks using SAM with bounding box prompts.

        Uses multimask_output=True and selects the highest-IoU mask per box
        for best quality.
        """
        inputs = self.sam_proc(
            img, input_boxes=[boxes], return_tensors="pt"
        ).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        out = self.sam(**inputs, multimask_output=True)
        pred_masks = out.pred_masks.float().cpu()    # (1, N, 3, H, W)
        iou_scores = out.iou_scores.float().cpu()    # (1, N, 3)

        # Select highest-IoU mask for each detection
        best_idx = iou_scores[0].argmax(dim=-1)      # (N,)
        selected = []
        for k in range(pred_masks.shape[1]):
            selected.append(pred_masks[0, k, best_idx[k]])

        # Reshape to (1, N, 1, H, W) for post_process_masks
        selected_masks = torch.stack(selected).unsqueeze(0).unsqueeze(2)
        masks = self.sam_proc.image_processor.post_process_masks(
            selected_masks,
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        return masks[0][:, 0].numpy().astype(bool)   # (N, H, W)

    # ------------------------------------------------------------------
    # Step 3: Mask Refinement — Smooth Anti-Aliased Edges
    # ------------------------------------------------------------------
    @staticmethod
    def _refine_mask(mask_bool: np.ndarray, target_W: int, target_H: int) -> np.ndarray:
        """Upscale boolean mask to target resolution with clean anti-aliased edges.

        Returns uint8 alpha channel [0..255].
        """
        raw = mask_bool.astype(np.uint8) * 255

        # Upscale if mask is at working resolution (different from target)
        if raw.shape[1] != target_W or raw.shape[0] != target_H:
            pil_mask = Image.fromarray(raw, mode="L").resize(
                (target_W, target_H), Image.BILINEAR
            )
            raw = np.array(pil_mask)

        # Threshold to binary
        binary = (raw > 127).astype(np.uint8)

        # Very gentle 1px erosion to trim background fringe at edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(binary, kernel, iterations=1)

        # Gaussian blur for natural anti-aliased edge transition
        smooth = cv2.GaussianBlur(eroded.astype(np.float32) * 255.0, (3, 3), 0)

        # Interior stays fully opaque; only the edge band is soft
        alpha = np.where(eroded > 0, 255.0, smooth)
        return np.clip(alpha, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Main Extraction Pipeline
    # ------------------------------------------------------------------
    def extract(
        self,
        data: bytes,
        labels: list[str] | None = None,
        return_diagnostics: bool = False,
        # Accept legacy parameters for backward compatibility (ignored)
        **kwargs,
    ) -> list[Cutout] | tuple[list[Cutout], dict]:
        """GPT Vision + SAM extraction pipeline.

        1. GPT-4o Vision analyzes the image → structured object list with bounding boxes.
        2. SAM generates pixel-precise masks from the bounding boxes.
        3. Masks are refined and applied to produce transparent RGBA cutouts.
        """
        full, work, scale = self._load(data)
        W, H = full.size
        w, h = work.size

        # ---- Step 1: GPT understands the image ----
        gpt_objects = self._analyze_with_gpt(data, W, H)

        diagnostics = {
            "dynamic_labels_identified": [o["label"] for o in gpt_objects],
            "vlm_raw_boxes_count": 0,
            "dino_raw_boxes_count": 0,
            "area_filtered_count": 0,
            "containment_suppressed_count": 0,
            "total_extracted": 0,
            "review_needed": False,
            "detected_objects": [],
        }

        if not gpt_objects:
            log.info("GPT found no objects to extract.")
            return ([], diagnostics) if return_diagnostics else []

        # ---- Step 2: Scale GPT boxes to working resolution → SAM ----
        work_boxes = []
        for obj in gpt_objects:
            x1, y1, x2, y2 = obj["bbox"]
            work_boxes.append([
                x1 * scale, y1 * scale,
                x2 * scale, y2 * scale,
            ])

        masks = self._segment(work, work_boxes)

        # ---- Step 3: Refine masks and produce cutouts ----
        # Pre-refine all masks to full resolution
        full_alphas = [self._refine_mask(masks[k], W, H) for k in range(len(gpt_objects))]

        results: list[Cutout] = []
        for i, obj in enumerate(gpt_objects):
            label = obj["label"]
            confidence = obj["confidence"]
            area_ratio = obj["area_ratio"]
            x1, y1, x2, y2 = obj["bbox"]

            # Add padding around the bounding box
            pad_x = (x2 - x1) * self.s.padding_ratio
            pad_y = (y2 - y1) * self.s.padding_ratio
            bx = (
                max(0, int(x1 - pad_x)),
                max(0, int(y1 - pad_y)),
                min(W, int(x2 + pad_x)),
                min(H, int(y2 + pad_y)),
            )
            rx0, ry0, rx1, ry1 = bx

            # Crop the alpha channel to the bounding box region
            alpha_crop = full_alphas[i][ry0:ry1, rx0:rx1]

            if not np.any(alpha_crop > 127):
                log.warning("SAM mask empty for '%s', falling back to crop", label)
                results.append(Cutout(
                    label=label, score=confidence, bbox=bx,
                    png=_png(full.crop(bx)),
                    kind="crop",
                    confidence_status="confirmed",
                    area_ratio=area_ratio,
                ))
                continue

            # Apply alpha mask to produce transparent cutout
            rgba = full.convert("RGBA").crop(bx)
            rgba.putalpha(Image.fromarray(alpha_crop))

            status = "confirmed" if confidence >= 0.5 else "needs_review"
            results.append(Cutout(
                label=label,
                score=confidence,
                bbox=bx,
                png=_png(rgba),
                kind="cutout",
                confidence_status=status,
                area_ratio=area_ratio,
            ))

        # ---- Diagnostics ----
        review_needed = any(r.confidence_status == "needs_review" for r in results)
        diagnostics.update({
            "total_extracted": len(results),
            "review_needed": review_needed,
            "detected_objects": [
                {
                    "label": o["label"],
                    "score": round(o["confidence"], 4),
                    "confidence_status": "confirmed" if o["confidence"] >= 0.5 else "needs_review",
                    "area_ratio": o["area_ratio"],
                    "box": o["bbox"],
                }
                for o in gpt_objects
            ],
        })

        log.info("Extracted %d cutout assets (review_needed=%s)", len(results), review_needed)
        return (results, diagnostics) if return_diagnostics else results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
