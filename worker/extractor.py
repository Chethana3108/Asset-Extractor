"""Universal AI Object Extraction Engine — GPT Decomposition + SAM 3 Full-Canvas Layer Extraction.

Rules:
1. Every extracted PNG has EXACTLY the same width, height and coordinate system as the original image.
   The extracted object remains in its original position; all pixels outside its mask become transparent.
2. Only meaningful, independent objects become layers. Sub-components (flaps, caps, blister cavities)
   are never extracted separately. Grouped items (scattered tablets, capsules) become single unified group layers.
3. GPT Vision acts as the semantic asset decomposition planner, determining what to extract.
4. SAM 3 generates pixel-level masks via text-promptable concept segmentation.
"""
from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageOps

from config import Settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class Cutout:
    label: str
    score: float
    bbox: tuple[int, int, int, int]           # [x0, y0, x1, y1] non-transparent extent in original pixels
    png: bytes                                 # Full-canvas transparent RGBA PNG
    kind: str = "cutout"
    confidence_status: str = "confirmed"       # 'confirmed' | 'needs_review'
    area_ratio: float = 0.0                    # Fraction of image area occupied
    category: str = "product"
    canvas_width: int = 0
    canvas_height: int = 0
    role: str = "standalone"


# ---------------------------------------------------------------------------
# GPT Asset Decomposition Planner Prompt
# ---------------------------------------------------------------------------
GPT_CATALOG_PROMPT = """You are an expert visual asset decomposition planner.
Analyze the complete image and identify only meaningful, independent visual objects that should become separate image layers.

IMPORTANT RULES:
1. Treat each complete physical product or independent object as one layer.
2. Never extract structural parts of an object separately.
   Examples:
   - Box flaps are part of the box.
   - Bottle caps are part of the bottle.
   - Handles are part of their parent object.
   - Packaging folds are part of the packaging.
   - Product surfaces and printed details are not separate objects.
3. Do not create separate layers for tiny fragments, edges, reflections, shadows, highlights or visual imperfections.
4. Do not extract small objects that are physically part of a larger assembled object.
5. If several loose, similar small objects form one visual group, identify them as one group layer.
   Examples: scattered tablets, capsules, small beads.
6. Separate objects only when they are genuinely independent visible objects.
7. Do not include the background, floor or studio surface.
8. Do not invent objects that are not visible.
9. Do not split a single object into multiple overlapping layers.
10. Do not force a minimum number of objects. If only two meaningful objects exist, return two.

For every object return:
- name: short descriptive name (e.g. "open cardboard box", "white medicine bottle", "blister pack of yellow tablets", "scattered white tablets")
- prompt: concise visual phrase for SAM 3 segmentation (e.g. "open cardboard box", "white medicine bottle", "blister pack of yellow tablets", "scattered tablets")
- role: "standalone" or "group"
- parent: null if independent, or name of parent object if it's a part of something
- extract: true or false
- category: e.g. "box", "bottle", "blister pack", "sachet", "tablet", "capsule", "product"
- reason: why it should be an independent layer

Return valid JSON:
{
  "inventory": [
    {
      "name": "open cardboard box",
      "prompt": "open cardboard box",
      "role": "standalone",
      "parent": null,
      "extract": true,
      "category": "box",
      "reason": "Complete independent packaging object"
    }
  ]
}"""


# ---------------------------------------------------------------------------
# Mask Helpers
# ---------------------------------------------------------------------------
def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate intersection over union between two boolean masks."""
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return float(intersection / max(union, 1))


def _clean_mask(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    """Remove tiny isolated noise fragments from a mask using connected components."""
    mask_u8 = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8)
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned[labels == i] = 1
    return cleaned.astype(bool)


# ---------------------------------------------------------------------------
# Object Extractor Engine
# ---------------------------------------------------------------------------
class ObjectExtractor:
    """GPT Asset Decomposition Planner + SAM 3 Full-Canvas Layer Extraction."""

    def __init__(self, s: Settings):
        self.s = s
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        log.info("Initializing ObjectExtractor on %s (dtype=%s)", self.device, self.dtype)

        # Stage 1: GPT Vision Setup
        self.openai_key = s.openai_api_key
        log.info("OpenAI GPT Vision configured (model: %s)", s.gpt_model)

        # Stage 2: SAM 3 / Segmentation Setup
        log.info("Loading SAM 3 model: %s", s.sam_model)
        self.sam_processor = None
        self.sam_model = None

        try:
            from transformers import Sam3Model, Sam3Processor
            self.sam_processor = Sam3Processor.from_pretrained(s.sam_model)
            self.sam_model = Sam3Model.from_pretrained(
                s.sam_model,
                torch_dtype=self.dtype
            ).to(self.device).eval()
            log.info("SAM 3 loaded successfully from %s", s.sam_model)
        except Exception as e:
            log.warning("Could not load Sam3Model/Sam3Processor directly: %s", e)
            # Fallback to AutoProcessor / AutoModel
            try:
                from transformers import AutoModel, AutoProcessor
                log.info("Attempting AutoModel/AutoProcessor load for %s", s.sam_model)
                self.sam_processor = AutoProcessor.from_pretrained(s.sam_model)
                self.sam_model = AutoModel.from_pretrained(
                    s.sam_model,
                    torch_dtype=self.dtype
                ).to(self.device).eval()
                log.info("SAM 3 loaded via AutoModel successfully")
            except Exception as e2:
                log.error("Failed to load SAM 3 model (%s). Make sure transformers is updated and HuggingFace terms accepted.", e2)
                raise RuntimeError(
                    f"SAM 3 model '{s.sam_model}' could not be loaded: {e2}. "
                    "Ensure access is granted at https://huggingface.co/facebook/sam3 and run `huggingface-cli login`."
                ) from e2

    # ------------------------------------------------------------------
    # Stage 1: GPT Vision Semantic Analysis & Validation
    # ------------------------------------------------------------------
    def _semantic_analysis(self, image_data: bytes) -> list[dict]:
        """Query GPT-4o Vision to produce a structured inventory of independent objects."""
        if not self.openai_key:
            log.warning("No OpenAI API key provided. Cannot run visual asset decomposition.")
            return []

        b64 = base64.b64encode(image_data).decode("utf-8")
        mime = "image/png" if image_data[:4] == b"\x89PNG" else "image/jpeg"

        payload = {
            "model": self.s.gpt_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": GPT_CATALOG_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 1500
        }

        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openai_key}"},
                json=payload,
                timeout=45
            )
            if r.status_code != 200:
                log.error("OpenAI API returned status %d: %s", r.status_code, r.text[:300])
                return []

            data = r.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            items = parsed.get("inventory", [])
            log.info("GPT Vision identified %d raw elements: %s",
                     len(items), [it.get("name") for it in items])
            return items

        except Exception as e:
            log.error("GPT Vision semantic analysis failed: %s", e)
            return []

    def _validate_inventory(self, inventory: list[dict]) -> list[dict]:
        """Filter the GPT inventory: only extractable, independent objects with no parent."""
        valid = []
        for item in inventory:
            # Must explicitly be marked for extraction
            if not item.get("extract", False):
                log.debug("Skipping non-extractable element: %s", item.get("name"))
                continue
            # Sub-components with a parent object are excluded
            if item.get("parent"):
                log.info("Excluding sub-component '%s' (parent: '%s')",
                         item.get("name"), item.get("parent"))
                continue

            name = str(item.get("name", "")).strip()
            prompt = str(item.get("prompt", "")).strip()
            if not name or not prompt:
                continue

            valid.append({
                "name": name,
                "prompt": prompt,
                "role": item.get("role", "standalone"),
                "category": item.get("category", "product"),
                "parent": item.get("parent"),
                "reason": item.get("reason", "")
            })

        log.info("Validated %d independent asset layers to extract", len(valid))
        return valid

    # ------------------------------------------------------------------
    # Stage 2: SAM 3 Concept-by-Concept Segmentation
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _segment_concepts(self, image: Image.Image, inventory: list[dict]) -> list[dict]:
        """Segment each concept individually with SAM 3 text-prompt interface.

        Groups (e.g. scattered tablets, capsules) are merged into one combined mask.
        """
        candidates = []

        for item in inventory:
            prompt = item["prompt"]
            log.info("SAM 3 segmenting prompt: '%s'", prompt)

            inputs = self.sam_processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.sam_model(**inputs)

            # Post-process instance segmentation to get masks, boxes, scores
            if hasattr(self.sam_processor, "post_process_instance_segmentation"):
                results = self.sam_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.s.sam_score_threshold,
                    mask_threshold=self.s.sam_mask_threshold,
                    target_sizes=inputs["original_sizes"].tolist()
                )[0]
            elif hasattr(self.sam_processor, "post_process_masks"):
                # Alternative SAM 3 transformers output format
                masks = self.sam_processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"]
                )[0]
                results = {
                    "masks": masks[:, 0] if masks.ndim == 4 else masks,
                    "scores": getattr(outputs, "iou_scores", torch.ones(len(masks)))[0],
                    "boxes": getattr(outputs, "pred_boxes", torch.zeros((len(masks), 4)))[0]
                }
            else:
                log.error("SAM 3 processor has no supported post-processing method.")
                continue

            masks = results.get("masks", [])
            scores = results.get("scores", [])

            prompt_masks = []
            prompt_scores = []
            for i in range(len(masks)):
                m_tensor = masks[i].detach().cpu()
                score_val = float(scores[i].detach().cpu()) if i < len(scores) else 1.0

                if score_val < self.s.sam_score_threshold:
                    continue

                mask_bool = m_tensor.numpy().astype(bool)
                if not np.any(mask_bool):
                    continue

                prompt_masks.append(mask_bool)
                prompt_scores.append(score_val)

            if not prompt_masks:
                log.info("No mask met threshold (%.2f) for prompt '%s'", self.s.sam_score_threshold, prompt)
                continue

            if item.get("role") == "group":
                # Single unified layer for the entire group (e.g. scattered tablets/capsules)
                combined_mask = np.logical_or.reduce(prompt_masks)
                candidates.append({
                    "label": item["name"],
                    "category": item.get("category", "product"),
                    "role": "group",
                    "mask": combined_mask,
                    "score": max(prompt_scores)
                })
                log.info("Merged %d instances into 1 group layer for '%s'", len(prompt_masks), item["name"])
            else:
                # Standalone instances
                for mask, score in zip(prompt_masks, prompt_scores):
                    candidates.append({
                        "label": item["name"],
                        "category": item.get("category", "product"),
                        "role": "standalone",
                        "mask": mask,
                        "score": score
                    })

        return candidates

    # ------------------------------------------------------------------
    # Stage 3: Mask Validation, Fragment Cleanup & Duplicate Suppression
    # ------------------------------------------------------------------
    def _validate_candidates(self, candidates: list[dict], width: int, height: int) -> list[dict]:
        """Validate masks: remove tiny fragments, suppress duplicates and containment."""
        valid = []
        image_area = width * height
        min_area = int(image_area * self.s.min_object_area_ratio)
        min_width = int(width * self.s.min_object_width_ratio)
        min_height = int(height * self.s.min_object_height_ratio)
        min_component_area = max(8, int(image_area * 0.00005))

        for c in candidates:
            # Clean tiny noise specks
            mask = _clean_mask(c["mask"], min_component_area)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue

            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            area = int(np.count_nonzero(mask))
            box_width = x1 - x0
            box_height = y1 - y0

            # Reject tiny fragments below minimum area or dimension ratios
            if area < min_area:
                log.debug("Rejecting '%s': area %d < min_area %d", c["label"], area, min_area)
                continue
            if box_width < min_width and box_height < min_height:
                log.debug("Rejecting '%s': dimensions (%dx%d) too small", c["label"], box_width, box_height)
                continue

            c["mask"] = mask
            c["area_ratio"] = round(area / image_area, 4)
            valid.append(c)

        if not valid:
            return []

        # Sort descending by score
        valid.sort(key=lambda x: x["score"], reverse=True)

        # Duplicate & containment suppression
        kept: list[dict] = []
        for cand in valid:
            c_mask = cand["mask"]
            c_area = np.count_nonzero(c_mask)
            is_duplicate = False

            for parent in kept:
                p_mask = parent["mask"]
                p_area = np.count_nonzero(p_mask)

                # Pairwise IoU check
                iou = _mask_iou(c_mask, p_mask)
                if iou >= self.s.duplicate_mask_iou:
                    log.info("Suppressed duplicate '%s' (IoU %.2f with '%s')",
                             cand["label"], iou, parent["label"])
                    is_duplicate = True
                    break

                # Containment check: candidate almost entirely inside another larger mask
                inter = np.count_nonzero(c_mask & p_mask)
                if c_area > 0 and (inter / c_area) >= self.s.contained_mask_ratio:
                    # Only suppress if candidate is substantially smaller sub-part
                    if (c_area / p_area) < 0.60:
                        log.info("Suppressed contained sub-layer '%s' inside '%s'",
                                 cand["label"], parent["label"])
                        is_duplicate = True
                        break

            if not is_duplicate:
                kept.append(cand)

        return kept[:self.s.max_objects]

    # ------------------------------------------------------------------
    # Stage 4: Full-Canvas Preservation PNG Export
    # ------------------------------------------------------------------
    def _export_full_canvas(self, original: Image.Image, mask: np.ndarray) -> bytes:
        """Export full-canvas PNG: object stays in exact original position; outside is transparent.

        Canvas dimensions, coordinate system, and scale are 100% identical to the source image.
        """
        width, height = original.size

        # Resize mask to exact original image dimensions
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        alpha = mask_image.resize((width, height), Image.Resampling.BILINEAR)

        # Preserve the original canvas and exact pixel positions
        layer = original.convert("RGBA").copy()
        layer.putalpha(alpha)

        buffer = io.BytesIO()
        layer.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    # ------------------------------------------------------------------
    # Main Extraction Orchestration
    # ------------------------------------------------------------------
    def extract(
        self,
        data: bytes,
        return_diagnostics: bool = False,
        **kwargs
    ) -> list[Cutout] | tuple[list[Cutout], dict]:
        """Universal Extraction Engine:
        1. GPT Semantic Asset Decomposition Planner
        2. SAM 3 Text-Prompted Concept Segmentation
        3. Mask Validation, Fragment & Duplicate Filtering
        4. Full-Canvas PNG Generation
        """
        # Load original image and working image
        original = ImageOps.exif_transpose(
            Image.open(io.BytesIO(data))
        ).convert("RGBA")
        width, height = original.size

        work = original.convert("RGB")
        max_side = self.s.max_working_side
        scale = 1.0
        if max(work.size) > max_side:
            scale = max_side / max(work.size)
            work = work.resize(
                (round(work.width * scale), round(work.height * scale)),
                Image.Resampling.LANCZOS
            )

        # 1. GPT semantic inventory
        raw_inventory = self._semantic_analysis(data)
        inventory = self._validate_inventory(raw_inventory)

        if not inventory:
            diagnostics = {
                "status": "NeedsReview",
                "reason": "No valid independent objects identified by visual decomposition planner",
                "total_extracted": 0,
                "layers": []
            }
            log.warning("Decomposition planner returned no valid independent objects.")
            return ([], diagnostics) if return_diagnostics else []

        # 2. SAM 3 concept-by-concept segmentation
        candidates = self._segment_concepts(work, inventory)

        # 3. Mask validation, fragment cleanup & duplicate suppression
        w_work, h_work = work.size
        validated = self._validate_candidates(candidates, w_work, h_work)

        # 4. Full-canvas PNG generation
        cutouts: list[Cutout] = []
        for cand in validated:
            work_mask = cand["mask"]
            # Upscale mask to full original resolution for precise bounding box computation
            mask_pil = Image.fromarray(work_mask.astype(np.uint8) * 255, mode="L").resize(
                (width, height), Image.Resampling.BILINEAR
            )
            full_mask = np.array(mask_pil) > 127

            ys, xs = np.where(full_mask)
            if len(xs) == 0:
                continue

            # Bounding box measured in original image coordinates
            bbox = (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1
            )

            # Export full-canvas PNG (same dimensions, same coordinates, outside transparent)
            png_bytes = self._export_full_canvas(original, full_mask)

            status = "confirmed" if cand["score"] >= self.s.sam_score_threshold else "needs_review"
            cutouts.append(Cutout(
                label=cand["label"],
                score=round(cand["score"], 4),
                bbox=bbox,
                png=png_bytes,
                kind="cutout",
                confidence_status=status,
                area_ratio=cand["area_ratio"],
                category=cand["category"],
                canvas_width=width,
                canvas_height=height,
                role=cand.get("role", "standalone")
            ))

        review_needed = any(c.confidence_status == "needs_review" for c in cutouts)
        diagnostics = {
            "status": "NeedsReview" if review_needed else "Completed",
            "canvas_dimensions": {"width": width, "height": height},
            "inventory_planned": [it["name"] for it in inventory],
            "total_extracted": len(cutouts),
            "review_needed": review_needed,
            "layers": [
                {
                    "label": c.label,
                    "score": c.score,
                    "status": c.confidence_status,
                    "bbox": list(c.bbox),
                    "area_ratio": c.area_ratio,
                    "category": c.category,
                    "role": c.role,
                    "canvas": [c.canvas_width, c.canvas_height]
                }
                for c in cutouts
            ]
        }

        log.info("Extracted %d full-canvas layers (review_needed=%s)", len(cutouts), review_needed)
        return (cutouts, diagnostics) if return_diagnostics else cutouts
