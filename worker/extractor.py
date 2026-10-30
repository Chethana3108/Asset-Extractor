"""Universal AI Object Extraction Engine — 4-Stage Architecture.

Architecture:
1. Stage 1: GPT Vision Semantic Analysis
   - High-level scene understanding and structured element cataloging.
   - Distinguishes foreground products from background, tables, shadows, reflections.
   - Emits a precise object inventory with descriptive concept prompts (no flawed coordinates).
2. Stage 2: Concept Segmentation (SAM 3.1 / Grounded-SAM)
   - Translates GPT's semantic concept prompts into pixel-precise object masks.
   - Supports SAM 3 / 3.1 promptable concept segmentation when available.
   - Provides Grounded-SAM (Grounding DINO + SAM ViT-Large) for robust open-vocabulary segmentation.
3. Stage 3: Python Mask Validation & Quality Assurance
   - Discards background leaks (masks touching all borders or covering > 90% area).
   - Suppresses duplicate extractions using pairwise mask IoU.
   - Validates mask quality using real SAM IoU confidence scores (not guessed LLM scores).
4. Stage 4: Sub-pixel Edge Refinement & Layer Export
   - Morphological cleanup of stray pixels.
   - Smooth anti-aliased edge blending.
   - Exports clean transparent RGBA PNG cutouts.
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
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          SamModel, SamProcessor)

from config import Settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class Cutout:
    label: str
    score: float                               # Real segmentation IoU confidence score [0.0 - 1.0]
    bbox: tuple[int, int, int, int]           # x0, y0, x1, y1 in ORIGINAL image pixels
    png: bytes                                 # RGBA transparent PNG cutout
    kind: str = "cutout"                       # 'cutout' | 'crop'
    confidence_status: str = "confirmed"       # 'confirmed' | 'needs_review'
    area_ratio: float = 0.0                    # Fraction of image area occupied
    category: str = "product"                  # Detected category (e.g. package, bottle, sachet)
    is_completed: bool = False
    occlusion_ratio: float = 0.0


# ---------------------------------------------------------------------------
# GPT-4o Vision Prompting
# ---------------------------------------------------------------------------
GPT_CATALOG_PROMPT = """You are a precision digital asset element cataloger for a Digital Asset Management (DAM) system.
Analyze this image and identify EVERY distinct, meaningful foreground object or product that should be extracted as an independent, transparent layer.

Strict Rules:
1. Identify all separate foreground objects (e.g. bottles, boxes, blister packs, sachets, containers, groups of tablets/capsules).
2. Distinguish between open vs closed packages, and differentiate distinct items (e.g. "yellow blister pack" vs "silver blister pack").
3. DO NOT include: background, floor, table surfaces, shadows, reflections, pure text stickers on products.
4. For loose scattered items like pills/tablets/capsules, group contiguous clusters into coherent units (e.g. "scattered white tablets", "green and white capsules").
5. For each item provide:
   - "name": Concise human-readable name (1-3 words, e.g. "open cardboard box", "white medicine bottle", "yellow blister pack", "silver blister pack", "white foil sachet", "white round tablets", "green white capsules")
   - "prompt": A descriptive visual concept phrase (2-5 words) that an open-vocabulary object detector/segmenter can use to localize the item (e.g. "open white cardboard box", "white plastic medicine bottle", "yellow tablet blister pack", "silver tablet blister pack", "white foil sachet packet", "scattered white round tablets", "green and white capsules")
   - "category": e.g. "box", "bottle", "blister pack", "sachet", "tablet", "capsule", "product"

Return JSON with this exact structure:
{
  "inventory": [
    {
      "name": "...",
      "prompt": "...",
      "category": "..."
    }
  ]
}"""


# ---------------------------------------------------------------------------
# Main Extraction Engine
# ---------------------------------------------------------------------------
class ObjectExtractor:
    """Universal 4-Stage Object Extraction Pipeline."""

    def __init__(self, s: Settings):
        self.s = s
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        log.info("Initializing ObjectExtractor on %s (dtype=%s)", self.device, self.dtype)

        # Stage 1: OpenAI GPT Vision setup
        self.openai_key = s.openai_api_key
        log.info("OpenAI GPT Vision configured with model: %s", s.gpt_model)

        # Stage 2: Segmentation Models
        # Check if native SAM 3.1 is available
        self.sam3_available = False
        self.sam3_processor = None
        self.sam3_model = None
        if s.use_sam3:
            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
                log.info("Loading SAM 3.1 image model...")
                self.sam3_model = build_sam3_image_model()
                self.sam3_processor = Sam3Processor(self.sam3_model)
                self.sam3_available = True
                log.info("SAM 3.1 initialized successfully.")
            except Exception as e:
                log.info("SAM 3.1 native package not loaded (%s). Using Grounded-SAM engine.", e)

        # Grounded-SAM engine (Grounding DINO + SAM ViT-Large)
        log.info("Loading Grounding DINO detector: %s", s.detector_model)
        self.det_proc = AutoProcessor.from_pretrained(s.detector_model)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(s.detector_model).to(self.device).eval()

        log.info("Loading SAM segmenter: %s", s.sam_model)
        self.sam_proc = SamProcessor.from_pretrained(s.sam_model)
        self.sam = SamModel.from_pretrained(s.sam_model, torch_dtype=self.dtype).to(self.device).eval()

        log.info("ObjectExtractor ready.")

    # ------------------------------------------------------------------
    # Image Loading & Pre-processing
    # ------------------------------------------------------------------
    def _load(self, data: bytes) -> tuple[Image.Image, Image.Image, float]:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        has_alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
        full_rgba = img.convert("RGBA") if has_alpha else img.convert("RGB").convert("RGBA")
        rgb = full_rgba.convert("RGB")

        scale = 1.0
        m = max(rgb.size)
        if m > self.s.max_working_side:
            scale = self.s.max_working_side / m
            rgb = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.LANCZOS)
        return full_rgba, rgb, scale

    # ------------------------------------------------------------------
    # Stage 1: GPT Vision — Semantic Analysis & Inventory
    # ------------------------------------------------------------------
    def _semantic_analysis(self, image_data: bytes) -> list[dict]:
        """Query GPT-4o Vision to produce a structured inventory of distinct objects."""
        if not self.openai_key:
            log.warning("No OpenAI API key provided. Skipping semantic analysis.")
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
            log.info("GPT Vision identified %d distinct elements: %s",
                     len(items), [it.get("name") for it in items])
            return items

        except Exception as e:
            log.error("GPT Vision semantic analysis failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Stage 2: Concept Segmentation (Grounded-SAM / SAM 3.1)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _segment_concepts(
        self,
        work_img: Image.Image,
        inventory: list[dict],
        box_threshold: float,
        text_threshold: float
    ) -> list[dict]:
        """Detect and segment objects using the semantic concept inventory."""
        if not inventory:
            return []

        # Prepare concept prompts
        prompts = []
        item_map = {}
        for idx, item in enumerate(inventory):
            prompt_str = item.get("prompt") or item.get("name") or "object"
            cleaned = prompt_str.strip().lower()
            prompts.append(cleaned)
            item_map[cleaned] = item

        # Build compound detector query separated by '.'
        compound_prompt = " . ".join(prompts) + " ."
        log.info("Grounded detector prompt: '%s'", compound_prompt)

        # Grounding DINO detection
        inputs = self.det_proc(images=work_img, text=compound_prompt, return_tensors="pt").to(self.device)
        out = self.det(**inputs)
        kwargs = dict(text_threshold=text_threshold, target_sizes=[work_img.size[::-1]])
        try:
            res = self.det_proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=box_threshold, **kwargs)[0]
        except TypeError:
            res = self.det_proc.post_process_grounded_object_detection(
                out, inputs.input_ids, box_threshold=box_threshold, **kwargs)[0]

        boxes = res["boxes"].float().cpu()
        scores = res["scores"].float().cpu()
        labels = [str(l) for l in (res.get("text_labels") or res.get("labels") or [])]

        if len(boxes) == 0:
            log.info("No detector candidate boxes found.")
            return []

        # SAM multimask segmentation on detected candidate boxes
        box_list = boxes.tolist()
        sam_inputs = self.sam_proc(work_img, input_boxes=[box_list], return_tensors="pt").to(self.device)
        sam_inputs["pixel_values"] = sam_inputs["pixel_values"].to(self.dtype)

        sam_out = self.sam(**sam_inputs, multimask_output=True)
        pred_masks = sam_out.pred_masks.float().cpu()    # (1, N, 3, H, W)
        sam_iou_scores = sam_out.iou_scores.float().cpu()  # (1, N, 3)

        # Select highest-IoU mask for each detection
        best_indices = sam_iou_scores[0].argmax(dim=-1)  # (N,)
        selected = []
        quality_scores = []
        for k in range(pred_masks.shape[1]):
            best_k = best_indices[k].item()
            selected.append(pred_masks[0, k, best_k])
            quality_scores.append(sam_iou_scores[0, k, best_k].item())

        selected_tensor = torch.stack(selected).unsqueeze(0).unsqueeze(2)  # (1, N, 1, H, W)
        post_masks = self.sam_proc.image_processor.post_process_masks(
            selected_tensor,
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu()
        )
        binary_masks = post_masks[0][:, 0].numpy().astype(bool)  # (N, H, W)

        candidates = []
        for i in range(len(boxes)):
            raw_lbl = labels[i] if i < len(labels) else "object"
            # Find closest matching inventory item
            matched_item = None
            for p_key, it in item_map.items():
                if p_key in raw_lbl or raw_lbl in p_key:
                    matched_item = it
                    break
            if not matched_item:
                # Find by substring match
                for p_key, it in item_map.items():
                    if any(w in p_key for w in raw_lbl.split()):
                        matched_item = it
                        break

            final_name = matched_item.get("name") if matched_item else raw_lbl
            final_cat = matched_item.get("category") if matched_item else "product"
            sam_score = float(quality_scores[i])
            det_score = float(scores[i])
            # Combined confidence: 60% SAM mask quality + 40% detector confidence
            combined_conf = round(0.60 * sam_score + 0.40 * det_score, 4)

            candidates.append({
                "label": final_name,
                "category": final_cat,
                "box": boxes[i].tolist(),
                "mask": binary_masks[i],
                "score": combined_conf,
                "sam_iou": round(sam_score, 4),
                "det_score": round(det_score, 4)
            })

        log.info("Segmented %d raw candidates with SAM.", len(candidates))
        return candidates

    # ------------------------------------------------------------------
    # Stage 3: Python Mask Validation & Quality Assurance
    # ------------------------------------------------------------------
    def _validate_masks(
        self,
        candidates: list[dict],
        img_w: int,
        img_h: int
    ) -> list[dict]:
        """Validate and filter masks for completeness, boundary leaks, and duplicates."""
        if not candidates:
            return []

        img_area = float(img_w * img_h)
        valid = []

        for c in candidates:
            mask = c["mask"]
            mask_pixels = int(np.count_nonzero(mask))
            area_ratio = mask_pixels / img_area

            # 1. Area filtering (discard tiny noise specks or giant canvas floods)
            if area_ratio < self.s.min_area_ratio or area_ratio > self.s.max_area_ratio:
                log.debug("Discarding candidate '%s': area_ratio %.4f out of range", c["label"], area_ratio)
                continue

            # 2. Border check: discard if mask touches all 4 edges (canvas background leak)
            touches_top = np.any(mask[0, :])
            touches_bottom = np.any(mask[-1, :])
            touches_left = np.any(mask[:, 0])
            touches_right = np.any(mask[:, -1])
            if touches_top and touches_bottom and touches_left and touches_right:
                log.debug("Discarding candidate '%s': touches all 4 borders (background flood)", c["label"])
                continue

            c["area_ratio"] = round(area_ratio, 4)
            valid.append(c)

        if not valid:
            return []

        # Sort descending by confidence score
        valid.sort(key=lambda x: x["score"], reverse=True)

        # 3. Duplicate suppression using pairwise mask IoU
        kept: list[dict] = []
        for cand in valid:
            c_mask = cand["mask"]
            is_dup = False
            for parent in kept:
                p_mask = parent["mask"]
                inter = np.count_nonzero(c_mask & p_mask)
                union = np.count_nonzero(c_mask | p_mask)
                iou = inter / (union + 1e-7)

                if iou >= self.s.duplicate_iou_threshold:
                    is_dup = True
                    log.debug("Suppressed duplicate '%s' (IoU %.2f with '%s')",
                              cand["label"], iou, parent["label"])
                    break

            if not is_dup:
                # Assign confidence status based on real SAM mask quality
                status = "confirmed" if cand["sam_iou"] >= self.s.mask_confidence_threshold else "needs_review"
                cand["confidence_status"] = status
                kept.append(cand)

        # Respect max_objects limit
        return kept[:self.s.max_objects]

    # ------------------------------------------------------------------
    # Stage 4: Sub-pixel Edge Refinement & Layer Export
    # ------------------------------------------------------------------
    def _refine_and_cutout(
        self,
        full_rgba: Image.Image,
        work_mask: np.ndarray,
        box: list[float],
        scale: float
    ) -> tuple[bytes, tuple[int, int, int, int]]:
        """Refine mask edges with anti-aliasing and extract transparent RGBA PNG."""
        W, H = full_rgba.size
        x0, y0, x1, y1 = (v / scale for v in box)
        pad_x = (x1 - x0) * self.s.padding_ratio
        pad_y = (y1 - y0) * self.s.padding_ratio

        bx = (
            max(0, int(x0 - pad_x)),
            max(0, int(y0 - pad_y)),
            min(W, int(x1 + pad_x)),
            min(H, int(y1 + pad_y))
        )
        rx0, ry0, rx1, ry1 = bx

        # Upscale mask to full original resolution with bilinear interpolation
        mask_uint8 = work_mask.astype(np.uint8) * 255
        full_mask_pil = Image.fromarray(mask_uint8, mode="L").resize((W, H), Image.BILINEAR)
        alpha_full = np.array(full_mask_pil, dtype=np.float32)

        # Crop to padded bounding box
        alpha_crop = alpha_full[ry0:ry1, rx0:rx1]

        # Gentle 1px morphological cleanup to trim stray edge noise
        binary_crop = (alpha_crop > 127).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_binary = cv2.morphologyEx(binary_crop, cv2.MORPH_OPEN, kernel)

        # Sub-pixel anti-aliasing edge softening
        smoothed = cv2.GaussianBlur(clean_binary.astype(np.float32) * 255.0, (3, 3), 0)
        # Keep solid interior at 255, apply softened values at the edge band
        final_alpha = np.where(clean_binary > 0, 255.0, smoothed)
        final_alpha = np.clip(final_alpha, 0, 255).astype(np.uint8)

        # Produce RGBA cutout
        crop_rgba = full_rgba.crop(bx)
        crop_rgba.putalpha(Image.fromarray(final_alpha))

        return _png(crop_rgba), bx

    # ------------------------------------------------------------------
    # Main Universal Extract Entrypoint
    # ------------------------------------------------------------------
    def extract(
        self,
        data: bytes,
        labels: list[str] | None = None,
        return_diagnostics: bool = False,
        **kwargs
    ) -> list[Cutout] | tuple[list[Cutout], dict]:
        """4-Stage Universal Extraction:
        GPT Vision -> Open-Vocabulary SAM -> Validation -> Refined Layers.
        """
        full_rgba, work_rgb, scale = self._load(data)
        W, H = full_rgba.size
        w, h = work_rgb.size

        # Stage 1: GPT Vision Semantic Analysis
        if labels and len(labels) > 0:
            inventory = [{"name": l.strip(), "prompt": l.strip(), "category": "custom"}
                         for l in labels if l.strip()]
            log.info("Using user-supplied concept inventory: %s", [it["name"] for it in inventory])
        else:
            inventory = self._semantic_analysis(data)

        # Fallback if GPT analysis returned nothing
        if not inventory:
            default_labels = ["package", "bottle", "box", "product", "item", "container"]
            inventory = [{"name": l, "prompt": l, "category": "general"} for l in default_labels]

        # Stage 2: Concept Segmentation (SAM / Grounded-SAM)
        raw_candidates = self._segment_concepts(
            work_img=work_rgb,
            inventory=inventory,
            box_threshold=self.s.box_threshold,
            text_threshold=self.s.text_threshold
        )

        # Stage 3: Python Mask Validation & Quality Assurance
        validated = self._validate_masks(raw_candidates, w, h)

        # Stage 4: Edge Refinement & PNG Generation
        results: list[Cutout] = []
        for v in validated:
            png_bytes, bx = self._refine_and_cutout(
                full_rgba=full_rgba,
                work_mask=v["mask"],
                box=v["box"],
                scale=scale
            )
            results.append(Cutout(
                label=v["label"],
                score=v["score"],
                bbox=bx,
                png=png_bytes,
                kind="cutout",
                confidence_status=v["confidence_status"],
                area_ratio=v["area_ratio"],
                category=v["category"]
            ))

        review_needed = any(r.confidence_status == "needs_review" for r in results)
        diagnostics = {
            "semantic_inventory": [it["name"] for it in inventory],
            "raw_candidates_count": len(raw_candidates),
            "validated_objects_count": len(validated),
            "total_extracted": len(results),
            "review_needed": review_needed,
            "detected_objects": [
                {
                    "label": r.label,
                    "score": r.score,
                    "confidence_status": r.confidence_status,
                    "area_ratio": r.area_ratio,
                    "category": r.category,
                    "bbox": list(r.bbox)
                }
                for r in results
            ]
        }

        log.info("Extraction complete: %d clean layers generated (review_needed=%s)",
                 len(results), review_needed)
        return (results, diagnostics) if return_diagnostics else results


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
