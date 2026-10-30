"""Universal AI Object Extraction Engine.

Architecture:
1. Florence-2: Open-world vision-language model understands the uploaded image and
   dynamically identifies visible objects (no industry-specific hardcoded labels).
2. Grounding DINO: Open-vocabulary text-grounded detector locates objects accurately.
3. Filtering & Validation: Area filtering, sub-part/containment suppression, duplicate removal
   (NMS), and confidence validation (confirmed vs needs_review).
4. Segment Anything Model (SAM): Computes pixel-precise masks and outputs transparent RGBA PNG cutouts.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

# --- Transformers compatibility patches for Florence-2 in transformers >= 4.45 / 5.x ---
import transformers

_orig_config_getattribute = transformers.PretrainedConfig.__getattribute__


def _safe_config_getattribute(self, key):
    if key == "forced_bos_token_id" and key not in self.__dict__:
        return None
    return _orig_config_getattribute(self, key)


transformers.PretrainedConfig.__getattribute__ = _safe_config_getattribute

if not hasattr(transformers.PreTrainedTokenizerBase, "additional_special_tokens"):
    transformers.PreTrainedTokenizerBase.additional_special_tokens = property(
        lambda self: self.special_tokens_map.get("additional_special_tokens", [])
    )

_orig_module_getattr = torch.nn.Module.__getattr__


def _safe_module_getattr(self, name: str):
    if name == "_supports_sdpa":
        return False
    return _orig_module_getattr(self, name)


torch.nn.Module.__getattr__ = _safe_module_getattr
# ----------------------------------------------------------------------------------------

try:
    from torchvision.ops import nms
except (ImportError, ModuleNotFoundError):
    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """Pure PyTorch fallback when torchvision is not installed."""
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
        _, order = scores.sort(0, descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            w = (xx2 - xx1).clamp(min=0.0)
            h = (yy2 - yy1).clamp(min=0.0)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
            inds = (ovr <= iou_threshold).nonzero(as_tuple=False).squeeze(1)
            order = order[inds + 1]
        return torch.tensor(keep, dtype=torch.int64, device=boxes.device)

from transformers import (AutoModelForCausalLM,
                          AutoModelForZeroShotObjectDetection, AutoProcessor,
                          SamModel, SamProcessor)

from config import Settings

log = logging.getLogger(__name__)

# Non-object terms to discard from dynamic image understanding
NON_OBJECT_WORDS = {
    "text", "letter", "letters", "writing", "word", "words", "font", "logo", "symbol",
    "shadow", "reflection", "reflections", "glare", "border", "background", "surface", "pattern",
    "floor", "wall", "ceiling", "table", "counter", "desk", "shelf", "wood", "tile", "white background",
    "image", "photo", "picture", "scene", "display", "view", "area", "side", "color", "colour",
    "the", "a", "an", "of", "and", "in", "on", "at", "is", "are", "with", "for", "to",
    # Scene-level / category terms that cause DINO to draw one giant box
    "items", "products", "supplies", "stuff", "things", "objects", "collection", "set", "group",
    "assortment", "variety", "arrangement", "contents", "accessories", "equipment", "materials",
    "medical supplies", "healthcare", "pharmaceutical",
}


@dataclass
class Cutout:
    label: str
    score: float
    bbox: tuple[int, int, int, int]           # x0, y0, x1, y1 in ORIGINAL image pixels
    png: bytes                                 # RGBA cutout (or RGB crop in 'crop' mode)
    kind: str                                  # 'cutout' | 'crop'
    confidence_status: str = "confirmed"       # 'confirmed' | 'needs_review'
    area_ratio: float = 0.0                    # Fraction of image area occupied
    is_completed: bool = False                 # True if occluded notches were completed with inpainting
    occlusion_ratio: float = 0.0               # Ratio of occluded surface completed


def _box_area(box: list[float] | tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(b1: list[float], b2: list[float]) -> float:
    ix0 = max(b1[0], b2[0])
    iy0 = max(b1[1], b2[1])
    ix1 = min(b1[2], b2[2])
    iy1 = min(b1[3], b2[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _format_prompt(labels: list[str]) -> str:
    cleaned = [l.strip().lower() for l in labels if l.strip()]
    return " . ".join(cleaned) + " ." if cleaned else ""


class ObjectExtractor:
    """Universal 3-Model Extraction Pipeline: Florence-2 + Grounding DINO + SAM."""

    def __init__(self, s: Settings):
        self.s = s
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        log.info("Initializing Universal ObjectExtractor on %s (dtype=%s)", self.device, self.dtype)

        # 1. Florence-2 Vision-Language Model (Dynamic Image Understanding)
        log.info("Loading Vision-Language Model: %s", s.vlm_model)
        self.vlm_proc = AutoProcessor.from_pretrained(s.vlm_model, trust_remote_code=True)
        self.vlm_model = AutoModelForCausalLM.from_pretrained(
            s.vlm_model,
            torch_dtype=self.dtype,
            trust_remote_code=True
        ).to(self.device).eval()

        # Tie language model embeddings if needed for modern transformers compatibility
        self._tie_vlm_weights()

        # 2. Grounding DINO Detector (Open-Vocabulary Localization)
        log.info("Loading Detector Model: %s", s.detector_model)
        self.det_proc = AutoProcessor.from_pretrained(s.detector_model)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(s.detector_model).to(self.device).eval()

        # 3. SAM Model (Segment Anything for Precise Masks)
        log.info("Loading SAM Model: %s", s.sam_model)
        self.sam_proc = SamProcessor.from_pretrained(s.sam_model)
        self.sam = SamModel.from_pretrained(s.sam_model, torch_dtype=self.dtype).to(self.device).eval()

        # 4. LaMa Inpainting Model (Amodal Occlusion Completion)
        self.lama = None
        if s.enable_amodal_completion:
            try:
                log.info("Loading LaMa inpainting model for amodal object completion...")
                from simple_lama_inpainting import SimpleLama
                self.lama = SimpleLama()
                log.info("LaMa model loaded successfully.")
            except Exception as e:
                log.warning("Could not load SimpleLama: %s", e)

    def _tie_vlm_weights(self) -> None:
        """Ensure tied embeddings in Florence-2 point to shared weights across encoder/decoder/lm_head."""
        if hasattr(self.vlm_model, "language_model"):
            lm = self.vlm_model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "shared"):
                shared = lm.model.shared.weight
                if hasattr(lm.model, "encoder") and hasattr(lm.model.encoder, "embed_tokens"):
                    lm.model.encoder.embed_tokens.weight = shared
                if hasattr(lm.model, "decoder") and hasattr(lm.model.decoder, "embed_tokens"):
                    lm.model.decoder.embed_tokens.weight = shared
                if hasattr(lm, "lm_head"):
                    lm.lm_head.weight = shared
                log.info("Florence-2 embeddings tied successfully.")

    def _load(self, data: bytes) -> tuple[Image.Image, Image.Image, float]:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))  # respect camera rotation
        has_alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
        rgba = img.convert("RGBA") if has_alpha else img.convert("RGB")
        rgb = rgba.convert("RGB")
        scale = 1.0
        m = max(rgb.size)
        if m > self.s.max_working_side:
            scale = self.s.max_working_side / m
            rgb = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.LANCZOS)
        return rgba, rgb, scale

    # ------------------------------------------------------------------
    # Step 1: Dynamic Image Understanding with Florence-2
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _understand_image(self, img: Image.Image) -> tuple[list[str], list[dict]]:
        """Understands the image content dynamically and returns unique object labels and raw detections.

        Uses a multi-pass strategy:
        1. <OD> for primary object detection labels.
        2. <DENSE_REGION_CAPTION> always runs to catch objects that <OD> misses.
        Labels are cleaned: max 3 words, deduplicated, no repeated words, no scene-level terms.
        """
        w, h = img.size
        # Florence-2 DaViT backbone expects square 768x768 RGB inputs
        square_img = img.convert("RGB").resize((768, 768), Image.BILINEAR)

        discovered_labels: list[str] = []
        vlm_detections: list[dict] = []
        seen: set[str] = set()

        def _add_label(lbl: str) -> bool:
            """Clean, validate, and deduplicate a label. Returns True if new."""
            cleaned = lbl.strip().lower().strip(".")
            if not cleaned or cleaned in NON_OBJECT_WORDS:
                return False
            if len(cleaned) < 2 or not any(c.isalnum() for c in cleaned):
                return False
            # Remove repeated words (e.g. "medical supplies medical supplies" -> "medical supplies")
            words = cleaned.split()
            deduped = []
            for word in words:
                if word not in NON_OBJECT_WORDS and (not deduped or word != deduped[-1]):
                    deduped.append(word)
            cleaned = " ".join(deduped)
            if not cleaned:
                return False
            # Max 3 words per label — longer phrases are scene descriptions, not objects
            if len(cleaned.split()) > 3:
                return False
            if cleaned in seen:
                return False
            seen.add(cleaned)
            discovered_labels.append(cleaned)
            return True

        # --- Pass 1: <OD> object detection ---
        task_prompt = self.s.vlm_task
        inputs = self.vlm_proc(text=task_prompt, images=square_img, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        try:
            generated_ids = self.vlm_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=1,
                do_sample=False,
                use_cache=False,
            )
            generated_text = self.vlm_proc.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = self.vlm_proc.post_process_generation(generated_text, task=task_prompt, image_size=(w, h))

            raw_results = parsed.get(task_prompt, {})
            raw_labels = raw_results.get("labels", [])
            raw_boxes = raw_results.get("bboxes", [])

            for idx, lbl in enumerate(raw_labels):
                cleaned = lbl.strip().lower()
                _add_label(cleaned)
                if idx < len(raw_boxes):
                    vlm_detections.append({
                        "label": cleaned,
                        "bbox": [round(float(coord), 1) for coord in raw_boxes[idx]]
                    })
        except Exception as e:
            log.warning("Florence-2 <OD> understanding failed: %s", e)

        # --- Pass 2: <DENSE_REGION_CAPTION> always runs to catch missed objects ---
        try:
            caption_task = "<DENSE_REGION_CAPTION>"
            c_inputs = self.vlm_proc(text=caption_task, images=square_img, return_tensors="pt").to(self.device)
            c_inputs["pixel_values"] = c_inputs["pixel_values"].to(self.dtype)
            c_ids = self.vlm_model.generate(
                input_ids=c_inputs["input_ids"],
                pixel_values=c_inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=1,
                do_sample=False,
                use_cache=False,
            )
            c_text = self.vlm_proc.batch_decode(c_ids, skip_special_tokens=False)[0]
            c_parsed = self.vlm_proc.post_process_generation(c_text, task=caption_task, image_size=(w, h))
            dense_results = c_parsed.get(caption_task, {})
            for lbl in dense_results.get("labels", []):
                c_lbl = lbl.strip().lower().rstrip(".").strip()
                _add_label(c_lbl)
                # Extract tail noun phrases for better grounding
                words = [w for w in c_lbl.split() if w not in NON_OBJECT_WORDS]
                if len(words) >= 2:
                    _add_label(" ".join(words[-2:]))
                if len(words) >= 3:
                    _add_label(" ".join(words[-3:]))
                if words:
                    _add_label(words[-1])
        except Exception as e:
            log.warning("Florence-2 dense region caption pass failed: %s", e)

        log.info("Florence-2 dynamically identified %d labels: %s", len(discovered_labels), discovered_labels)
        return discovered_labels, vlm_detections

    # ------------------------------------------------------------------
    # Step 2: Grounding DINO Object Localization
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _detect(self, img: Image.Image, prompt: str, box_threshold: float, text_threshold: float) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        if not prompt.strip():
            return torch.empty((0, 4)), torch.empty((0,)), []

        inputs = self.det_proc(images=img, text=prompt, return_tensors="pt").to(self.device)
        out = self.det(**inputs)
        kwargs = dict(text_threshold=text_threshold, target_sizes=[img.size[::-1]])
        try:  # transformers >= 4.51 renamed box_threshold -> threshold
            res = self.det_proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=box_threshold, **kwargs)[0]
        except TypeError:
            res = self.det_proc.post_process_grounded_object_detection(
                out, inputs.input_ids, box_threshold=box_threshold, **kwargs)[0]
        labels = res.get("text_labels") or res.get("labels") or []
        return res["boxes"].float().cpu(), res["scores"].float().cpu(), [str(l) for l in labels]

    # ------------------------------------------------------------------
    # Step 3: Filtering, Containment Suppression & Duplicate Removal
    # ------------------------------------------------------------------
    def _filter_and_validate(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: list[str],
        w: int,
        h: int,
        min_area_ratio: float,
        max_area_ratio: float,
        containment_threshold: float,
        nms_iou: float,
        confidence_review_threshold: float,
        max_objects: int,
    ) -> tuple[list[dict], dict]:
        """Filters noise, suppresses internal decorative sub-parts, removes duplicates, and classifies confidence."""
        if len(boxes) == 0:
            return [], {"raw_count": 0, "area_filtered": 0, "containment_suppressed": 0, "kept_count": 0}

        img_area = float(w * h)
        raw_candidates = []
        for i in range(len(boxes)):
            box = boxes[i].tolist()
            score = float(scores[i])
            label = labels[i] if i < len(labels) and labels[i] else "object"
            bw = max(0.0, box[2] - box[0])
            bh = max(0.0, box[3] - box[1])
            area = bw * bh
            area_ratio = area / img_area

            # Area filtering
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue

            # Aspect ratio filtering: eliminate extreme borders or thin line shadows
            aspect = bw / (bh + 1e-6)
            if aspect > 14.0 or aspect < (1.0 / 14.0):
                continue

            raw_candidates.append({
                "box": box,
                "score": score,
                "label": label,
                "area_ratio": area_ratio,
                "area": area,
            })

        # Sort descending by area and score for containment analysis
        # Larger parent objects come first
        raw_candidates.sort(key=lambda c: (c["area"], c["score"]), reverse=True)

        # Containment filter: suppress sub-parts completely or heavily inside a larger parent object
        # (e.g. text/icon on a phone screen, decorative sticker on a cream jar, internal details)
        # BUT preserve distinct smaller objects that overlap a larger one (e.g. bottle in front of box)
        kept_after_containment = []
        containment_suppressed = 0

        for candidate in raw_candidates:
            c_box = candidate["box"]
            c_area = candidate["area"]
            c_label = candidate["label"]
            is_subpart = False

            for parent in kept_after_containment:
                p_box = parent["box"]
                p_area = parent["area"]
                p_label = parent["label"]
                if p_area <= c_area:
                    continue

                inter = _intersection_area(c_box, p_box)
                containment_ratio = inter / (c_area + 1e-7)
                area_ratio_to_parent = c_area / p_area

                # Only suppress if:
                # 1. Candidate is almost entirely inside parent (high containment)
                # 2. Candidate is very small relative to parent (decorative sub-part)
                # 3. Candidate has the SAME label as parent (duplicate region of same object)
                #    OR is extremely contained and tiny
                same_label = (c_label == p_label)
                if containment_ratio >= containment_threshold:
                    if same_label and area_ratio_to_parent < 0.65:
                        is_subpart = True
                    elif area_ratio_to_parent < 0.15:
                        # Very tiny sub-part deeply contained → likely decoration/detail
                        is_subpart = True

                if is_subpart:
                    containment_suppressed += 1
                    log.debug("Suppressed sub-part '%s' inside parent '%s'", candidate["label"], parent["label"])
                    break

            if not is_subpart:
                kept_after_containment.append(candidate)

        if not kept_after_containment:
            return [], {
                "raw_count": len(boxes),
                "area_filtered": len(boxes) - len(raw_candidates),
                "containment_suppressed": containment_suppressed,
                "kept_count": 0
            }

        # Class-agnostic NMS for duplicate removal
        cand_boxes = torch.tensor([c["box"] for c in kept_after_containment], dtype=torch.float32)
        cand_scores = torch.tensor([c["score"] for c in kept_after_containment], dtype=torch.float32)
        keep_indices = nms(cand_boxes, cand_scores, nms_iou)[:max_objects]

        final_detections = []
        for idx in keep_indices:
            c = kept_after_containment[idx.item()]
            score = c["score"]
            status = "confirmed" if score >= confidence_review_threshold else "needs_review"
            final_detections.append({
                "box": c["box"],
                "score": score,
                "label": c["label"],
                "confidence_status": status,
                "area_ratio": round(c["area_ratio"], 4),
            })

        filter_diag = {
            "raw_count": len(boxes),
            "area_filtered": len(boxes) - len(raw_candidates),
            "containment_suppressed": containment_suppressed,
            "kept_count": len(final_detections),
        }
        return final_detections, filter_diag

    # ------------------------------------------------------------------
    # Step 4: SAM Mask Generation & Cutout Production
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _masks(self, img: Image.Image, boxes: list[list[float]]) -> np.ndarray:
        inputs = self.sam_proc(img, input_boxes=[boxes], return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        # Use multimask_output=True so SAM returns 3 masks per box; pick the best one
        out = self.sam(**inputs, multimask_output=True)
        pred_masks = out.pred_masks.float().cpu()    # (1, N, 3, H, W)
        iou_scores = out.iou_scores.float().cpu()    # (1, N, 3)
        # Select the highest-IoU mask for each detection
        best_idx = iou_scores[0].argmax(dim=-1)      # (N,)
        selected = []
        for k in range(pred_masks.shape[1]):
            selected.append(pred_masks[0, k, best_idx[k]])  # (H, W)
        # Reshape to (1, N, 1, H, W) — batch=1, N queries, 1 mask each
        selected_masks = torch.stack(selected).unsqueeze(0).unsqueeze(2)
        masks = self.sam_proc.image_processor.post_process_masks(
            selected_masks, inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        return masks[0][:, 0].numpy().astype(bool)  # (N, H, W)

    # ------------------------------------------------------------------
    # Step 4b: Mask Refinement — smooth edges, anti-alias, erode fringe
    # ------------------------------------------------------------------
    def _refine_mask_to_full(self, mask_bool: np.ndarray, W: int, H: int) -> np.ndarray:
        """Upscale a boolean mask to full resolution with smooth, anti-aliased edges.

        Returns a float32 alpha map [0..255] instead of hard binary, so cutout
        edges blend naturally rather than showing staircase artifacts.

        Steps:
        1. Upscale with BILINEAR (not NEAREST) for smooth sub-pixel edges.
        2. Gentle morphological erosion (1-2px) to prevent background fringe.
        3. Gaussian blur on the edge band only for natural anti-aliasing.
        """
        # Step 1: Bilinear upscale from working resolution to full resolution
        mask_uint8 = (mask_bool.astype(np.uint8) * 255)
        mask_pil = Image.fromarray(mask_uint8, mode="L").resize((W, H), Image.BILINEAR)
        alpha = np.array(mask_pil, dtype=np.float32)  # 0..255 with smooth gradients

        # Step 2: Very gentle erosion — just 1px to trim background fringe
        binary = (alpha > 127).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(binary, kernel, iterations=1)

        # Step 3: Gaussian blur on the eroded mask for soft anti-aliased edges
        smoothed = cv2.GaussianBlur(eroded.astype(np.float32) * 255.0, (3, 3), 0)

        # Combine: keep the interior solid (255), only soften at edges
        result = np.where(eroded > 0, 255.0, smoothed)
        return np.clip(result, 0, 255).astype(np.float32)

    # ------------------------------------------------------------------
    # Main Universal Extraction Method
    # ------------------------------------------------------------------
    def extract(
        self,
        data: bytes,
        labels: list[str] | None = None,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
        min_area_ratio: float | None = None,
        max_area_ratio: float | None = None,
        containment_threshold: float | None = None,
        confidence_review_threshold: float | None = None,
        nms_iou: float | None = None,
        max_objects: int | None = None,
        min_objects_to_split: int | None = None,
        use_dynamic_vlm: bool | None = None,
        use_multi_pass: bool | None = None,  # backwards compatibility alias
        return_diagnostics: bool = False,
    ) -> list[Cutout] | tuple[list[Cutout], dict]:
        """Universal extraction: Understand -> Locate -> Segment -> Output."""
        # Resolve parameters with settings defaults
        dynamic = use_dynamic_vlm if use_dynamic_vlm is not None else (
            use_multi_pass if use_multi_pass is not None else self.s.use_dynamic_vlm
        )
        box_th = box_threshold if box_threshold is not None else self.s.box_threshold
        text_th = text_threshold if text_threshold is not None else self.s.text_threshold
        min_area = min_area_ratio if min_area_ratio is not None else self.s.min_area_ratio
        max_area = max_area_ratio if max_area_ratio is not None else self.s.max_area_ratio
        contain_th = containment_threshold if containment_threshold is not None else self.s.containment_threshold
        conf_review_th = confidence_review_threshold if confidence_review_threshold is not None else self.s.confidence_review_threshold
        iou = nms_iou if nms_iou is not None else self.s.nms_iou
        max_obj = max_objects if max_objects is not None else self.s.max_objects
        min_split = min_objects_to_split if min_objects_to_split is not None else self.s.min_objects_to_split

        full, work, scale = self._load(data)
        W, H = full.size
        w, h = work.size

        # 1. Determine detection prompt
        dynamic_labels_found: list[str] = []
        vlm_proposals: list[dict] = []

        if labels and len(labels) > 0:
            active_labels = [l.strip().lower() for l in labels if l.strip()]
            prompt = _format_prompt(active_labels)
            log.info("Using user-supplied labels: %s", active_labels)
        elif dynamic:
            dynamic_labels_found, vlm_proposals = self._understand_image(work)
            active_labels = dynamic_labels_found if dynamic_labels_found else self.s.labels
            prompt = _format_prompt(active_labels)
            log.info("Universal dynamic prompt: '%s'", prompt)
        else:
            active_labels = self.s.labels
            prompt = _format_prompt(active_labels)

        # 2. Grounding DINO detection
        boxes, scores, det_labels = self._detect(work, prompt, box_th, text_th)

        # 3. Filtering, containment suppression, duplicate removal, confidence classification
        dets, filter_diag = self._filter_and_validate(
            boxes=boxes,
            scores=scores,
            labels=det_labels,
            w=w,
            h=h,
            min_area_ratio=min_area,
            max_area_ratio=max_area,
            containment_threshold=contain_th,
            nms_iou=iou,
            confidence_review_threshold=conf_review_th,
            max_objects=max_obj,
        )

        review_needed = any(d["confidence_status"] == "needs_review" for d in dets)

        diagnostics = {
            "dynamic_labels_identified": dynamic_labels_found,
            "vlm_raw_boxes_count": len(vlm_proposals),
            "dino_raw_boxes_count": filter_diag.get("raw_count", 0),
            "area_filtered_count": filter_diag.get("area_filtered", 0),
            "containment_suppressed_count": filter_diag.get("containment_suppressed", 0),
            "total_extracted": len(dets),
            "review_needed": review_needed,
            "detected_objects": [
                {
                    "label": d["label"],
                    "score": round(d["score"], 4),
                    "confidence_status": d["confidence_status"],
                    "area_ratio": d["area_ratio"],
                    "box": [round(coord / scale, 1) for coord in d["box"]],
                }
                for d in dets
            ]
        }

        if len(dets) < min_split:
            log.info("Detected %d objects, less than min_objects_to_split (%d)", len(dets), min_split)
            return ([], diagnostics) if return_diagnostics else []

        # 4. SAM Masks and RGBA cutouts (with clean edge refinement)
        masks = self._masks(work, [d["box"] for d in dets]) if self.s.output_mode != "crop" else None
        results: list[Cutout] = []

        # Precompute full-resolution refined masks for all detections
        full_masks = []
        if masks is not None:
            for k in range(len(dets)):
                full_masks.append(self._refine_mask_to_full(masks[k], W, H))

        for i, d in enumerate(dets):
            box = d["box"]
            score = d["score"]
            label = d["label"]
            status = d["confidence_status"]
            area_ratio = d["area_ratio"]

            x0, y0, x1, y1 = (v / scale for v in box)
            pad_x = (x1 - x0) * self.s.padding_ratio
            pad_y = (y1 - y0) * self.s.padding_ratio
            bx = (
                max(0, int(x0 - pad_x)),
                max(0, int(y0 - pad_y)),
                min(W, int(x1 + pad_x)),
                min(H, int(y1 + pad_y))
            )

            if self.s.output_mode in ("crop", "both"):
                results.append(Cutout(
                    label=label,
                    score=score,
                    bbox=bx,
                    png=_png(full.convert("RGB").crop(bx)),
                    kind="crop",
                    confidence_status=status,
                    area_ratio=area_ratio
                ))

            if masks is not None:
                rx0, ry0, rx1, ry1 = bx
                # full_masks[i] is a float32 alpha [0..255] for smooth edges
                alpha_full = full_masks[i]
                alpha_crop = alpha_full[ry0:ry1, rx0:rx1]
                m_crop_bool = alpha_crop > 127  # binary for area checks

                if not np.any(m_crop_bool):
                    log.warning("SAM mask empty for %s (score %.2f, box %s), falling back to crop", label, score, bx)
                    results.append(Cutout(
                        label=label,
                        score=score,
                        bbox=bx,
                        png=_png(full.convert("RGB").crop(bx)),
                        kind="crop",
                        confidence_status=status,
                        area_ratio=area_ratio
                    ))
                    continue

                # Detect genuine inter-object occlusion ONLY (no convex hull nonsense)
                occ_crop = np.zeros_like(m_crop_bool, dtype=bool)
                for j in range(len(dets)):
                    if j == i:
                        continue
                    other_alpha = full_masks[j][ry0:ry1, rx0:rx1]
                    occ_crop |= (other_alpha > 127)

                # Only count occlusion where another object overlaps THIS object's mask
                genuine_overlap = occ_crop & m_crop_bool
                occ_pixels = np.count_nonzero(genuine_overlap)
                vis_pixels = np.count_nonzero(m_crop_bool)
                occ_ratio = occ_pixels / (vis_pixels + 1e-7)

                # Check if severely occluded
                if self.s.skip_severely_occluded and occ_ratio > self.s.max_occlusion_ratio_to_complete:
                    log.info("Skipping severely occluded object %s (occlusion ratio %.2f)", label, occ_ratio)
                    continue

                crop_rgb = full.convert("RGB").crop(bx)
                is_completed = False

                # Only inpaint if genuine inter-object overlap is detected (not shape concavities)
                if (self.s.enable_amodal_completion and self.lama is not None
                        and occ_ratio >= 0.02 and occ_pixels > 100):
                    try:
                        # Create inpainting mask only for the genuinely occluded overlap region
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        inpaint_mask = cv2.dilate(genuine_overlap.astype(np.uint8) * 255, kernel)
                        inpainted_rgb = self.lama(crop_rgb, Image.fromarray(inpaint_mask))
                        inpainted_rgb = inpainted_rgb.crop((0, 0, crop_rgb.width, crop_rgb.height))

                        # Alpha = original smooth mask + filled overlap zone
                        completed_alpha = alpha_crop.copy()
                        completed_alpha[genuine_overlap] = 255
                        rgba = inpainted_rgb.convert("RGBA")
                        rgba.putalpha(Image.fromarray(completed_alpha.astype(np.uint8)))
                        is_completed = True
                        log.info("Amodal inpainting completed overlap for %s (occlusion ratio %.2f)", label, occ_ratio)
                    except Exception as e:
                        log.warning("Amodal inpainting failed for %s, using standard cutout: %s", label, e)
                        rgba = full.convert("RGBA").crop(bx)
                        rgba.putalpha(Image.fromarray(alpha_crop.astype(np.uint8)))
                else:
                    rgba = full.convert("RGBA").crop(bx)
                    rgba.putalpha(Image.fromarray(alpha_crop.astype(np.uint8)))

                results.append(Cutout(
                    label=label,
                    score=score,
                    bbox=bx,
                    png=_png(rgba),
                    kind="cutout",
                    confidence_status=status,
                    area_ratio=area_ratio,
                    is_completed=is_completed,
                    occlusion_ratio=round(occ_ratio, 4)
                ))

        log.info("Extracted %d cutout assets (review_needed=%s)", len(results), review_needed)
        return (results, diagnostics) if return_diagnostics else results


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
