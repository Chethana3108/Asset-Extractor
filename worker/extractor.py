"""Object extraction: Grounding DINO (open-vocabulary boxes) -> NMS/filters -> SAM (pixel masks) -> RGBA cutouts.

Why not Azure AI Vision: Image Analysis 4.0 background removal/segmentation was retired in 2025 and the
service itself is deprecated (retires Sept 2028). Self-hosted open models give boxes AND clean masks,
which Gen-AI compositing needs (a rectangular crop drags the old background along).
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.ops import nms
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          SamModel, SamProcessor)

from config import Settings

log = logging.getLogger(__name__)


@dataclass
class Cutout:
    label: str
    score: float
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in ORIGINAL image pixels
    png: bytes                        # RGBA cutout (or RGB crop in 'crop' mode)
    kind: str                         # 'cutout' | 'crop'


@dataclass
class CategoryRule:
    name: str
    prompt: str
    box_threshold: float
    text_threshold: float
    min_area_ratio: float
    max_area_ratio: float = 0.95


DEFAULT_CATEGORY_RULES: list[CategoryRule] = [
    CategoryRule(
        name="box",
        prompt="a medicine carton . a medicine box . a packaging box . a box .",
        box_threshold=0.22,
        text_threshold=0.18,
        min_area_ratio=0.005,
    ),
    CategoryRule(
        name="bottle",
        prompt="a medicine bottle . a plastic bottle . a pill bottle . a bottle . a pill container .",
        box_threshold=0.22,
        text_threshold=0.18,
        min_area_ratio=0.005,
    ),
    CategoryRule(
        name="sachet",
        prompt="a sachet . a packet . a white pouch . a medicine sachet . a paper sachet . a foil packet .",
        box_threshold=0.14,
        text_threshold=0.14,
        min_area_ratio=0.001,
    ),
    CategoryRule(
        name="blister_pack",
        prompt="a blister pack . a tablet strip . a pill pack . a medicine strip . a blister foil .",
        box_threshold=0.16,
        text_threshold=0.15,
        min_area_ratio=0.002,
    ),
    CategoryRule(
        name="pill",
        prompt="a tablet . a capsule . a pill . a loose pill . a white pill . a green capsule .",
        box_threshold=0.14,
        text_threshold=0.14,
        min_area_ratio=0.00005,
    ),
]


def _box_area(box: list[float]) -> float:
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
    return " . ".join(cleaned) + " ."


class ObjectExtractor:
    def __init__(self, s: Settings):
        self.s = s
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        log.info("Loading models on %s", self.device)
        self.det_proc = AutoProcessor.from_pretrained(s.detector_model)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(s.detector_model).to(self.device).eval()
        self.sam_proc = SamProcessor.from_pretrained(s.sam_model)
        self.sam = SamModel.from_pretrained(s.sam_model, torch_dtype=dtype).to(self.device).eval()
        self.dtype = dtype
        self.prompt = _format_prompt(s.labels)

    # ------------------------------------------------------------------
    def _load(self, data: bytes) -> tuple[Image.Image, Image.Image, float]:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))  # respect camera rotation
        has_alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
        rgba = img.convert("RGBA") if has_alpha else img.convert("RGB")
        rgb = rgba.convert("RGB")
        scale = 1.0
        m = max(rgb.size)
        if m > self.s.max_working_side:  # keep GPU memory bounded; map boxes back to full res later
            scale = self.s.max_working_side / m
            rgb = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.LANCZOS)
        return rgba, rgb, scale

    @torch.inference_mode()
    def _detect(self, img: Image.Image, prompt: str, box_threshold: float, text_threshold: float) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        inputs = self.det_proc(images=img, text=prompt, return_tensors="pt").to(self.device)
        out = self.det(**inputs)
        kwargs = dict(text_threshold=text_threshold, target_sizes=[img.size[::-1]])
        try:  # transformers >= 4.51 renamed box_threshold -> threshold
            res = self.det_proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=box_threshold, **kwargs)[0]
        except TypeError:
            res = self.det_proc.post_process_grounded_object_detection(
                out, inputs.input_ids, box_threshold=box_threshold, **kwargs)[0]
        labels = res.get("text_labels") or res.get("labels")
        return res["boxes"].float().cpu(), res["scores"].float().cpu(), [str(l) for l in labels]

    def _filter(self, boxes, scores, labels, w, h, min_area_ratio: float, max_area_ratio: float, nms_iou: float, max_objects: int):
        if len(boxes) == 0:
            return []
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / float(w * h)
        keep = (area >= min_area_ratio) & (area <= max_area_ratio)
        idx = torch.nonzero(keep).flatten()
        if len(idx) == 0:
            return []
        # class-agnostic NMS: "bottle" and "product" on the same object must collapse to one
        k = nms(boxes[idx], scores[idx], nms_iou)
        idx = idx[k][: max_objects]
        return [(boxes[i].tolist(), float(scores[i]), labels[i] or "object") for i in idx]

    @torch.inference_mode()
    def _masks(self, img: Image.Image, boxes: list[list[float]]) -> np.ndarray:
        inputs = self.sam_proc(img, input_boxes=[boxes], return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        out = self.sam(**inputs, multimask_output=False)
        masks = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.float().cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        return masks[0][:, 0].numpy().astype(bool)  # (N, H, W)

    def _multi_pass_detect(self, img: Image.Image, max_objects: int, nms_iou: float) -> list[tuple[list[float], float, str]]:
        w, h = img.size
        category_detections: list[tuple[list[float], float, str]] = []

        for rule in DEFAULT_CATEGORY_RULES:
            boxes, scores, labels = self._detect(img, rule.prompt, rule.box_threshold, rule.text_threshold)
            filtered = self._filter(
                boxes, scores, labels, w, h,
                min_area_ratio=rule.min_area_ratio,
                max_area_ratio=rule.max_area_ratio,
                nms_iou=0.60,
                max_objects=max_objects
            )
            for box, score, _ in filtered:
                category_detections.append((box, score, rule.name))

        if not category_detections:
            return []

        # Filter out pills that are inside a blister pack
        blister_boxes = [d[0] for d in category_detections if d[2] == "blister_pack"]
        final_candidates: list[tuple[list[float], float, str]] = []
        for box, score, label in category_detections:
            if label == "pill" and blister_boxes:
                p_area = _box_area(box)
                inside = any((_intersection_area(box, bb) / p_area) > 0.60 for bb in blister_boxes if p_area > 0)
                if inside:
                    continue  # Keep blister pack intact, skip individual bubbles
            final_candidates.append((box, score, label))

        if not final_candidates:
            return []

        # Cross-category NMS to collapse duplicate detections of the exact same object
        c_boxes = torch.tensor([d[0] for d in final_candidates], dtype=torch.float32)
        c_scores = torch.tensor([d[1] for d in final_candidates], dtype=torch.float32)
        keep_idx = nms(c_boxes, c_scores, 0.70)

        selected = [final_candidates[i] for i in keep_idx][:max_objects]
        return selected

    # ------------------------------------------------------------------
    def extract(
        self,
        data: bytes,
        labels: list[str] | None = None,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
        min_area_ratio: float | None = None,
        max_area_ratio: float | None = None,
        nms_iou: float | None = None,
        max_objects: int | None = None,
        min_objects_to_split: int | None = None,
        use_multi_pass: bool | None = None,
    ) -> list[Cutout]:
        multi_pass = use_multi_pass if use_multi_pass is not None else self.s.use_multi_pass
        max_obj = max_objects if max_objects is not None else self.s.max_objects
        min_split = min_objects_to_split if min_objects_to_split is not None else self.s.min_objects_to_split
        iou = nms_iou if nms_iou is not None else self.s.nms_iou

        full, work, scale = self._load(data)

        if multi_pass and labels is None:
            dets = self._multi_pass_detect(work, max_objects=max_obj, nms_iou=iou)
            log.info("Multi-pass detected %d objects", len(dets))
        else:
            box_th = box_threshold if box_threshold is not None else self.s.box_threshold
            text_th = text_threshold if text_threshold is not None else self.s.text_threshold
            min_area = min_area_ratio if min_area_ratio is not None else self.s.min_area_ratio
            max_area = max_area_ratio if max_area_ratio is not None else self.s.max_area_ratio
            prompt = _format_prompt(labels if labels is not None else self.s.labels)
            boxes, scores, det_labels = self._detect(work, prompt, box_th, text_th)
            dets = self._filter(boxes, scores, det_labels, *work.size, min_area, max_area, iou, max_obj)
            log.info("Single-pass detected %d objects after filtering (threshold=%.2f, min_area=%.4f)", len(dets), box_th, min_area)

        if len(dets) < min_split:
            return []

        masks = self._masks(work, [d[0] for d in dets]) if self.s.output_mode != "crop" else None
        W, H = full.size
        results: list[Cutout] = []
        for i, (box, score, label) in enumerate(dets):
            x0, y0, x1, y1 = (v / scale for v in box)
            pad_x, pad_y = (x1 - x0) * self.s.padding_ratio, (y1 - y0) * self.s.padding_ratio
            bx = (max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y)),
                  min(W, int(x1 + pad_x)), min(H, int(y1 + pad_y)))

            if self.s.output_mode in ("crop", "both"):
                results.append(Cutout(label, score, bx, _png(full.convert("RGB").crop(bx)), "crop"))

            if masks is not None:
                m = Image.fromarray(masks[i].astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
                if m.getbbox() is None:
                    continue
                rgba = full.convert("RGBA")
                alpha = Image.fromarray(np.minimum(np.array(rgba.getchannel("A")), np.array(m)))
                rgba.putalpha(alpha)
                results.append(Cutout(label, score, bx, _png(rgba.crop(bx)), "cutout"))
        return results


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
