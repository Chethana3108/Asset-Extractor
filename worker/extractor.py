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
from PIL import Image, ImageEnhance, ImageOps

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
    use_contrast_enhancement: bool = False
    use_tiles: bool = False


DEFAULT_CATEGORY_RULES: list[CategoryRule] = [
    CategoryRule(
        name="box",
        prompt="a medicine carton . a medicine box . a cardboard box . an open box . a packaging box .",
        box_threshold=0.28,
        text_threshold=0.20,
        min_area_ratio=0.005,
    ),
    CategoryRule(
        name="bottle",
        prompt="a medicine bottle . a white plastic bottle . a pill bottle with cap . a medicine bottle with white cap .",
        box_threshold=0.28,
        text_threshold=0.20,
        min_area_ratio=0.005,
    ),
    CategoryRule(
        name="sachet",
        prompt="a white sachet . a sealed paper packet . a rectangular medicine sachet .",
        box_threshold=0.25,
        text_threshold=0.20,
        min_area_ratio=0.002,
        use_contrast_enhancement=True,
    ),
    CategoryRule(
        name="blister_pack",
        prompt="a blister pack with pills . a tablet strip . a medicine blister pack .",
        box_threshold=0.28,
        text_threshold=0.20,
        min_area_ratio=0.005,
    ),
    CategoryRule(
        name="capsule",
        prompt="a green capsule . a two-tone capsule . a medicine capsule . a capsule .",
        box_threshold=0.25,
        text_threshold=0.20,
        min_area_ratio=0.00005,
        use_tiles=True,
    ),
    CategoryRule(
        name="tablet",
        prompt="a round white tablet . a white pill . a round tablet . a medicine tablet . a pill .",
        box_threshold=0.25,
        text_threshold=0.20,
        min_area_ratio=0.00005,
        use_tiles=True,
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


def _enhance_contrast(img: Image.Image) -> Image.Image:
    """Enhance edges and contrast for low-contrast items like white sachets on white backgrounds."""
    ac = ImageOps.autocontrast(img.convert("RGB"), cutoff=2)
    return ImageEnhance.Contrast(ac).enhance(1.4)


def _get_overlapping_tiles(img: Image.Image, overlap: float = 0.20) -> list[tuple[Image.Image, int, int]]:
    """Generate 2x2 overlapping image tiles for high-resolution small object detection."""
    w, h = img.size
    tw, th = int(w * (0.5 + overlap / 2)), int(h * (0.5 + overlap / 2))
    offsets = [
        (0, 0),
        (w - tw, 0),
        (0, h - th),
        (w - tw, h - th),
    ]
    tiles = []
    for ox, oy in offsets:
        box = (ox, oy, ox + tw, oy + th)
        tile = img.crop(box)
        tiles.append((tile, ox, oy))
    return tiles


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

    def _multi_pass_detect(self, img: Image.Image, max_objects: int, nms_iou: float) -> tuple[list[tuple[list[float], float, str]], dict]:
        w, h = img.size
        category_detections: list[tuple[list[float], float, str]] = []
        diagnostics = {"raw_by_category": {}, "kept_by_category": {}, "all_raw_boxes": []}

        # Precompute enhanced contrast image for low-contrast items like white sachets
        contrast_img = _enhance_contrast(img)
        # Precompute overlapping tiles for small items like tablets and capsules
        tiles = _get_overlapping_tiles(img, overlap=0.20)

        for rule in DEFAULT_CATEGORY_RULES:
            raw_boxes_list = []
            raw_scores_list = []
            raw_labels_list = []

            # 1. Full image pass
            eval_img = contrast_img if rule.use_contrast_enhancement else img
            b_full, s_full, l_full = self._detect(eval_img, rule.prompt, rule.box_threshold, rule.text_threshold)
            if len(b_full) > 0:
                raw_boxes_list.append(b_full)
                raw_scores_list.append(s_full)
                raw_labels_list.extend(l_full)

            # 2. Overlapping tile passes for small items (tablets, capsules)
            if rule.use_tiles:
                for tile_img, ox, oy in tiles:
                    tb, ts, tl = self._detect(tile_img, rule.prompt, rule.box_threshold, rule.text_threshold)
                    if len(tb) > 0:
                        shifted_b = tb.clone()
                        shifted_b[:, 0] += ox
                        shifted_b[:, 1] += oy
                        shifted_b[:, 2] += ox
                        shifted_b[:, 3] += oy
                        raw_boxes_list.append(shifted_b)
                        raw_scores_list.append(ts)
                        raw_labels_list.extend(tl)

            if not raw_boxes_list:
                diagnostics["raw_by_category"][rule.name] = 0
                diagnostics["kept_by_category"][rule.name] = 0
                continue

            all_boxes = torch.cat(raw_boxes_list, dim=0)
            all_scores = torch.cat(raw_scores_list, dim=0)
            diagnostics["raw_by_category"][rule.name] = len(all_boxes)
            log.info("Category '%s' raw candidates: %d", rule.name, len(all_boxes))

            # Per-category NMS
            filtered = self._filter(
                all_boxes, all_scores, raw_labels_list, w, h,
                min_area_ratio=rule.min_area_ratio,
                max_area_ratio=rule.max_area_ratio,
                nms_iou=0.55,
                max_objects=max_objects
            )
            diagnostics["kept_by_category"][rule.name] = len(filtered)
            for box, score, _ in filtered:
                category_detections.append((box, score, rule.name))
                diagnostics["all_raw_boxes"].append({"box": box, "score": round(float(score), 4), "category": rule.name})

        if not category_detections:
            return [], diagnostics

        # Remove sub-part detections enclosed inside a larger parent object (e.g. blister bubbles inside a blister pack, bottle cap slivers)
        final_candidates: list[tuple[list[float], float, str]] = []
        for i, (box_a, score_a, label_a) in enumerate(category_detections):
            area_a = _box_area(box_a)
            is_subpart = False
            for j, (box_b, score_b, label_b) in enumerate(category_detections):
                if i == j:
                    continue
                area_b = _box_area(box_b)
                if area_b > 1.6 * area_a:
                    inter = _intersection_area(box_a, box_b)
                    if (inter / area_a) > 0.60:
                        is_subpart = True
                        break
            if not is_subpart:
                final_candidates.append((box_a, score_a, label_a))

        if not final_candidates:
            return [], diagnostics

        # Standard NMS (IoU 0.45) to collapse duplicate boxes and tile overlaps into one best box per product
        c_boxes = torch.tensor([d[0] for d in final_candidates], dtype=torch.float32)
        c_scores = torch.tensor([d[1] for d in final_candidates], dtype=torch.float32)
        keep_idx = nms(c_boxes, c_scores, 0.45)

        selected = [final_candidates[i] for i in keep_idx][:max_objects]
        log.info("Total clean detections after deduplication: %d", len(selected))
        return selected, diagnostics

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
        return_diagnostics: bool = False,
    ) -> list[Cutout] | tuple[list[Cutout], dict]:
        multi_pass = use_multi_pass if use_multi_pass is not None else self.s.use_multi_pass
        max_obj = max_objects if max_objects is not None else self.s.max_objects
        min_split = min_objects_to_split if min_objects_to_split is not None else self.s.min_objects_to_split
        iou = nms_iou if nms_iou is not None else self.s.nms_iou

        full, work, scale = self._load(data)
        diagnostics = {}

        if multi_pass and labels is None:
            dets, diagnostics = self._multi_pass_detect(work, max_objects=max_obj, nms_iou=iou)
            log.info("Multi-pass detected %d objects", len(dets))
        else:
            box_th = box_threshold if box_threshold is not None else self.s.box_threshold
            text_th = text_threshold if text_threshold is not None else self.s.text_threshold
            min_area = min_area_ratio if min_area_ratio is not None else self.s.min_area_ratio
            max_area = max_area_ratio if max_area_ratio is not None else self.s.max_area_ratio
            prompt = _format_prompt(labels if labels is not None else self.s.labels)
            boxes, scores, det_labels = self._detect(work, prompt, box_th, text_th)
            dets = self._filter(boxes, scores, det_labels, *work.size, min_area, max_area, iou, max_obj)
            diagnostics = {
                "raw_by_category": {"single_pass": len(boxes)},
                "kept_by_category": {"single_pass": len(dets)},
                "all_raw_boxes": [{"box": d[0], "score": round(d[1], 4), "category": d[2]} for d in dets]
            }
            log.info("Single-pass detected %d objects after filtering (threshold=%.2f, min_area=%.4f)", len(dets), box_th, min_area)

        if len(dets) < min_split:
            return ([], diagnostics) if return_diagnostics else []

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
                    log.warning("SAM mask empty for %s (score %.2f, box %s), falling back to crop", label, score, bx)
                    results.append(Cutout(label, score, bx, _png(full.convert("RGB").crop(bx)), "crop"))
                    continue
                rgba = full.convert("RGBA")
                alpha = Image.fromarray(np.minimum(np.array(rgba.getchannel("A")), np.array(m)))
                rgba.putalpha(alpha)
                results.append(Cutout(label, score, bx, _png(rgba.crop(bx)), "cutout"))

        return (results, diagnostics) if return_diagnostics else results


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
