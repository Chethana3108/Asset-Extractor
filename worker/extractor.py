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
    ) -> list[Cutout]:
        box_th = box_threshold if box_threshold is not None else self.s.box_threshold
        text_th = text_threshold if text_threshold is not None else self.s.text_threshold
        min_area = min_area_ratio if min_area_ratio is not None else self.s.min_area_ratio
        max_area = max_area_ratio if max_area_ratio is not None else self.s.max_area_ratio
        iou = nms_iou if nms_iou is not None else self.s.nms_iou
        max_obj = max_objects if max_objects is not None else self.s.max_objects
        min_split = min_objects_to_split if min_objects_to_split is not None else self.s.min_objects_to_split

        prompt = _format_prompt(labels if labels is not None else self.s.labels)

        full, work, scale = self._load(data)
        boxes, scores, det_labels = self._detect(work, prompt, box_th, text_th)
        dets = self._filter(boxes, scores, det_labels, *work.size, min_area, max_area, iou, max_obj)
        log.info("Detected %d objects after filtering (threshold=%.2f, min_area=%.4f)", len(dets), box_th, min_area)
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
