import os
from dataclasses import dataclass, field


def _env(name: str, default: str | None = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var {name}")
    return val or ""


@dataclass
class Settings:
    # --- Content Hub ---
    ch_base_url: str = field(default_factory=lambda: _env("CH_BASE_URL", required=True).rstrip("/"))
    ch_client_id: str = field(default_factory=lambda: _env("CH_CLIENT_ID", required=True))
    ch_client_secret: str = field(default_factory=lambda: _env("CH_CLIENT_SECRET", required=True))
    ch_username: str = field(default_factory=lambda: _env("CH_USERNAME"))
    ch_password: str = field(default_factory=lambda: _env("CH_PASSWORD"))
    ch_grant_type: str = field(default_factory=lambda: _env("CH_GRANT_TYPE", "password"))  # password | client_credentials

    # Schema names (must match what you create in Schema)
    prop_is_derived: str = field(default_factory=lambda: _env("PROP_IS_DERIVED", "IsDerivedCutout"))
    prop_label: str = field(default_factory=lambda: _env("PROP_LABEL", "CutoutLabel"))
    prop_confidence: str = field(default_factory=lambda: _env("PROP_CONFIDENCE", "CutoutConfidence"))
    prop_bbox: str = field(default_factory=lambda: _env("PROP_BBOX", "CutoutBoundingBox"))
    prop_status: str = field(default_factory=lambda: _env("PROP_STATUS", "ObjectExtractionStatus"))
    prop_count: str = field(default_factory=lambda: _env("PROP_COUNT", "ObjectExtractionCount"))
    prop_message: str = field(default_factory=lambda: _env("PROP_MESSAGE", "ObjectExtractionMessage"))
    prop_source_hash: str = field(default_factory=lambda: _env("PROP_SOURCE_HASH", "ObjectExtractionSourceHash"))
    rel_parent: str = field(default_factory=lambda: _env("REL_PARENT", "ParentAssetToCutoutAsset"))

    cutout_marker: str = "__cutout_"  # loop guard: filenames containing this are never processed

    # --- Service Bus ---
    sb_connection: str = field(default_factory=lambda: _env("SB_CONNECTION", required=True))
    sb_queue: str = field(default_factory=lambda: _env("SB_QUEUE", "ch-object-extraction"))

    # --- Models / extraction ---
    # Vision-Language Model for dynamic object identification
    vlm_model: str = field(default_factory=lambda: _env("VLM_MODEL", "microsoft/Florence-2-base"))
    vlm_task: str = field(default_factory=lambda: _env("VLM_TASK", "<OD>"))
    use_dynamic_vlm: bool = field(
        default_factory=lambda: _env("USE_DYNAMIC_VLM", "true").lower() in ("true", "1", "yes")
    )

    # Grounding DINO (open-vocabulary object locator) and SAM (mask segmenter)
    detector_model: str = field(default_factory=lambda: _env("DETECTOR_MODEL", "IDEA-Research/grounding-dino-base"))
    sam_model: str = field(default_factory=lambda: _env("SAM_MODEL", "facebook/sam-vit-large"))

    # Optional manual fallback labels (empty by default: dynamic VLM identifies objects automatically)
    labels: list[str] = field(default_factory=lambda: [
        s.strip() for s in _env("DETECT_LABELS", "").split(",") if s.strip()
    ])

    box_threshold: float = field(
        default_factory=lambda: float(_env("BOX_THRESHOLD", "0.18"))
    )
    text_threshold: float = field(
        default_factory=lambda: float(_env("TEXT_THRESHOLD", "0.18"))
    )
    nms_iou: float = field(
        default_factory=lambda: float(_env("NMS_IOU", "0.55"))
    )
    min_area_ratio: float = field(
        default_factory=lambda: float(_env("MIN_AREA_RATIO", "0.003"))
    )
    max_area_ratio: float = field(
        default_factory=lambda: float(_env("MAX_AREA_RATIO", "0.95"))
    )
    containment_threshold: float = field(
        default_factory=lambda: float(_env("CONTAINMENT_THRESHOLD", "0.88"))
    )
    confidence_review_threshold: float = field(
        default_factory=lambda: float(_env("CONFIDENCE_REVIEW_THRESHOLD", "0.40"))
    )
    max_objects: int = field(
        default_factory=lambda: int(_env("MAX_OBJECTS", "20"))
    )
    min_objects_to_split: int = field(
        default_factory=lambda: int(_env("MIN_OBJECTS_TO_SPLIT", "1"))
    )

    # Amodal Completion / Occlusion Inpainting (LaMa)
    enable_amodal_completion: bool = field(
        default_factory=lambda: _env("ENABLE_AMODAL_COMPLETION", "true").lower() in ("true", "1", "yes")
    )
    max_occlusion_ratio_to_complete: float = field(
        default_factory=lambda: float(_env("MAX_OCCLUSION_RATIO_TO_COMPLETE", "0.45"))
    )
    skip_severely_occluded: bool = field(
        default_factory=lambda: _env("SKIP_SEVERELY_OCCLUDED", "false").lower() in ("true", "1", "yes")
    )

    # Backward compatibility flag
    use_multi_pass: bool = field(
        default_factory=lambda: _env("USE_MULTI_PASS", "true").lower() in ("true", "1", "yes")
    )
    padding_ratio: float = field(default_factory=lambda: float(_env("PADDING_RATIO", "0.04")))
    max_working_side: int = field(default_factory=lambda: int(_env("MAX_WORKING_SIDE", "4096")))
    output_mode: str = field(default_factory=lambda: _env("OUTPUT_MODE", "cutout"))  # cutout | crop | both

    rendition_wait_seconds: int = field(default_factory=lambda: int(_env("RENDITION_WAIT_SECONDS", "240")))
