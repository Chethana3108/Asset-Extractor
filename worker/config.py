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
    ch_grant_type: str = field(default_factory=lambda: _env("CH_GRANT_TYPE", "password"))

    # Schema names
    prop_is_derived: str = field(default_factory=lambda: _env("PROP_IS_DERIVED", "IsDerivedCutout"))
    prop_label: str = field(default_factory=lambda: _env("PROP_LABEL", "CutoutLabel"))
    prop_confidence: str = field(default_factory=lambda: _env("PROP_CONFIDENCE", "CutoutConfidence"))
    prop_bbox: str = field(default_factory=lambda: _env("PROP_BBOX", "CutoutBoundingBox"))
    prop_canvas_width: str = field(default_factory=lambda: _env("PROP_CANVAS_WIDTH", "CutoutCanvasWidth"))
    prop_canvas_height: str = field(default_factory=lambda: _env("PROP_CANVAS_HEIGHT", "CutoutCanvasHeight"))
    prop_area_ratio: str = field(default_factory=lambda: _env("PROP_AREA_RATIO", "CutoutAreaRatio"))
    prop_status: str = field(default_factory=lambda: _env("PROP_STATUS", "ObjectExtractionStatus"))
    prop_count: str = field(default_factory=lambda: _env("PROP_COUNT", "ObjectExtractionCount"))
    prop_message: str = field(default_factory=lambda: _env("PROP_MESSAGE", "ObjectExtractionMessage"))
    prop_source_hash: str = field(default_factory=lambda: _env("PROP_SOURCE_HASH", "ObjectExtractionSourceHash"))
    rel_parent: str = field(default_factory=lambda: _env("REL_PARENT", "ParentAssetToCutoutAsset"))

    cutout_marker: str = "__cutout_"

    # --- Service Bus ---
    sb_connection: str = field(default_factory=lambda: _env("SB_CONNECTION", required=True))
    sb_queue: str = field(default_factory=lambda: _env("SB_QUEUE", "ch-object-extraction"))

    # --- GPT Vision (Semantic Analysis & Asset Decomposition) ---
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    gpt_model: str = field(default_factory=lambda: _env("GPT_MODEL", "gpt-4o"))

    # --- SAM 3 Segmentation ---
    sam_model: str = field(
        default_factory=lambda: _env("SAM_MODEL", "facebook/sam3")
    )
    sam_score_threshold: float = field(
        default_factory=lambda: float(_env("SAM_SCORE_THRESHOLD", "0.55"))
    )
    sam_mask_threshold: float = field(
        default_factory=lambda: float(_env("SAM_MASK_THRESHOLD", "0.50"))
    )

    # --- Object Size & Fragment Filters ---
    min_object_area_ratio: float = field(
        default_factory=lambda: float(_env("MIN_OBJECT_AREA_RATIO", "0.003"))
    )
    min_object_width_ratio: float = field(
        default_factory=lambda: float(_env("MIN_OBJECT_WIDTH_RATIO", "0.025"))
    )
    min_object_height_ratio: float = field(
        default_factory=lambda: float(_env("MIN_OBJECT_HEIGHT_RATIO", "0.025"))
    )

    # --- Duplicate & Containment Controls ---
    duplicate_mask_iou: float = field(
        default_factory=lambda: float(_env("DUPLICATE_MASK_IOU", "0.75"))
    )
    contained_mask_ratio: float = field(
        default_factory=lambda: float(_env("CONTAINED_MASK_RATIO", "0.90"))
    )
    max_objects: int = field(
        default_factory=lambda: int(_env("MAX_OBJECTS", "20"))
    )
    min_objects_to_split: int = field(
        default_factory=lambda: int(_env("MIN_OBJECTS_TO_SPLIT", "1"))
    )

    # --- Image Processing ---
    max_working_side: int = field(
        default_factory=lambda: int(_env("MAX_WORKING_SIDE", "4096"))
    )
    rendition_wait_seconds: int = field(
        default_factory=lambda: int(_env("RENDITION_WAIT_SECONDS", "240"))
    )
