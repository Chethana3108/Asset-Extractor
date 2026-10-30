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
    detector_model: str = field(default_factory=lambda: _env("DETECTOR_MODEL", "IDEA-Research/grounding-dino-base"))
    sam_model: str = field(default_factory=lambda: _env("SAM_MODEL", "facebook/sam-vit-large"))
    # Grounding DINO is open-vocabulary: these are the "things" it looks for.
    labels: list[str] = field(default_factory=lambda: [
        s.strip() for s in _env(
            "DETECT_LABELS",
            # 'person' deliberately excluded: cutting real people out of source shots
            # creates likeness/consent issues downstream. Add it only if rights are cleared.
            "product,bottle,box,package,tube,jar,can,pouch,sachet,blister pack,tablet,pill,capsule,device,logo,packet,container",
        ).split(",") if s.strip()
    ])
    box_threshold: float = field(default_factory=lambda: float(_env("BOX_THRESHOLD", "0.23")))
    text_threshold: float = field(default_factory=lambda: float(_env("TEXT_THRESHOLD", "0.20")))
    nms_iou: float = field(default_factory=lambda: float(_env("NMS_IOU", "0.55")))
    min_area_ratio: float = field(default_factory=lambda: float(_env("MIN_AREA_RATIO", "0.0005")))   # keep small items like pills
    max_area_ratio: float = field(default_factory=lambda: float(_env("MAX_AREA_RATIO", "0.95")))   # drop "whole image"
    max_objects: int = field(default_factory=lambda: int(_env("MAX_OBJECTS", "30")))
    min_objects_to_split: int = field(default_factory=lambda: int(_env("MIN_OBJECTS_TO_SPLIT", "1")))
    padding_ratio: float = field(default_factory=lambda: float(_env("PADDING_RATIO", "0.04")))
    max_working_side: int = field(default_factory=lambda: int(_env("MAX_WORKING_SIDE", "4096")))
    output_mode: str = field(default_factory=lambda: _env("OUTPUT_MODE", "cutout"))  # cutout | crop | both

    rendition_wait_seconds: int = field(default_factory=lambda: int(_env("RENDITION_WAIT_SECONDS", "240")))
