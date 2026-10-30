"""Service Bus consumer: Content Hub trigger -> Azure Service Bus action -> this worker.

Message body produced by the Content Hub 'Azure Service Bus' action looks like:
  {"saveEntityMessage": {"EventType": "EntityCreated", "TargetDefinition": "M.Asset",
                         "TargetId": 40562, "ChangeSet": {...}, ...}, "context": {...}}
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re

from azure.servicebus import AutoLockRenewer, ServiceBusClient

from config import Settings
from content_hub import ContentHubClient
from extractor import ObjectExtractor

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("worker")

IMAGE_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def _target_id(body: dict) -> int | None:
    msg = body.get("saveEntityMessage") or body
    tid = msg.get("TargetId") or msg.get("targetId")
    return int(tid) if tid else None


def _prop(ent: dict, name: str):
    return (ent.get("properties") or {}).get(name)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "object"


class Processor:
    def __init__(self, s: Settings):
        self.s = s
        self.ch = ContentHubClient(s)
        self.ex = ObjectExtractor(s)

    def _status(self, aid: int, status: str, msg: str = "", **extra) -> None:
        props = {self.s.prop_status: status, self.s.prop_message: msg[:900]} | extra
        self.ch.update_entity(aid, properties=props)

    def handle(self, asset_id: int) -> None:
        s = self.s
        ent = self.ch.get_entity(asset_id)
        file_name = _prop(ent, "FileName") or ""

        # ---- loop guards: never process our own outputs ----
        if _prop(ent, s.prop_is_derived) or s.cutout_marker in file_name:
            log.info("Asset %s is a derived cutout, skipping", asset_id)
            return
        if os.path.splitext(file_name)[1].lower() not in IMAGE_EXT:
            log.info("Asset %s (%s) is not a raster image, skipping", asset_id, file_name)
            return

        ent, original_url = self.ch.wait_for_original(asset_id, s.rendition_wait_seconds)
        data = self.ch.download(original_url)
        digest = hashlib.sha256(data).hexdigest()

        # ---- idempotency: Service Bus is at-least-once; triggers can fire twice ----
        if _prop(ent, s.prop_source_hash) == digest and _prop(ent, s.prop_status) in ("Completed", "Processing"):
            log.info("Asset %s already processed for this file version", asset_id)
            return

        self._status(asset_id, "Processing", **{s.prop_source_hash: digest})
        try:
            cutouts = self.ex.extract(data)
            if not cutouts:
                self._status(asset_id, "NoObjects",
                             f"Fewer than {s.min_objects_to_split} distinct objects detected")
                return

            stem = os.path.splitext(file_name)[0]
            created: list[int] = []
            for n, c in enumerate(cutouts, 1):
                name = f"{stem}{s.cutout_marker}{n:02d}_{_slug(c.label)}_{c.kind}.png"
                new_id = self.ch.upload_new_asset(name, c.png)
                self.ch.update_entity(
                    new_id,
                    properties={
                        s.prop_is_derived: True,
                        s.prop_label: c.label,
                        s.prop_confidence: round(c.score, 4),
                        s.prop_bbox: json.dumps({"x0": c.bbox[0], "y0": c.bbox[1],
                                                 "x1": c.bbox[2], "y1": c.bbox[3], "kind": c.kind}),
                    },
                    # OneToMany self-relation: child side holds a single 'parent'
                    relations={s.rel_parent: {"parent": {"href": self.ch.entity_href(asset_id)}}},
                )
                created.append(new_id)
                log.info("Asset %s -> cutout %s (%s %.2f)", asset_id, new_id, c.label, c.score)

            self._status(asset_id, "Completed", f"Created {len(created)} assets: {created}",
                         **{s.prop_count: len(created)})
        except Exception as e:  # mark the parent so authors see failures in the UI
            log.exception("Asset %s failed", asset_id)
            self._status(asset_id, "Failed", f"{type(e).__name__}: {e}")
            raise


def run() -> None:
    s = Settings()
    proc = Processor(s)
    renewer = AutoLockRenewer(max_lock_renewal_duration=900)  # model + uploads can exceed the 60s default lock
    with ServiceBusClient.from_connection_string(s.sb_connection) as sb:
        with sb.get_queue_receiver(s.sb_queue, max_wait_time=30) as rx:
            log.info("Listening on queue %s", s.sb_queue)
            while True:
                for msg in rx.receive_messages(max_message_count=1, max_wait_time=30):
                    renewer.register(rx, msg)
                    try:
                        body = json.loads(str(msg))
                        tid = _target_id(body)
                        if tid is None:
                            log.warning("No TargetId in message, dead-lettering: %s", str(msg)[:300])
                            rx.dead_letter_message(msg, reason="NoTargetId")
                            continue
                        proc.handle(tid)
                        rx.complete_message(msg)
                    except Exception:
                        # abandon -> redelivery; after MaxDeliveryCount it goes to the DLQ
                        rx.abandon_message(msg)


if __name__ == "__main__":
    run()
