"""V3 — Triple riding (A + B). Pure logic — IoU person count on two-wheeler."""
from __future__ import annotations

from ..types import Detection, Track
from .geometry import make_violation, object_overlaps_vehicle


def check_triple_riding(track: Track, persons: list[Detection], cfg: dict,
                        camera_id: str, timestamp: float = 0.0):
    rider_iou = cfg.get("rider_iou", 0.30)
    min_riders = cfg.get("min_riders", 3)
    persist = cfg.get("persist_frames", 5)

    rider_count = sum(
        1 for p in persons
        if p.cls_name == "person" and object_overlaps_vehicle(track.bbox, p.bbox, rider_iou)
    )

    if rider_count >= min_riders:
        track.triple_riding_frames += 1
        if track.triple_riding_frames >= persist:
            # confidence scales with rider count over the threshold
            conf = min(0.75 + 0.05 * (rider_count - min_riders), 0.95)
            yield make_violation("TRIPLE_RIDING", track, camera_id, conf, timestamp,
                                 rider_count=rider_count,
                                 consistent_frames=track.triple_riding_frames)
    else:
        track.triple_riding_frames = 0
