"""V1 — Helmet non-compliance (A + B + E)."""
from __future__ import annotations

from ..types import Detection, Track
from .geometry import crop, make_violation, object_overlaps_vehicle


def check_helmet(frame, track: Track, persons: list[Detection], classifier,
                 cfg: dict, camera_id: str, timestamp: float = 0.0):
    """Yields HELMET_VIOLATION for riders without a helmet (3-frame persistence)."""
    rider_iou = cfg.get("rider_iou", 0.30)
    head_frac = cfg.get("head_frac", 0.30)
    conf_th = cfg.get("conf", 0.80)
    persist = cfg.get("persist_frames", 3)

    riders = [p for p in persons if p.cls_name == "person"
              and object_overlaps_vehicle(track.bbox, p.bbox, rider_iou)]

    violated = False
    best_conf = 0.0
    for rider in riders:
        x1, y1, x2, y2 = rider.bbox
        head_h = (y2 - y1) * head_frac
        head_crop = crop(frame, (x1, y1, x2, y1 + head_h))
        res = classifier.classify(head_crop)
        if res.class_name == "no_helmet" and res.confidence > conf_th:
            violated = True
            best_conf = max(best_conf, res.confidence)

    if violated:
        track.helmet_violation_frames += 1
        if track.helmet_violation_frames >= persist:
            yield make_violation("HELMET_VIOLATION", track, camera_id, best_conf,
                                 timestamp, consistent_frames=track.helmet_violation_frames)
    else:
        track.helmet_violation_frames = 0
