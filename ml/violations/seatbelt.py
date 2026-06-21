"""V2 — Seatbelt non-compliance (A + D)."""
from __future__ import annotations

from ..types import Track
from .geometry import crop, make_violation


def check_seatbelt(frame, track: Track, seatbelt_model, cfg: dict,
                   camera_id: str, timestamp: float = 0.0):
    """Crop windshield region, run seatbelt detector, 3-frame persistence."""
    conf_th = cfg.get("conf", 0.80)
    persist = cfg.get("persist_frames", 3)
    top = cfg.get("windshield_top", 0.40)
    side = cfg.get("windshield_side", 0.15)

    x1, y1, x2, y2 = track.bbox
    windshield = crop(frame, (x1, y1, x2, y2))
    if windshield is None:
        track.seatbelt_violation_frames = 0
        return

    dets = seatbelt_model.detect(windshield, conf=conf_th) if seatbelt_model.available else []
    violating = [d for d in dets if d.cls_name == "no_seatbelt" and d.confidence > conf_th]

    if violating:
        track.seatbelt_violation_frames += 1
        best = max(d.confidence for d in violating)
        if track.seatbelt_violation_frames >= persist:
            yield make_violation("SEATBELT_VIOLATION", track, camera_id, best,
                                 timestamp, consistent_frames=track.seatbelt_violation_frames)
    else:
        track.seatbelt_violation_frames = 0
