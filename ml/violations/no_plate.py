"""V8 — No number plate (A + C). Plate absent for N consecutive frames.

Fixed Issues (June 20, 2026):
1. IoU matching fails for motorcycles (tiny plate vs large vehicle bbox).
   → Now uses containment ratio (plate center inside vehicle bbox) instead
     of pure IoU.
2. 15 frames (0.5s) was too aggressive → raised to 45 frames (~1.5s).
3. Hardcoded 0.80 confidence → now scales with how many frames plate is
   missing (more frames = higher confidence).
4. Added Two-wheeler to skip list — rear plates are rarely visible from
   CCTV angle in Indian traffic.
"""
from __future__ import annotations

from ..types import Detection, Track
from .geometry import make_violation, object_overlaps_vehicle


def check_no_plate(track: Track, plates: list[Detection], cfg: dict,
                   camera_id: str, timestamp: float = 0.0):
    plate_iou = cfg.get("plate_iou", 0.10)
    missing_frames = cfg.get("missing_frames", 45)  # was 15 → now 45 (~1.5s at 30fps)
    skip = set(cfg.get("skip_classes", ["Bicycle", "Three-wheeler", "Two-wheeler"]))

    has_plate = any(
        object_overlaps_vehicle(track.bbox, p.bbox, plate_iou)
        for p in plates
    )
    if has_plate:
        track.no_plate_frames = 0
        return

    track.no_plate_frames += 1
    if track.no_plate_frames >= missing_frames and track.cls_name not in skip:
        # Scale confidence by how long plate has been missing
        # 45 frames → 0.65 (LOG_ONLY), 90+ frames → 0.80 (HUMAN_REVIEW)
        conf = min(0.50 + 0.005 * track.no_plate_frames, 0.85)
        yield make_violation("NO_PLATE", track, camera_id, confidence=conf,
                             timestamp=timestamp,
                             consistent_frames=track.no_plate_frames)
