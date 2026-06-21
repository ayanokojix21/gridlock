"""Shared dataclasses passed between pipeline stages.

These are deliberately framework-agnostic (plain dataclasses, numpy bboxes) so
the orchestrator, model wrappers, and violation analyzers stay decoupled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

BBox = tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels


def compute_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two axis-aligned boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_center(b: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class Detection:
    """A single raw detection from any detector model."""

    bbox: BBox
    confidence: float
    cls_id: int
    cls_name: str
    source: str = ""  # which model produced it: "vehicle" | "person" | "plate" | ...

    @property
    def center(self) -> tuple[float, float]:
        return bbox_center(self.bbox)


@dataclass
class Track:
    """A tracked vehicle with temporal state for violation logic."""

    id: int
    bbox: BBox
    confidence: float
    cls_id: int
    cls_name: str
    trajectory: list[tuple[float, float]] = field(default_factory=list)  # centers over time

    # Per-violation persistence counters (frames in violating state)
    helmet_violation_frames: int = 0
    seatbelt_violation_frames: int = 0
    triple_riding_frames: int = 0
    wrong_side_frames: int = 0
    no_plate_frames: int = 0

    # Best confidence observed for each violation kind (for evidence)
    _conf_cache: dict[str, float] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        return bbox_center(self.bbox)


@dataclass
class Violation:
    """A detected violation, enriched through Stages 5 & 6."""

    type: str                      # "HELMET_VIOLATION", ...
    confidence: float              # raw model/logic confidence
    vehicle_track_id: int
    camera_id: str
    vehicle_bbox: BBox
    vehicle_type: str = ""
    timestamp: float = 0.0
    consistent_frames: int = 1

    # populated in Stage 5/6
    plate_number: Optional[str] = None
    plate_confidence: float = 0.0
    raw_confidence: float = 0.0
    final_confidence: float = 0.0
    action: str = "LOG_ONLY"       # AUTO_ENFORCE | HUMAN_REVIEW | LOG_ONLY

    # metadata
    violation_code: str = ""
    fine_inr: int = 0
    evidence: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.raw_confidence:
            self.raw_confidence = self.confidence


# Maps used across the pipeline ------------------------------------------------

# UVH-26 class id -> normalized vehicle category
UVH26_CLASSES: dict[int, str] = {
    0: "Hatchback", 1: "Sedan", 2: "SUV", 3: "MUV", 4: "Bus",
    5: "Truck", 6: "Three-wheeler", 7: "Two-wheeler", 8: "LCV",
    9: "Mini-bus", 10: "Tempo-traveller", 11: "Bicycle", 12: "Van", 13: "Others",
}

CAR_CLASSES = {"Hatchback", "Sedan", "SUV", "MUV", "Van", "Truck", "Bus", "LCV", "Mini-bus", "Tempo-traveller"}
TWO_WHEELER = "Two-wheeler"

# Violation type -> (MVA section, fine ₹)
VIOLATION_META: dict[str, tuple[str, int]] = {
    "HELMET_VIOLATION": ("S129", 1000),
    "SEATBELT_VIOLATION": ("S138(3)", 1000),
    "TRIPLE_RIDING": ("S128", 1000),
    "WRONG_SIDE_DRIVING": ("S184", 5000),
    "STOP_LINE_VIOLATION": ("S177A", 1000),
    "RED_LIGHT_VIOLATION": ("S119/177", 5000),
    "ILLEGAL_PARKING": ("S122/177", 500),
    "NO_PLATE": ("S39", 5000),
}
