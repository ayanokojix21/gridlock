"""Base for YOLO-style detector wrappers with lazy load + graceful fallback."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..types import Detection

log = logging.getLogger("ml.models")


class YoloDetectorBase:
    """Wraps an Ultralytics YOLO model. If the weights file is absent or
    ultralytics isn't installed, ``available`` is False and ``detect`` returns
    an empty list — the pipeline keeps running.
    """

    source = "detector"

    def __init__(self, weights: str | Path, device: str = "cpu", imgsz: int = 640):
        self.weights = str(weights)
        self.device = device
        self.imgsz = imgsz
        self._model = None
        self._tried = False

    # -- loading -------------------------------------------------------
    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    def _weights_exist(self) -> bool:
        # Ultralytics auto-download names (e.g. "yolo11n.pt") have no path sep.
        if "/" not in self.weights and "\\" not in self.weights:
            return True
        return Path(self.weights).exists()

    def _ensure_loaded(self) -> None:
        if self._tried:
            return
        self._tried = True
        if not self._weights_exist():
            log.warning("[%s] weights not found: %s — running in stub mode", self.source, self.weights)
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
            log.info("[%s] loaded %s on %s", self.source, self.weights, self.device)
        except Exception as e:  # pragma: no cover - depends on env
            log.warning("[%s] failed to load %s (%s) — stub mode", self.source, self.weights, e)
            self._model = None

    # -- inference -----------------------------------------------------
    def _map_class(self, cls_id: int) -> str:
        return str(cls_id)

    def _allowed_classes(self) -> Optional[list[int]]:
        return None

    def detect(self, frame, conf: float = 0.25, iou: float = 0.7) -> list[Detection]:
        self._ensure_loaded()
        if self._model is None:
            return []
        kwargs = dict(conf=conf, iou=iou, imgsz=self.imgsz, verbose=False)
        classes = self._allowed_classes()
        if classes is not None:
            kwargs["classes"] = classes
        if self.device.startswith("cuda"):
            kwargs["device"] = 0
        try:
            results = self._model(frame, **kwargs)
        except Exception as e:  # pragma: no cover
            log.warning("[%s] inference error: %s", self.source, e)
            return []
        return self._parse(results)

    def _parse(self, results) -> list[Detection]:
        out: list[Detection] = []
        if not results:
            return out
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                xyxy = b.xyxy[0].tolist()
                cls_id = int(b.cls[0])
                out.append(
                    Detection(
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        confidence=float(b.conf[0]),
                        cls_id=cls_id,
                        cls_name=self._map_class(cls_id),
                        source=self.source,
                    )
                )
            return out
        
        # Fallback for YOLO Image Classification models (like the seatbelt model)
        probs = getattr(results[0], "probs", None)
        if probs is not None:
            cls_id = int(probs.top1)
            conf = float(probs.top1conf)
            out.append(
                Detection(
                    bbox=(0.0, 0.0, 0.0, 0.0),  # dummy bbox for classification
                    confidence=conf,
                    cls_id=cls_id,
                    cls_name=self._map_class(cls_id),
                    source=self.source,
                )
            )
        return out
