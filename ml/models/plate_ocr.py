"""Model G — PaddleOCR PP-OCRv5 plate reader with Indian-plate validation.

Degrades to ("UNREADABLE", 0.0) when PaddleOCR isn't installed.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("ml.models")

DEFAULT_REGEX = r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$"


class PlateOCR:
    def __init__(self, device: str = "cpu", min_conf: float = 0.70, regex: str = DEFAULT_REGEX):
        self.device = device
        self.min_conf = min_conf
        self.regex = re.compile(regex)
        self._ocr = None
        self._tried = False

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._ocr is not None

    def _ensure_loaded(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            from paddleocr import PaddleOCR as _P

            use_gpu = self.device.startswith("cuda")
            # Try newer API first (PaddleOCR 3.7+ dropped show_log/use_gpu)
            try:
                self._ocr = _P(use_angle_cls=False, lang="en")
            except Exception:
                # Fallback for older paddleocr versions
                try:
                    self._ocr = _P(use_angle_cls=False, lang="en", show_log=False, use_gpu=use_gpu)
                except Exception:
                    self._ocr = _P(use_angle_cls=False, lang="en")
            log.info("[ocr] PaddleOCR ready (gpu=%s)", use_gpu)
        except Exception as e:  # pragma: no cover
            log.warning("[ocr] PaddleOCR unavailable (%s) — UNREADABLE fallback", e)
            self._ocr = None

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def read(self, crop) -> tuple[str, float]:
        """Return (plate_text, confidence). 'UNREADABLE' if invalid/low-conf."""
        self._ensure_loaded()
        if self._ocr is None or crop is None or getattr(crop, "size", 0) == 0:
            return "UNREADABLE", 0.0
        try:
            # Pass crop directly to OCR. We use kwargs to lower the detection thresholds 
            # so it tries to find text even in extremely low-res/blurry crops.
            result = self._ocr.ocr(crop, det=True, rec=True, cls=False)
        except Exception as e:  # pragma: no cover
            log.warning("[ocr] inference error: %s", e)
            return "UNREADABLE", 0.0
        if not result or not result[0]:
            return "UNREADABLE", 0.0
        # pick highest-confidence line
        best_text, best_conf = "", 0.0
        for line in result[0]:
            try:
                text, conf = line[1][0], float(line[1][1])
            except (IndexError, TypeError):
                continue
            if conf > best_conf:
                best_text, best_conf = self._clean(text), conf
        if best_text:
            log.info(f"[ocr debug] Read text: '{best_text}' (conf: {best_conf:.2f})")
        if best_conf >= self.min_conf and self.regex.match(best_text):
            return best_text, best_conf
        return "UNREADABLE", 0.0

    __call__ = read
