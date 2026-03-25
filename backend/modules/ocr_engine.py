"""
modules/ocr_engine.py
======================
Dual-engine OCR using EasyOCR and Tesseract with ensemble merging.

Strategy
--------
1. Run EasyOCR  → returns (text, confidence, bounding_boxes)
2. Run Tesseract → returns (text, per-word confidence, bounding_boxes)
3. Ensemble:
   - Where both engines agree on a word → keep it (high confidence)
   - Where they disagree → pick the word with higher per-word confidence
   - Words only seen by one engine → include if confidence > threshold
4. Reconstruct final text preserving reading order (top-to-bottom, left-to-right)

Free / Open-Source
------------------
• EasyOCR   : pip install easyocr   (MIT licence, no API key)
• Tesseract : system package         (Apache 2, no API key)
              Windows: https://github.com/UB-Mannheim/tesseract/wiki
              Ubuntu : sudo apt install tesseract-ocr
              macOS  : brew install tesseract
"""

from __future__ import annotations

import os
import re
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

try:
    import pytesseract
    from pytesseract import Output as TessOutput
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

from config import Config


@dataclass
class WordBox:
    text      : str
    x1        : int
    y1        : int
    x2        : int
    y2        : int
    confidence: float          # 0.0 – 1.0
    engine    : str = "ensemble"


@dataclass
class OCRResult:
    text       : str
    confidence : float            # mean confidence 0–100
    word_boxes : list[WordBox] = field(default_factory=list)


class OCREngine:

    def __init__(self):
        self._easy_reader: Optional["easyocr.Reader"] = None
        self._ready       = False
        self._init_engines()

    # ──────────────────────────────────────────────────────────────────────────
    #  Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def _init_engines(self):
        """Lazy-initialise engines; failures are soft (fallback to other engine)."""
        global _TESSERACT_AVAILABLE

        if _EASYOCR_AVAILABLE:
            try:
                langs = Config.OCR_LANGUAGES  # e.g. ["en", "hi"]
                self._easy_reader = easyocr.Reader(
                    langs,
                    gpu=Config.OCR_GPU,
                    verbose=False,
                )
                print(f"[OCR] EasyOCR ready — languages: {langs}")
            except Exception as e:
                print(f"[OCR] EasyOCR init failed: {e}")

        if _TESSERACT_AVAILABLE:
            try:
                pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
                ver = pytesseract.get_tesseract_version()
                print(f"[OCR] Tesseract ready — version: {ver}")
            except Exception as e:
                _TESSERACT_AVAILABLE = False
                print(f"[OCR] Tesseract not available (falling back to EasyOCR only): {e}")

        self._ready = bool(self._easy_reader) or _TESSERACT_AVAILABLE

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def extract(self, image_path: str) -> dict:
        """
        Run OCR on image_path and return:
        {
            "text"       : str,         # clean full text
            "confidence" : float,       # mean confidence 0–100
            "word_boxes" : list[dict],  # bounding-box records
        }
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"text": "", "confidence": 0.0, "word_boxes": []}

        easy_words  = self._run_easyocr(image_path)
        tess_words  = self._run_tesseract(image_path)
        merged      = self._ensemble(easy_words, tess_words, img.shape)

        text        = self._reconstruct_text(merged)
        confidence  = (
            float(np.mean([w.confidence for w in merged])) * 100
            if merged else 0.0
        )
        word_boxes  = [
            {
                "text"      : w.text,
                "x1": w.x1, "y1": w.y1, "x2": w.x2, "y2": w.y2,
                "confidence": round(w.confidence * 100, 1),
                "engine"    : w.engine,
            }
            for w in merged
        ]

        return {
            "text"      : text,
            "confidence": round(confidence, 1),
            "word_boxes": word_boxes,
        }

    def is_ready(self) -> bool:
        return self._ready

    # ──────────────────────────────────────────────────────────────────────────
    #  Engine Runners
    # ──────────────────────────────────────────────────────────────────────────

    def _run_easyocr(self, image_path: str) -> list[WordBox]:
        if not self._easy_reader:
            return []
        try:
            results = self._easy_reader.readtext(image_path, detail=1)
            words   = []
            for (bbox, text, conf) in results:
                xs  = [p[0] for p in bbox]
                ys  = [p[1] for p in bbox]
                words.append(WordBox(
                    text=text.strip(),
                    x1=int(min(xs)), y1=int(min(ys)),
                    x2=int(max(xs)), y2=int(max(ys)),
                    confidence=float(conf),
                    engine="easyocr"
                ))
            return words
        except Exception as e:
            print(f"[OCR] EasyOCR extraction error: {e}")
            return []

    def _run_tesseract(self, image_path: str) -> list[WordBox]:
        if not _TESSERACT_AVAILABLE:
            return []
        try:
            # Build lang string: "eng+hin"
            lang_map = {"en": "eng", "hi": "hin", "bn": "ben",
                        "or": "ori", "ta": "tam", "te": "tel",
                        "fr": "fra", "de": "deu", "es": "spa"}
            lang_str = "+".join(
                lang_map.get(l.strip(), "eng") for l in Config.OCR_LANGUAGES
            )

            data = pytesseract.image_to_data(
                image_path,
                lang=lang_str,
                output_type=TessOutput.DICT,
                config="--psm 3 --oem 3"
            )

            words = []
            for i, text in enumerate(data["text"]):
                text = str(text).strip()
                if not text:
                    continue
                conf = int(data["conf"][i])
                if conf < 0:
                    continue
                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])
                words.append(WordBox(
                    text=text,
                    x1=x, y1=y,
                    x2=x + w, y2=y + h,
                    confidence=conf / 100.0,
                    engine="tesseract"
                ))
            return words
        except Exception as e:
            print(f"[OCR] Tesseract extraction error: {e}")
            return []

    # ──────────────────────────────────────────────────────────────────────────
    #  Ensemble Merging
    # ──────────────────────────────────────────────────────────────────────────

    def _ensemble(
        self,
        easy_words : list[WordBox],
        tess_words : list[WordBox],
        img_shape  : tuple,
    ) -> list[WordBox]:
        """
        Merge two word lists:
        - Prefer the word whose bounding box overlaps with the other engine,
          selecting the higher-confidence variant.
        - Non-overlapping words are included if confidence ≥ 0.4.
        - Fallback: if one engine returned nothing, return the other.
        """
        if not easy_words:
            return [w for w in tess_words if w.confidence >= 0.3]
        if not tess_words:
            return [w for w in easy_words if w.confidence >= 0.3]

        merged  = []
        used_t  = set()
        IOU_THR = 0.30      # minimum IOU to consider same word

        for ew in easy_words:
            best_t   = None
            best_iou = 0.0
            for ti, tw in enumerate(tess_words):
                iou = self._iou(ew, tw)
                if iou > best_iou:
                    best_iou = iou
                    best_t   = (ti, tw)

            if best_t and best_iou >= IOU_THR:
                ti, tw = best_t
                used_t.add(ti)
                winner = ew if ew.confidence >= tw.confidence else tw
                winner.engine = "ensemble"
                merged.append(winner)
            else:
                if ew.confidence >= 0.40:
                    merged.append(ew)

        # Add Tesseract-only words not matched
        for ti, tw in enumerate(tess_words):
            if ti not in used_t and tw.confidence >= 0.45:
                merged.append(tw)

        return merged

    @staticmethod
    def _iou(a: WordBox, b: WordBox) -> float:
        """Intersection over Union for two bounding boxes."""
        xi1 = max(a.x1, b.x1)
        yi1 = max(a.y1, b.y1)
        xi2 = min(a.x2, b.x2)
        yi2 = min(a.y2, b.y2)

        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter   = inter_w * inter_h

        if inter == 0:
            return 0.0

        area_a = max(1, (a.x2 - a.x1) * (a.y2 - a.y1))
        area_b = max(1, (b.x2 - b.x1) * (b.y2 - b.y1))
        union  = area_a + area_b - inter
        return inter / union

    # ──────────────────────────────────────────────────────────────────────────
    #  Text Reconstruction
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _reconstruct_text(words: list[WordBox]) -> str:
        """
        Convert word-box list to readable text preserving reading order.

        Uses a simple line-grouping algorithm:
        - Sort words by top-y
        - Group words whose vertical midpoints are within LINE_GAP pixels
        - Within each line, sort by x1
        - Join lines with newlines
        """
        if not words:
            return ""

        LINE_GAP = 15   # pixels; words within this vertical range = same line

        # Sort by vertical position first
        words_sorted = sorted(words, key=lambda w: (w.y1 + w.y2) / 2)

        lines      : list[list[WordBox]] = []
        cur_line   : list[WordBox]       = [words_sorted[0]]
        cur_mid_y  = (words_sorted[0].y1 + words_sorted[0].y2) / 2

        for word in words_sorted[1:]:
            mid_y = (word.y1 + word.y2) / 2
            if abs(mid_y - cur_mid_y) <= LINE_GAP:
                cur_line.append(word)
                cur_mid_y = (cur_mid_y + mid_y) / 2   # rolling average
            else:
                lines.append(sorted(cur_line, key=lambda w: w.x1))
                cur_line  = [word]
                cur_mid_y = mid_y

        lines.append(sorted(cur_line, key=lambda w: w.x1))

        text_lines = []
        for line in lines:
            line_text = " ".join(w.text for w in line)
            line_text = re.sub(r"\s+", " ", line_text).strip()
            if line_text:
                text_lines.append(line_text)

        return "\n".join(text_lines)
