"""
modules/pdf_handler.py
=======================
Converts PDF files into per-page PNG images for OCR processing.

Uses PyMuPDF (fitz) — zero-dependency, no external tools needed.
  pip install pymupdf

Features
--------
• Respects Config.PDF_DPI for resolution (200 DPI = good OCR quality)
• Applies auto-contrast normalisation per page
• Skips blank pages (heuristic: mostly white)
• Returns list of {path, origin, page_num} dicts compatible with pipeline
"""

from __future__ import annotations

import os
import uuid
import numpy as np
from pathlib import Path

try:
    import fitz   # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False
    print("[PDF] PyMuPDF not installed. PDF support disabled.")
    print("      Run: pip install pymupdf")

import cv2
from config import Config


class PDFHandler:

    def __init__(self):
        self._ready = _FITZ_AVAILABLE

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def split_to_images(self, pdf_path: str, output_dir: str) -> list[dict]:
        """
        Render each page of a PDF to a PNG and return a list of dicts:
        [
            {"path": "/uploads/<id>.png", "origin": "notes.pdf", "page_num": 1},
            ...
        ]

        Blank pages (>95% white pixels) are automatically skipped.
        """
        if not _FITZ_AVAILABLE:
            raise RuntimeError(
                "PyMuPDF is not installed. Run: pip install pymupdf"
            )

        origin   = Path(pdf_path).name
        results  = []
        mat      = fitz.Matrix(Config.PDF_DPI / 72.0, Config.PDF_DPI / 72.0)

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF '{pdf_path}': {e}")

        total_pages = min(len(doc), Config.PDF_MAX_PAGES)

        for page_num in range(total_pages):
            page  = doc[page_num]
            pix   = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            # Convert raw bytes → numpy uint8 array
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width
            )

            # Skip blank/near-blank pages
            white_ratio = np.sum(img_array > 240) / img_array.size
            if white_ratio > 0.97:
                print(f"[PDF] Skipping blank page {page_num + 1}/{total_pages}")
                continue

            # Auto-contrast stretch
            img_array = self._auto_contrast(img_array)

            out_path = os.path.join(output_dir, f"pdf_{uuid.uuid4().hex}.png")
            cv2.imwrite(out_path, img_array)

            results.append({
                "path"    : out_path,
                "origin"  : origin,
                "page_num": page_num + 1,
            })

        doc.close()
        print(f"[PDF] Split '{origin}': {len(results)} usable pages")
        return results

    def is_ready(self) -> bool:
        return self._ready

    def page_count(self, pdf_path: str) -> int:
        """Return number of pages in a PDF without full rendering."""
        if not _FITZ_AVAILABLE:
            return 0
        try:
            doc = fitz.open(pdf_path)
            n   = len(doc)
            doc.close()
            return n
        except Exception:
            return 0

    # ──────────────────────────────────────────────────────────────────────────
    #  Internal
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _auto_contrast(img: np.ndarray) -> np.ndarray:
        """
        Stretch the histogram so the darkest pixel becomes 0 and
        the brightest becomes 255. Improves OCR on faded photocopies.
        """
        p2,  p98 = np.percentile(img, (2, 98))
        if p98 - p2 < 10:
            return img    # already flat (blank page guard)

        stretched = np.clip((img.astype(np.float32) - p2) / (p98 - p2) * 255, 0, 255)
        return stretched.astype(np.uint8)
