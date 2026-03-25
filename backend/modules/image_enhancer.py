"""
modules/image_enhancer.py
==========================
Pre-processes handwritten note images before OCR.

Pipeline (applied in order):
  1. Load & auto-rotate EXIF orientation
  2. Resize to target resolution  (preserves aspect ratio)
  3. Convert to greyscale
  4. Deskew  (correct tilt up to ±15°)
  5. Denoise (Non-local Means)
  6. Adaptive thresholding → clean black-on-white
  7. Morphological opening  (remove tiny specks)
  8. Optional sharpen
  9. Save as high-quality PNG

Modes:
  "auto"   → runs the full pipeline  (recommended)
  "light"  → skip denoise + morph    (fast, already clean images)
  "scan"   → optimised for camera-captured pages (extra contrast boost)
"""

import os
import uuid
import math
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ExifTags


class ImageEnhancer:

    def __init__(self):
        self._ready = True

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def enhance(self, image_path: str, output_dir: str, mode: str = "auto") -> str:
        """
        Enhance an image for OCR and return the path to the enhanced PNG.

        Parameters
        ----------
        image_path : str   Path to source image
        output_dir : str   Directory to save enhanced output
        mode       : str   "auto" | "light" | "scan"

        Returns
        -------
        str  Path to enhanced PNG
        """
        img  = self._load_with_exif(image_path)
        img  = self._resize(img)
        gray = self._to_gray(img)

        if mode in ("auto", "scan"):
            gray = self._deskew(gray)
            gray = self._denoise(gray)
            gray = self._adaptive_threshold(gray)
            gray = self._morph_clean(gray)
            if mode == "scan":
                gray = self._contrast_boost(gray)
        elif mode == "light":
            gray = self._deskew(gray)
            gray = self._adaptive_threshold(gray)
        else:
            # fallback full pipeline
            gray = self._deskew(gray)
            gray = self._denoise(gray)
            gray = self._adaptive_threshold(gray)
            gray = self._morph_clean(gray)

        out_path = os.path.join(output_dir, f"enh_{uuid.uuid4().hex}.png")
        cv2.imwrite(out_path, gray)
        return out_path

    def is_ready(self) -> bool:
        return self._ready

    # ──────────────────────────────────────────────────────────────────────────
    #  Internal Steps
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_with_exif(path: str) -> np.ndarray:
        """Load via PIL (handles EXIF rotation), then convert to OpenCV BGR."""
        pil = Image.open(path)
        pil = ImageOps.exif_transpose(pil)   # auto-rotate via EXIF
        pil = pil.convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _resize(img: np.ndarray, max_side: int = 2400) -> np.ndarray:
        """
        Downscale to max_side px on the longest edge (upscale if very small).
        Keeps aspect ratio.
        """
        h, w = img.shape[:2]
        longest = max(h, w)
        min_side = 800

        if longest < min_side:
            scale = min_side / longest
        elif longest > max_side:
            scale = max_side / longest
        else:
            return img

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _deskew(gray: np.ndarray) -> np.ndarray:
        """
        Detect document skew angle via Hough lines and rotate to correct it.
        Only corrects angles within ±15° (ignores landscape/portrait confusion).
        """
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180,
                                threshold=100,
                                minLineLength=gray.shape[1] // 4,
                                maxLineGap=20)

        if lines is None:
            return gray

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if -15 < angle < 15:
                angles.append(angle)

        if not angles:
            return gray

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.3:
            return gray

        h, w   = gray.shape
        center = (w // 2, h // 2)
        M      = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def _denoise(gray: np.ndarray) -> np.ndarray:
        """
        Non-local Means denoising — effective for scanned/photographed notes.
        h=10 is a balanced value; increase for noisier images.
        """
        return cv2.fastNlMeansDenoising(gray, h=10,
                                        templateWindowSize=7,
                                        searchWindowSize=21)

    @staticmethod
    def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
        """
        Adaptive Gaussian thresholding creates a clean black-on-white image
        that handles uneven lighting (common in phone-captured notes).
        blockSize=31 works well for A4 photos; reduce for small regions.
        """
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=11
        )

    @staticmethod
    def _morph_clean(binary: np.ndarray) -> np.ndarray:
        """
        Morphological opening removes salt-and-pepper noise (small white dots
        on black background and vice versa) without affecting text strokes.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def _contrast_boost(gray: np.ndarray) -> np.ndarray:
        """CLAHE-based local contrast enhancement for dark/faded notes."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
