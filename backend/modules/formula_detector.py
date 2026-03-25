"""
modules/formula_detector.py
============================
Detects and converts mathematical formulas / equations found in OCR text.

Two-stage approach
------------------
Stage 1 — Regex pattern matching on raw OCR text
  Identifies common formula patterns written in text form and converts them
  to proper Unicode / LaTeX notation.

Stage 2 — pix2tex (LaTeX-OCR) visual detection  [optional, requires GPU or patience]
  Scans the source image for regions that look like mathematical expressions,
  runs the pix2tex model, and inserts the LaTeX into the annotated text.

If pix2tex is not installed the module gracefully falls back to regex-only.
  pip install pix2tex[gui]   (optional, ~3 GB download due to model weights)

Unicode math symbols used (no LaTeX renderer needed in DOCX)
------------------------------------------------------------
  ²  ³  ⁴  ⁵  ⁶  ⁷  ⁸  ⁹  ⁰  → superscripts
  ₀  ₁  ₂  ₃  ₄  ₅  ₆  ₇  ₈  ₉ → subscripts
  ∫  ∑  ∏  √  ≠  ≈  ≤  ≥  ±  ÷  × → operators
  α  β  γ  δ  ε  θ  λ  μ  π  σ  φ  ω → Greek
  →  ←  ↔  ∞  ∈  ∉  ⊂  ⊃  ∩  ∪  ∅  ∂  ∇ → misc
"""

from __future__ import annotations

import re
import cv2
import numpy as np
from typing import Optional

try:
    from pix2tex.cli import LatexOCR
    _PIX2TEX_AVAILABLE = True
except ImportError:
    _PIX2TEX_AVAILABLE = False

from PIL import Image


# ── Unicode conversion maps ───────────────────────────────────────────────────

SUPERSCRIPTS = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "n": "ⁿ", "x": "ˣ", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ",
    "i": "ⁱ",
}

SUBSCRIPTS = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "n": "ₙ", "x": "ₓ", "a": "ₐ", "e": "ₑ", "i": "ᵢ",
}

GREEK_WORDS = {
    r"\balpha\b"  : "α",  r"\bbeta\b"   : "β",  r"\bgamma\b"  : "γ",
    r"\bdelta\b"  : "δ",  r"\bepsilon\b": "ε",  r"\btheta\b"  : "θ",
    r"\blambda\b" : "λ",  r"\bmu\b"     : "μ",  r"\bpi\b"     : "π",
    r"\bsigma\b"  : "σ",  r"\bphi\b"    : "φ",  r"\bomega\b"  : "ω",
    r"\bDelta\b"  : "Δ",  r"\bSigma\b"  : "Σ",  r"\bLambda\b" : "Λ",
    r"\bPi\b"     : "Π",  r"\bGamma\b"  : "Γ",  r"\bTheta\b"  : "Θ",
    r"\binfinity\b": "∞", r"\binfty\b"  : "∞",
}

OPERATOR_WORDS = {
    r"\btherefore\b"   : "∴",
    r"\bbecause\b"     : "∵",
    r"\bimplies\b"     : "⟹",
    r"\biff\b"         : "⟺",
    r"\bforall\b"      : "∀",
    r"\bexists\b"      : "∃",
    r"\bin\b"          : "∈",
    r"\bsubset\b"      : "⊂",
    r"\bunion\b"       : "∪",
    r"\bintersection\b": "∩",
    r"\bapprox\b"      : "≈",
    r"\bneq\b"         : "≠",
    r"\bleq\b"         : "≤",
    r"\bgeq\b"         : "≥",
    r"\bsqrt\b"        : "√",
    r"\bintegral\b"    : "∫",
    r"\bsummation\b"   : "∑",
    r"\bproduct\b"     : "∏",
    r"\bpartial\b"     : "∂",
    r"\bnabla\b"       : "∇",
}

# ASCII → Unicode operator pairs
ASCII_OPS = [
    (r"!=",  "≠"),
    (r"<=",  "≤"),
    (r">=",  "≥"),
    (r"->",  "→"),
    (r"<->", "↔"),
    (r"=>",  "⟹"),
    (r"<=>", "⟺"),
    (r"\+-", "±"),
    (r"~=",  "≈"),
]


class FormulaDetector:

    def __init__(self):
        self._pix2tex_model: Optional["LatexOCR"] = None
        self._init_pix2tex()

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def detect(self, image_path: str, raw_text: str) -> dict:
        """
        Detect formulas in raw_text and optionally in the image.

        Returns:
        {
            "annotated_text" : str,         # text with Unicode symbols
            "formulas"       : [            # list of detected formulas
                {
                  "original" : str,         # as seen in raw text
                  "unicode"  : str,         # Unicode representation
                  "latex"    : str,         # LaTeX (if pix2tex available)
                  "type"     : str,         # "inline" | "block"
                }
            ]
        }
        """
        formulas       = []
        annotated_text = raw_text

        # Stage 1: regex text-based detection
        annotated_text, text_formulas = self._regex_convert(annotated_text)
        formulas.extend(text_formulas)

        # Stage 2: visual pix2tex (optional)
        if self._pix2tex_model:
            visual_formulas = self._visual_detect(image_path)
            formulas.extend(visual_formulas)

            # Inject visual formulas at the end if not already in text
            for vf in visual_formulas:
                if vf["unicode"] not in annotated_text:
                    annotated_text += f"\n[Formula: {vf['unicode']}]"

        return {
            "annotated_text": annotated_text,
            "formulas"      : formulas,
        }

    def is_ready(self) -> bool:
        return True   # regex is always ready

    # ──────────────────────────────────────────────────────────────────────────
    #  Stage 1 — Regex Conversion
    # ──────────────────────────────────────────────────────────────────────────

    def _regex_convert(self, text: str) -> tuple[str, list[dict]]:
        """
        Convert text-encoded math to Unicode. Returns (modified_text, formulas).
        """
        formulas = []

        # ── Superscripts: x^2, x^n, x^(2n+1) ─────────────────────────────
        def replace_superscript(m):
            base = m.group(1)
            exp  = m.group(2)
            uni  = self._to_superscript(exp)
            orig = m.group(0)
            formulas.append({"original": orig, "unicode": base + uni,
                              "latex": f"{base}^{{{exp}}}", "type": "inline"})
            return base + uni

        text = re.sub(
            r"(\w)\^(\{[^}]+\}|[0-9a-zA-Z]+)",
            replace_superscript, text
        )

        # ── Subscripts: x_1, x_{n} ────────────────────────────────────────
        def replace_subscript(m):
            base = m.group(1)
            sub  = m.group(2).strip("{}")
            uni  = self._to_subscript(sub)
            orig = m.group(0)
            formulas.append({"original": orig, "unicode": base + uni,
                              "latex": f"{base}_{{{sub}}}", "type": "inline"})
            return base + uni

        text = re.sub(
            r"(\w)_(\{[^}]+\}|[0-9a-zA-Z]+)",
            replace_subscript, text
        )

        # ── Square root: sqrt(x) → √x ─────────────────────────────────────
        def replace_sqrt(m):
            arg  = m.group(1)
            uni  = f"√({arg})"
            orig = m.group(0)
            formulas.append({"original": orig, "unicode": uni,
                              "latex": f"\\sqrt{{{arg}}}", "type": "inline"})
            return uni

        text = re.sub(r"sqrt\(([^)]+)\)", replace_sqrt, text, flags=re.IGNORECASE)

        # ── Fractions: a/b written in isolated context ─────────────────────
        # e.g. "1/2" → ½ for common fractions
        common_fracs = {
            "1/2": "½", "1/3": "⅓", "2/3": "⅔",
            "1/4": "¼", "3/4": "¾", "1/8": "⅛",
        }
        for ascii_f, uni_f in common_fracs.items():
            if ascii_f in text:
                formulas.append({"original": ascii_f, "unicode": uni_f,
                                  "latex": ascii_f, "type": "inline"})
                text = text.replace(ascii_f, uni_f)

        # ── Greek letter words → symbols ──────────────────────────────────
        for pattern, symbol in GREEK_WORDS.items():
            if re.search(pattern, text, re.IGNORECASE):
                orig_matches = re.findall(pattern, text, re.IGNORECASE)
                for orig in orig_matches:
                    formulas.append({"original": orig, "unicode": symbol,
                                      "latex": f"\\{orig.lower()}", "type": "inline"})
                text = re.sub(pattern, symbol, text, flags=re.IGNORECASE)

        # ── Operator words → symbols ──────────────────────────────────────
        for pattern, symbol in OPERATOR_WORDS.items():
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, symbol, text, flags=re.IGNORECASE)

        # ── ASCII operator sequences ──────────────────────────────────────
        for ascii_op, uni_op in sorted(ASCII_OPS, key=lambda x: -len(x[0])):
            if ascii_op in text:
                text = text.replace(ascii_op, uni_op)

        # ── Integral notation: int(...) ───────────────────────────────────
        def replace_integral(m):
            expr = m.group(1)
            uni  = f"∫({expr})"
            formulas.append({"original": m.group(0), "unicode": uni,
                              "latex": f"\\int {{{expr}}}", "type": "block"})
            return uni

        text = re.sub(
            r"\bint(?:egral)?\s*\(([^)]+)\)",
            replace_integral, text, flags=re.IGNORECASE
        )

        # ── Sum notation: sum(i=1, n) ─────────────────────────────────────
        def replace_sum(m):
            inner = m.group(1)
            uni   = f"∑({inner})"
            formulas.append({"original": m.group(0), "unicode": uni,
                              "latex": f"\\sum_{{{inner}}}", "type": "block"})
            return uni

        text = re.sub(
            r"\bsum\s*\(([^)]+)\)",
            replace_sum, text, flags=re.IGNORECASE
        )

        return text, formulas

    # ──────────────────────────────────────────────────────────────────────────
    #  Stage 2 — Visual Formula Detection (pix2tex)
    # ──────────────────────────────────────────────────────────────────────────

    def _init_pix2tex(self):
        if not _PIX2TEX_AVAILABLE:
            return
        try:
            self._pix2tex_model = LatexOCR()
            print("[Formula] pix2tex (LaTeX-OCR) model loaded")
        except Exception as e:
            print(f"[Formula] pix2tex load failed: {e}")
            self._pix2tex_model = None

    def _visual_detect(self, image_path: str) -> list[dict]:
        """
        Find regions in the image that look like block equations and run
        pix2tex on each region.

        Heuristic for "formula region":
        - Find contours of connected components
        - Identify clusters of small symbols (< average char height)
          arranged horizontally → likely a formula line
        """
        if not self._pix2tex_model:
            return []

        results = []
        try:
            img  = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return []

            # Binarise
            _, binary = cv2.threshold(img, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Find horizontal bands with high density of small symbols
            formula_regions = self._find_formula_regions(binary)

            for (y1, y2) in formula_regions:
                # Add padding
                pad    = 10
                y1_pad = max(0, y1 - pad)
                y2_pad = min(img.shape[0], y2 + pad)
                region = img[y1_pad:y2_pad, :]

                pil_region = Image.fromarray(region)
                latex      = self._pix2tex_model(pil_region)

                if latex and len(latex.strip()) > 3:
                    # Convert LaTeX to Unicode approximation
                    uni = self._latex_to_unicode(latex)
                    results.append({
                        "original": f"[visual region y={y1}-{y2}]",
                        "unicode" : uni,
                        "latex"   : latex,
                        "type"    : "block",
                    })

        except Exception as e:
            print(f"[Formula] Visual detection error: {e}")

        return results

    @staticmethod
    def _find_formula_regions(binary: np.ndarray) -> list[tuple[int, int]]:
        """
        Return list of (y_start, y_end) for rows likely containing formulas.
        Heuristic: rows with many small contiguous blobs.
        """
        # Row projection: sum of non-zero pixels per row
        row_sum = np.sum(binary > 0, axis=1)
        h = binary.shape[0]

        # Find rows with significant ink but below average (formulas are denser)
        mean_ink  = np.mean(row_sum[row_sum > 0]) if np.any(row_sum > 0) else 1
        threshold = mean_ink * 0.3

        in_region  = False
        start_row  = 0
        regions    = []
        min_height = 15   # at least 15px tall to be a formula

        for r in range(h):
            if row_sum[r] > threshold and not in_region:
                in_region = True
                start_row = r
            elif row_sum[r] <= threshold and in_region:
                in_region = False
                if r - start_row >= min_height:
                    regions.append((start_row, r))

        return regions[:5]   # cap at 5 regions per page

    @staticmethod
    def _latex_to_unicode(latex: str) -> str:
        """Best-effort LaTeX → Unicode conversion for display."""
        replacements = {
            r"\\frac\{([^}]+)\}\{([^}]+)\}" : lambda m: f"({m.group(1)}/{m.group(2)})",
            r"\\sqrt\{([^}]+)\}"            : lambda m: f"√({m.group(1)})",
            r"\\sum"    : "∑", r"\\int"  : "∫", r"\\prod" : "∏",
            r"\\infty"  : "∞", r"\\alpha": "α", r"\\beta" : "β",
            r"\\gamma"  : "γ", r"\\delta": "δ", r"\\theta": "θ",
            r"\\pi"     : "π", r"\\sigma": "σ", r"\\mu"   : "μ",
            r"\\lambda" : "λ", r"\\omega": "ω", r"\\phi"  : "φ",
            r"\\leq"    : "≤", r"\\geq"  : "≥", r"\\neq"  : "≠",
            r"\\approx" : "≈", r"\\pm"   : "±", r"\\cdot" : "·",
            r"\\times"  : "×", r"\\div"  : "÷", r"\\nabla": "∇",
            r"\\partial": "∂", r"\\in"   : "∈", r"\\cup"  : "∪",
            r"\\cap"    : "∩", r"\\rightarrow": "→",
            r"\\left\(" : "(", r"\\right\)": ")",
            r"\\left\[" : "[", r"\\right\]": "]",
            r"\^2"      : "²", r"\^3"    : "³",
            r"\{|\}"    : "",
        }

        result = latex
        for pattern, repl in replacements.items():
            if callable(repl):
                result = re.sub(pattern, repl, result)
            else:
                result = re.sub(pattern, repl, result)

        # Remove remaining LaTeX commands
        result = re.sub(r"\\[a-zA-Z]+", "", result)
        result = re.sub(r"\s+", " ", result).strip()
        return result

    # ──────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_superscript(s: str) -> str:
        s = s.strip("{}")
        return "".join(SUPERSCRIPTS.get(c, c) for c in s)

    @staticmethod
    def _to_subscript(s: str) -> str:
        s = s.strip("{}")
        return "".join(SUBSCRIPTS.get(c, c) for c in s)
