"""
modules/ai_formatter.py
========================
Formats handwritten notes using Claude Vision (multimodal).

Strategy
--------
PRIMARY  : Send the actual image to Claude Vision.
           Claude reads the handwriting directly — far more accurate than OCR text.
FALLBACK : If image not available or vision fails, use corrected OCR text with
           a strong reconstruction prompt.

This completely bypasses the OCR garbling problem by letting Claude see
the original image, just like a human would.

Model
-----
claude-3-haiku-20240307  — supports vision, fast, cheap (~$0.00025/1K tokens)
claude-3-5-sonnet-20241022 — best quality vision

API key: https://console.anthropic.com  (free $5 credit on signup)
"""

from __future__ import annotations

import re
import json
import base64
import os
from typing import Optional

import anthropic

from config import Config


# ══════════════════════════════════════════════════════════════════════════════
#  OCR Pre-correction (used only for text-only fallback path)
# ══════════════════════════════════════════════════════════════════════════════

OCR_FIXUPS = [
    (r"\b([a-zA-Z0-9])\s*\+-\s*",       r"\1 ± "),
    (r"\b([a-zA-Z0-9])\s*\-\+\s*",      r"\1 ± "),
    (r"(?<![a-zA-Z])sqrt\s*\(",          r"√("),
    (r"\bS(?:qrt|q?rt)?\s*\(",           r"√("),
    (r"([a-zA-Z0-9])\s*[°º]\s*(\d)",    r"\1^\2"),
    (r"S[°º]",                           r"0"),
    (r"\s*=o\b",                         r" = 0"),
    (r"[¿¡]",                            r""),
    (r"\b([A-Za-z])\s*3\s*\+",          r"\1² +"),
    (r"\b([A-Za-z])\s*2\s*\+",          r"\1² +"),
    (r"  +",                             r" "),
    # Greek letters
    (r"\balpha\b", "α"), (r"\bbeta\b",  "β"), (r"\bgamma\b", "γ"),
    (r"\bdelta\b", "δ"), (r"\btheta\b", "θ"), (r"\bpi\b",    "π"),
    (r"\bsigma\b", "σ"), (r"\bomega\b", "ω"),
]

def _preprocess_ocr(text: str) -> str:
    corrected = text
    for pattern, replacement in OCR_FIXUPS:
        corrected = re.sub(pattern, replacement, corrected)
    return corrected


# ══════════════════════════════════════════════════════════════════════════════
#  Basic fallback (no API)
# ══════════════════════════════════════════════════════════════════════════════

def _basic_format(raw_text: str, subject: str) -> dict:
    lines    = [l.strip() for l in raw_text.splitlines() if l.strip()]
    title    = lines[0][:80] if lines else "Untitled Notes"
    body     = lines[1:] if len(lines) > 1 else lines
    sections = []
    cur      = {"heading": "", "content": "", "bullets": [], "formulas": [], "notes": ""}
    prose    = []

    for line in body:
        is_heading = (len(line) < 60 and line.isupper()) or line.endswith(":")
        is_bullet  = bool(re.match(r"^[-*•]\s+|^\d+[.)]\s+", line))
        is_formula = bool(re.search(r"[=+\-*/√∫∑±²³]", line) and len(line) < 80)

        if is_heading:
            cur["content"] = " ".join(prose)
            prose = []
            sections.append(cur)
            cur = {"heading": line.rstrip(":"), "content": "",
                   "bullets": [], "formulas": [], "notes": ""}
        elif is_formula:
            cur["formulas"].append(line)
        elif is_bullet:
            cur["bullets"].append(re.sub(r"^[-*•]\s+|^\d+[.)]\s+", "", line))
        else:
            prose.append(line)

    cur["content"] = " ".join(prose)
    sections.append(cur)

    return {
        "title"    : title,
        "subject"  : subject,
        "summary"  : " ".join(body[:2])[:300],
        "key_terms": [],
        "sections" : [s for s in sections if s["content"] or s["bullets"] or s["formulas"]],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Image → base64 helper
# ══════════════════════════════════════════════════════════════════════════════

def _image_to_base64(image_path: str) -> tuple[str, str]:
    """
    Read image file and return (base64_string, media_type).
    Raises FileNotFoundError if path invalid.
    """
    ext_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
        ".gif": "image/gif",  ".bmp":  "image/png",   # convert bmp→png treatment
    }
    ext        = os.path.splitext(image_path)[1].lower()
    media_type = ext_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        data = f.read()

    # Cap at ~4MB base64 to stay within API limits
    MAX_BYTES = 4 * 1024 * 1024
    if len(data) > MAX_BYTES:
        # Resize via PIL if too large
        try:
            from PIL import Image
            import io
            img = Image.open(image_path)
            img.thumbnail((1600, 1600), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
            media_type = "image/jpeg"
        except Exception:
            data = data[:MAX_BYTES]

    return base64.standard_b64encode(data).decode("utf-8"), media_type


# ══════════════════════════════════════════════════════════════════════════════
#  Prompts
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_VISION = """You are an expert academic note digitizer with perfect handwriting recognition.

Your task: Look at the handwritten note image and transcribe + structure it into clean, well-formatted academic content.

RULES:
1. Read the handwriting carefully and accurately — do NOT guess randomly.
2. Preserve all mathematical formulas exactly as written, but use proper Unicode math symbols:
   ² ³ ⁴ √ ± × ÷ ≠ ≈ ≤ ≥ ∑ ∫ π α β γ δ θ λ μ σ φ ω → ← ∞ ∈ ∂ ∇
3. Structure the content logically: title → summary → sections with headings, bullets, formulas.
4. For the formula x = (-b ± √(b²-4ac)) / 2a — write it exactly like that using Unicode.
5. Return ONLY valid JSON — no markdown fences, no explanation text outside the JSON.
6. If text is unclear, use your subject knowledge to infer the most likely correct reading."""

SYSTEM_TEXT_FALLBACK = """You are an expert academic OCR correction and note formatting specialist.

Your task: receive GARBLED OCR text from handwritten student notes and RECONSTRUCT the intended meaning.

CRITICAL RULES:
1. OCR text from handwriting is garbled. Use subject knowledge to reconstruct meaning.
2. Fix ALL OCR errors:
   - "forw/forn"           → "form"
   - "ovadmatic/qvadratic" → "quadratic"
   - "Taentify"            → "Identify"
   - "Calcvlate"           → "Calculate"
   - "diserntminant"       → "discriminant"
   - "S°ivton/S0"          → "0" (zero)
   - "A3/AZ" (math)        → "ax²"
   - "b¿/b?" (math)        → "bx"
   - "arb anda C"          → "a, b and c"
   - "cons tnts"           → "constants"
   - "Ccoefficients"       → "coefficients"
   - "foy mula"            → "formula"
   - "bre" (math context)  → "b²e" or "b² -"
   - "¿ ¡"                 → "" (remove junk)
3. If you recognise a known formula/theorem, write it CORRECTLY and COMPLETELY.
4. Return ONLY valid JSON — no markdown, no preamble."""

JSON_SCHEMA = """
Return EXACTLY this JSON structure:
{
  "title": "Short clear title (max 8 words, no brackets or debug tags)",
  "subject": "<detected subject>",
  "summary": "2-3 clear sentences summarising the page content",
  "key_terms": ["term1", "term2", "term3"],
  "sections": [
    {
      "heading": "Section heading in Title Case (empty string if none)",
      "content": "Flowing prose paragraph for this section.",
      "bullets": ["numbered or bulleted point 1", "point 2"],
      "formulas": ["formula using Unicode e.g.  x = (-b ± √(b²-4ac)) / 2a"],
      "notes": "Any clarification note (empty string if none)"
    }
  ]
}

IMPORTANT:
- title must be clean plain text — NO square brackets, NO debug tags like [LIKELY FORMULA: ...]
- Use Unicode math symbols everywhere: ² ³ √ ± × ÷ ≠ ≈ ≤ ≥ ∑ ∫ π α β
- formulas array = one entry per distinct formula
- bullets = short clear points (steps, facts, notes)
- content = flowing prose
- JSON must be perfectly valid"""


# ══════════════════════════════════════════════════════════════════════════════
#  Main Class
# ══════════════════════════════════════════════════════════════════════════════

class AIFormatter:

    def __init__(self):
        self._client: Optional[anthropic.Anthropic] = None
        self._ready  = False
        self._init_client()

    def _init_client(self):
        api_key = Config.ANTHROPIC_API_KEY
        if not api_key:
            print("[AI] WARNING: ANTHROPIC_API_KEY not set — using basic formatter.")
            return
        try:
            self._client = anthropic.Anthropic(api_key=api_key)
            self._ready  = True
            print(f"[AI] Claude client ready — model: {Config.CLAUDE_MODEL}")
        except Exception as e:
            print(f"[AI] Claude client init failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────

    def format_page(
        self,
        raw_text   : str,
        subject    : str,
        page_label : str,
        formulas   : list,
        settings   : dict,
        image_path : str = "",      # ← NEW: path to enhanced image
    ) -> dict:
        """
        Format a single OCR page.

        Tries Claude Vision first (reads image directly — best accuracy).
        Falls back to text-based reconstruction if image unavailable.
        Falls back to basic formatter if API unavailable.
        """
        if not raw_text.strip() and not image_path:
            return {
                "title"    : page_label,
                "subject"  : subject,
                "summary"  : "No text detected on this page.",
                "key_terms": [],
                "sections" : [],
            }

        if not self._client:
            corrected = _preprocess_ocr(raw_text)
            return _basic_format(corrected, subject)

        # ── Try Claude Vision (PRIMARY path) ──────────────────────────────
        if image_path and os.path.isfile(image_path):
            try:
                result = self._claude_vision(
                    image_path, raw_text, subject, page_label, settings
                )
                print(f"[AI] Vision path succeeded for: {page_label}")
                return result
            except Exception as e:
                print(f"[AI] Vision path failed ({e}) — falling back to text path")

        # ── Fallback: text reconstruction ─────────────────────────────────
        corrected = _preprocess_ocr(raw_text)
        try:
            result = self._claude_text(
                corrected, raw_text, subject, page_label, formulas, settings
            )
            print(f"[AI] Text path succeeded for: {page_label}")
            return result
        except Exception as e:
            print(f"[AI] Text path also failed ({e}) — using basic formatter")
            return _basic_format(corrected, subject)

    def is_ready(self) -> bool:
        return self._ready

    # ─────────────────────────────────────────────────────────────────────────
    #  Claude Vision Path
    # ─────────────────────────────────────────────────────────────────────────

    def _claude_vision(
        self,
        image_path : str,
        raw_text   : str,
        subject    : str,
        page_label : str,
        settings   : dict,
    ) -> dict:
        """Send the actual image to Claude Vision for direct handwriting reading."""

        b64, media_type = _image_to_base64(image_path)
        language        = settings.get("language", "English")
        theme           = settings.get("theme", "academic")

        user_prompt = f"""Please read this handwritten note image carefully and convert it to structured digital notes.

PAGE: {page_label}
DETECTED SUBJECT (from OCR): {subject}
OUTPUT LANGUAGE: {language}
STYLE: {theme}

OCR HINT (may be garbled — use image as primary source):
\"\"\"{raw_text[:500]}\"\"\"

Instructions:
- Read the image directly — ignore garbled OCR hint if it conflicts with what you see
- Transcribe ALL text accurately, including every formula exactly as written
- Use Unicode math symbols: ² ³ √ ± × ÷ ≠ ≈ ≤ ≥ ∑ ∫ π α β γ δ θ λ μ σ
- Structure into logical sections with headings, bullets, and formula blocks
- Write formulas clearly, e.g.: x = (-b ± √(b²-4ac)) / 2a

{JSON_SCHEMA}"""

        message = self._client.messages.create(
            model      = Config.CLAUDE_MODEL,
            max_tokens = Config.CLAUDE_MAX_TOKENS,
            system     = SYSTEM_VISION,
            messages   = [{
                "role": "user",
                "content": [
                    {
                        "type"  : "image",
                        "source": {
                            "type"       : "base64",
                            "media_type" : media_type,
                            "data"       : b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    }
                ],
            }],
        )

        raw_response = message.content[0].text.strip()
        print(f"[AI] Vision response: {len(raw_response)} chars")
        return self._parse_response(raw_response, raw_text, subject)

    # ─────────────────────────────────────────────────────────────────────────
    #  Claude Text Reconstruction Path (fallback)
    # ─────────────────────────────────────────────────────────────────────────

    def _claude_text(
        self,
        corrected_text : str,
        raw_text       : str,
        subject        : str,
        page_label     : str,
        formulas       : list,
        settings       : dict,
    ) -> dict:
        """Text-only path: send corrected OCR text for Claude to reconstruct."""

        formula_list = "\n".join(
            f"  • {f.get('original','?')} → {f.get('unicode','?')}"
            for f in formulas[:15]
        ) if formulas else "None auto-detected."

        language = settings.get("language", "English")
        theme    = settings.get("theme", "academic")

        user_prompt = f"""PAGE: {page_label}
SUBJECT: {subject}
LANGUAGE: {language}
STYLE: {theme}

DETECTED FORMULAS:
{formula_list}

RAW OCR TEXT (garbled):
\"\"\"{raw_text[:2000]}\"\"\"

PRE-CORRECTED TEXT:
\"\"\"{corrected_text[:2000]}\"\"\"

Reconstruct the correct academic content from both versions above.
Use your knowledge of {subject} to fix all OCR errors.
If you detect a known formula/theorem, write it completely and correctly.

{JSON_SCHEMA}"""

        message = self._client.messages.create(
            model      = Config.CLAUDE_MODEL,
            max_tokens = Config.CLAUDE_MAX_TOKENS,
            system     = SYSTEM_TEXT_FALLBACK,
            messages   = [{"role": "user", "content": user_prompt}],
        )

        raw_response = message.content[0].text.strip()
        print(f"[AI] Text response: {len(raw_response)} chars")
        return self._parse_response(raw_response, corrected_text, subject)

    # ─────────────────────────────────────────────────────────────────────────
    #  Response Parser
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw: str, fallback_text: str, subject: str) -> dict:
        """Parse Claude's JSON with robust error handling."""
        # Strip markdown fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$",           "", cleaned.strip(), flags=re.MULTILINE)
        cleaned = cleaned.strip()

        data = None
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if not data:
            print("[AI] JSON parse failed — using basic formatter")
            return _basic_format(fallback_text, subject)

        # Sanitise title — remove any debug tags like [LIKELY FORMULA: ...]
        raw_title = str(data.get("title", "Untitled Notes"))
        title     = re.sub(r"^\[.*?\]\s*", "", raw_title).strip() or raw_title
        title     = title[:100]

        result = {
            "title"    : title,
            "subject"  : str(data.get("subject",  subject)),
            "summary"  : str(data.get("summary",  ""))[:600],
            "key_terms": [str(t) for t in data.get("key_terms", []) if t],
            "sections" : [],
        }

        for sec in data.get("sections", []):
            if not isinstance(sec, dict):
                continue
            result["sections"].append({
                "heading" : str(sec.get("heading",  "")),
                "content" : str(sec.get("content",  "")),
                "bullets" : [str(b) for b in sec.get("bullets",  []) if b],
                "formulas": [str(f) for f in sec.get("formulas", []) if f],
                "notes"   : str(sec.get("notes", "")),
            })

        return result