"""
config.py — Centralised configuration for AI Notes Digitizer
=============================================================
All settings, API keys, and paths live here.
Copy .env.example to .env and fill in your keys.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:
    # ── Flask ──────────────────────────────────────────────────────────────────
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production-abc123xyz")
    HOST       = os.getenv("HOST", "0.0.0.0")
    PORT       = int(os.getenv("PORT", 5000))
    DEBUG      = os.getenv("DEBUG", "true").lower() == "true"

    # ── API Keys ───────────────────────────────────────────────────────────────
    # Anthropic (Claude) — https://console.anthropic.com  [FREE $5 credit on signup]
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    # Google Cloud Vision (optional fallback OCR for complex scripts)
    # https://cloud.google.com/vision — FREE 1000 units/month
    GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "")

    # ── Model Selection ────────────────────────────────────────────────────────
    # claude-3-haiku-20240307   → fastest, cheapest  (~$0.00025/1K input tokens)
    # claude-3-5-sonnet-20241022 → best quality
    CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
    CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))

    # ── File Handling ──────────────────────────────────────────────────────────
    UPLOADS_DIR       = os.getenv("UPLOADS_DIR",  str(BASE_DIR / "uploads"))
    OUTPUTS_DIR       = os.getenv("OUTPUTS_DIR",  str(BASE_DIR / "outputs"))
    SESSIONS_DIR      = os.getenv("SESSIONS_DIR", str(BASE_DIR / "sessions"))
    MAX_CONTENT_MB    = int(os.getenv("MAX_CONTENT_MB", "50"))
    MAX_CONTENT_LENGTH = MAX_CONTENT_MB * 1024 * 1024
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "pdf"}

    # ── OCR Settings ───────────────────────────────────────────────────────────
    # Languages for EasyOCR (ISO 639-1 codes)
    # Full list: https://www.jaided.ai/easyocr/
    OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "en,hi").split(",")
    OCR_GPU       = os.getenv("OCR_GPU", "false").lower() == "true"

    # Tesseract binary path (auto-detected if in PATH)
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")

    # ── PDF Settings ──────────────────────────────────────────────────────────
    PDF_DPI      = int(os.getenv("PDF_DPI", "200"))     # Higher = better OCR, slower
    PDF_MAX_PAGES = int(os.getenv("PDF_MAX_PAGES", "100"))

    # ── Image Enhancement ─────────────────────────────────────────────────────
    ENHANCE_RESIZE_MAX = int(os.getenv("ENHANCE_RESIZE_MAX", "2400"))  # px on longest side
    ENHANCE_DEFAULT_MODE = os.getenv("ENHANCE_DEFAULT_MODE", "auto")

    # ── DOCX Themes ───────────────────────────────────────────────────────────
    DOCX_THEMES = {
        "academic": {
            "title_font"   : "Times New Roman",
            "body_font"    : "Times New Roman",
            "title_size"   : 18,
            "heading_size" : 13,
            "body_size"    : 11,
            "title_color"  : "1F3864",   # dark navy
            "heading_color": "2E4057",
            "accent_color" : "C00000",
        },
        "minimal": {
            "title_font"   : "Arial",
            "body_font"    : "Arial",
            "title_size"   : 17,
            "heading_size" : 12,
            "body_size"    : 10,
            "title_color"  : "000000",
            "heading_color": "555555",
            "accent_color" : "0070C0",
        },
        "professional": {
            "title_font"   : "Calibri",
            "body_font"    : "Calibri",
            "title_size"   : 18,
            "heading_size" : 13,
            "body_size"    : 11,
            "title_color"  : "203864",
            "heading_color": "2F5496",
            "accent_color" : "ED7D31",
        },
    }
    DEFAULT_THEME = "academic"

    # ── Session Retention ─────────────────────────────────────────────────────
    SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "24"))
