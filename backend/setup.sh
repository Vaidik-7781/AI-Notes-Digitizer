#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
#  AI Notes Digitizer — One-command setup script
#  Usage: bash setup.sh
# ══════════════════════════════════════════════════════════════════════════

set -e
echo "🧠 AI Notes Digitizer — Setup"
echo "=============================="

# ── Detect OS ──────────────────────────────────────────────────────────────
OS="$(uname -s)"

# ── Python check ──────────────────────────────────────────────────────────
echo ""
echo "► Checking Python..."
python3 --version || { echo "❌ Python 3 not found. Install from https://python.org"; exit 1; }

# ── Virtualenv ────────────────────────────────────────────────────────────
echo ""
echo "► Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# ── pip upgrade ───────────────────────────────────────────────────────────
pip install --upgrade pip --quiet

# ── Python dependencies ───────────────────────────────────────────────────
echo ""
echo "► Installing Python packages..."
pip install -r requirements.txt

# ── Tesseract binary ──────────────────────────────────────────────────────
echo ""
echo "► Checking Tesseract OCR binary..."

if command -v tesseract &>/dev/null; then
    echo "   ✅ Tesseract found: $(tesseract --version 2>&1 | head -1)"
else
    echo "   ⚠️  Tesseract not found."
    if [ "$OS" = "Linux" ]; then
        echo "   Installing via apt..."
        sudo apt-get update -qq
        sudo apt-get install -y tesseract-ocr \
            tesseract-ocr-hin \
            tesseract-ocr-ben \
            tesseract-ocr-ori \
            tesseract-ocr-tam \
            tesseract-ocr-tel
    elif [ "$OS" = "Darwin" ]; then
        echo "   Installing via brew..."
        brew install tesseract
    else
        echo "   Please install manually: https://github.com/UB-Mannheim/tesseract/wiki"
    fi
fi

# ── LibreOffice (optional PDF export) ─────────────────────────────────────
echo ""
echo "► Checking LibreOffice (for PDF export)..."
if command -v libreoffice &>/dev/null || command -v soffice &>/dev/null; then
    echo "   ✅ LibreOffice found"
else
    echo "   ⚠️  LibreOffice not found (PDF export will be unavailable)"
    if [ "$OS" = "Linux" ]; then
        echo "   Install with: sudo apt-get install libreoffice"
    elif [ "$OS" = "Darwin" ]; then
        echo "   Install with: brew install --cask libreoffice"
    fi
fi

# ── .env setup ────────────────────────────────────────────────────────────
echo ""
echo "► Setting up .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "   ✅ Created .env from .env.example"
    echo "   ⚠️  IMPORTANT: Edit .env and add your ANTHROPIC_API_KEY"
    echo "      Get a free key at: https://console.anthropic.com"
else
    echo "   ✅ .env already exists"
fi

# ── Create directories ────────────────────────────────────────────────────
echo ""
echo "► Creating directories..."
mkdir -p uploads outputs sessions static templates
echo "   ✅ uploads/ outputs/ sessions/ static/ templates/"

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and set your ANTHROPIC_API_KEY"
echo "     (Get free key: https://console.anthropic.com)"
echo ""
echo "  2. Activate the virtualenv:"
echo "     source venv/bin/activate"
echo ""
echo "  3. Start the server:"
echo "     python app.py"
echo ""
echo "  4. Open in browser:"
echo "     http://localhost:5000"
echo "══════════════════════════════════════════════════════"
