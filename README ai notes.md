# 🧠 AI Notes Digitizer

> Transform handwritten notes into perfectly structured digital documents using OCR, Computer Vision, and Claude AI.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![Claude](https://img.shields.io/badge/Claude-Sonnet-orange?style=flat-square)
![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## 📌 What is this?

AI Notes Digitizer is a full-stack Flask web app that takes your messy handwritten notes — photos, scans, or PDFs — and converts them into clean, structured, downloadable Word documents using a multi-stage AI/ML pipeline.

Upload a photo of your notebook. Get back a formatted DOCX with headings, bullet points, math formulas in Unicode, a 3-line summary, key terms highlighted, and subject auto-detected — all in seconds.

---

## ✨ Features

### 📥 Input Methods
- **Image Upload** — JPG, PNG, WEBP, BMP, TIFF via drag-drop or browse
- **PDF Upload** — Auto page-by-page splitting, up to 100 pages
- **Camera Capture** — Live webcam capture directly in browser
- **Batch Mode** — Queue multiple files, process all at once

### 🔬 AI/ML Pipeline (per page)
```
Input Image / PDF Page
        ↓
Image Enhancement (OpenCV)
  → EXIF rotation fix
  → Resize to 800–2400px
  → Deskew via Hough Lines (±15°)
  → Non-local Means Denoise
  → Adaptive Gaussian Threshold
  → Morphological Opening
  → CLAHE Contrast Boost (scan mode)
        ↓
Dual-Engine OCR
  → EasyOCR (deep learning, multi-language)
  → Tesseract (classical OCR, PSM3 + OEM3)
  → IoU-based Ensemble Merge (threshold 0.30)
  → Reading-order reconstruction (line grouping)
        ↓
Formula Detection
  → Stage 1: Regex (superscripts, Greek, operators)
  → Stage 2: pix2tex visual OCR (optional)
  → LaTeX → Unicode conversion (∫ Σ √ ² → ∞)
        ↓
Subject Detection
  → Weighted keyword scoring (10 subjects)
  → Math symbol density boost
  → Chemical formula regex detection
        ↓
Claude AI Formatter (claude-3-haiku / sonnet)
  → OCR error correction in context
  → Heading + bullet point structure
  → 3-sentence summary generation
  → Key terms extraction
  → Subject-specific formatting rules
  → JSON-structured output with fallback parsing
        ↓
DOCX Builder (python-docx)
  → 3 themes: Academic / Minimal / Professional
  → Cover page + Table of Contents
  → Per-page sections with formula blocks
  → Key terms bolded + color-coded
  → PDF export via LibreOffice headless
```

### 🎯 Smart Features
| Feature | Details |
|---|---|
| Subject Detection | Math, Physics, Chemistry, Biology, CS, History, Geography, Economics, Literature, Law |
| Formula Parsing | Two-stage: fast regex + optional pix2tex visual OCR |
| Multi-language OCR | English, Hindi, Bengali, Odia, Tamil, Telugu |
| Confidence Scoring | Per-page OCR accuracy % with color indicator 🟢🟡🔴 |
| Edit Mode | Click any paragraph in preview to manually correct before export |
| Re-scan | Retry individual pages with different enhancement modes |
| Session History | Last 10 processed documents saved, reloadable |
| Merge Mode | Combine multiple scan sessions into one DOCX |
| Split Download | Per-page DOCX or all-in-one |
| PDF Export | DOCX → PDF via LibreOffice headless |

### 🎨 DOCX Output Themes
| Theme | Fonts | Colors |
|---|---|---|
| Academic | Times New Roman | Navy + Dark Red |
| Minimal | Arial | Black + Blue |
| Professional | Calibri | Dark Blue + Orange |

### 📊 Live UI Features
- Real-time SSE progress stream (Enhance → OCR → Formula → AI → DOCX)
- Animated pipeline step indicators
- Page-by-page viewer with arrow key navigation
- Keyboard shortcuts: `←→` pages, `Ctrl+E` edit mode, `Ctrl+D` download
- Dark glassmorphic design (Space Grotesk + Manrope)

---

## 🧪 Tech Stack

### Backend
| Layer | Technology |
|---|---|
| Web Framework | Flask 3.0 + Flask-CORS |
| AI Formatter | Anthropic Claude API (claude-3-haiku / claude-3-5-sonnet) |
| Handwriting OCR | EasyOCR 1.7 (deep learning, MIT) |
| Classical OCR | Tesseract 5 via pytesseract |
| Image Processing | OpenCV (headless), NumPy, Pillow |
| PDF Handling | PyMuPDF (fitz) |
| Math Formula OCR | pix2tex / LaTeX-OCR (optional) |
| DOCX Generation | python-docx |
| PDF Export | LibreOffice headless |
| Session Storage | JSON file persistence |
| Progress Streaming | Server-Sent Events (SSE) |
| Server | Gunicorn (production) |

### Frontend
| Layer | Technology |
|---|---|
| Markup | HTML5 + Jinja2 templates |
| Styling | Tailwind CSS (CDN) + custom CSS variables |
| JavaScript | Vanilla JS (ES2022, no framework) |
| Icons | Google Material Symbols |
| Fonts | Space Grotesk + Manrope (Google Fonts) |
| Camera | MediaDevices Web API |
| Progress | EventSource (SSE) |

---

## 📁 Project Structure

```
AI-Notes-Digitizer/
├── backend/
│   ├── app.py                  # Flask server — 12 API routes + SSE streaming
│   ├── config.py               # Centralized config — API keys, paths, themes
│   ├── requirements.txt        # Python dependencies
│   ├── setup.sh                # One-command installer (Linux/macOS)
│   ├── .env.example            # Environment variable template
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── ocr_engine.py       # Dual-engine OCR + IoU ensemble merge
│   │   ├── image_enhancer.py   # OpenCV 9-step preprocessing pipeline
│   │   ├── pdf_handler.py      # PyMuPDF page splitter + auto-contrast
│   │   ├── formula_detector.py # Regex + pix2tex two-stage formula parser
│   │   ├── subject_detector.py # Keyword scoring for 10 academic subjects
│   │   ├── ai_formatter.py     # Claude API integration + fallback chain
│   │   ├── docx_builder.py     # Themed Word doc builder with 7 custom styles
│   │   └── session_manager.py  # JSON session persistence + auto-cleanup
│   ├── uploads/                # Raw uploaded files (auto-created)
│   ├── outputs/                # Generated DOCX/PDF (auto-created)
│   └── sessions/               # Session JSON files (auto-created)
│
└── frontend/
    ├── templates/
    │   └── index.html          # Landing page + full app overlay
    └── static/
        ├── css/
        │   └── style.css       # Design system — tokens, animations, components
        └── js/
            └── main.js         # All frontend logic + Flask API integration
```

---

## 🌐 API Endpoints

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serve main HTML page |
| `/upload` | POST | Accept image/PDF files → returns `file_ids` |
| `/process` | POST | Start full AI pipeline → returns `job_id` |
| `/stream/<job_id>` | GET | SSE progress stream |
| `/preview/<job_id>` | GET | Formatted page JSON for live preview |
| `/download/<job_id>` | GET | Download final DOCX |
| `/download-pdf/<job_id>` | GET | Download as PDF |
| `/reprocess/<job_id>` | POST | Re-run page with new enhancement settings |
| `/history` | GET | Return last 10 sessions |
| `/merge` | POST | Merge multiple session outputs into one DOCX |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+
- Tesseract OCR binary
- LibreOffice (optional, for PDF export)
- Anthropic API key — [get free $5 credit](https://console.anthropic.com)

### 1. Clone the repo
```bash
git clone https://github.com/Vaidik-7781/AI-Notes-Digitizer.git
cd AI-Notes-Digitizer
```

### 2. Create virtual environment
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract binary

**Windows:**
Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and install.

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-ben
```

**macOS:**
```bash
brew install tesseract
```

### 5. Configure environment
```bash
cp .env.example .env
```

Edit `.env`:
```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
CLAUDE_MODEL=claude-3-haiku-20240307
SECRET_KEY=your-secret-key-here
DEBUG=true
PORT=5000
```

### 6. Tell Flask where frontend lives
In `backend/app.py`, update Flask init:
```python
import os
BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE, '..', 'frontend', 'templates'),
    static_folder=os.path.join(BASE, '..', 'frontend', 'static'),
)
```

### 7. Run the server
```bash
cd backend
python app.py
```

Open browser: **http://localhost:5000**

---

## 🐧 Linux/macOS One-Command Setup
```bash
cd backend
bash setup.sh
```

Auto-installs Tesseract, creates venv, installs packages, sets up `.env`, creates all required directories.

---

## 🌍 Deployment

### Gunicorn (Production)
```bash
cd backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Variables for Production
```env
DEBUG=false
SECRET_KEY=strong-random-secret-key
ANTHROPIC_API_KEY=sk-ant-xxx
CLAUDE_MODEL=claude-3-5-sonnet-20241022
OCR_GPU=false
PDF_DPI=200
MAX_CONTENT_MB=50
SESSION_TTL_HOURS=24
```

### Deploy on Railway / Render
1. Push repo to GitHub
2. Connect to Railway or Render
3. Set root directory to `backend/`
4. Add environment variables in dashboard
5. Start command: `gunicorn -w 2 -b 0.0.0.0:$PORT app:app`

### Deploy on VPS (Ubuntu)
```bash
sudo apt update
sudo apt install python3.11 tesseract-ocr libreoffice -y

git clone https://github.com/Vaidik-7781/AI-Notes-Digitizer.git
cd AI-Notes-Digitizer/backend
pip install -r requirements.txt

gunicorn -w 4 -b 127.0.0.1:5000 app:app
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude API key (required) |
| `CLAUDE_MODEL` | `claude-3-haiku-20240307` | Haiku = fast/cheap, Sonnet = best quality |
| `CLAUDE_MAX_TOKENS` | `4096` | Max tokens per Claude response |
| `OCR_LANGUAGES` | `en,hi` | Comma-separated ISO 639-1 codes |
| `OCR_GPU` | `false` | Enable GPU for EasyOCR |
| `PDF_DPI` | `200` | PDF render resolution (higher = better OCR) |
| `PDF_MAX_PAGES` | `100` | Max pages per PDF |
| `ENHANCE_RESIZE_MAX` | `2400` | Max image dimension in pixels |
| `ENHANCE_DEFAULT_MODE` | `auto` | `auto` / `light` / `scan` |
| `MAX_CONTENT_MB` | `50` | Max upload size |
| `SESSION_TTL_HOURS` | `24` | Session auto-cleanup after N hours |
| `DEFAULT_THEME` | `academic` | `academic` / `minimal` / `professional` |

---

## 📝 OCR Tips for Best Results

- Use **even, diffuse lighting** — avoid harsh shadows
- Shoot **perpendicular** to page, not at an angle
- **Dark ink on white paper** gives best accuracy
- Minimum **200 DPI** for scanned documents
- Use **scan mode** for camera-captured pages
- Use **light mode** for already-clean digital scans

---

## 🔑 Getting a Free Claude API Key

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up → get **$5 free credit**
3. API Keys → Generate new key
4. Paste into `.env` as `ANTHROPIC_API_KEY`

With `claude-3-haiku` (default), $5 processes ~20,000 pages.

---

## 📦 Key Dependencies

```
flask>=3.0.0
anthropic>=0.25.0
easyocr>=1.7.0
pytesseract>=0.3.10
opencv-python-headless>=4.9.0
numpy>=1.26.0
Pillow>=10.3.0
pymupdf>=1.24.0
python-docx>=1.1.0
python-dotenv>=1.0.0
gunicorn>=21.0.0
```

---

## 🤝 Contributing

1. Fork the repo
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👨‍💻 Developer

**Vaidik Gupta**

B.Tech — Electronics & Computer Science Engineering
KIIT University, Bhubaneswar (Batch 2023–2027)

[![GitHub](https://img.shields.io/badge/GitHub-Vaidik--7781-black?style=flat-square&logo=github)](https://github.com/Vaidik-7781)

---

*Built with Flask + Claude AI + EasyOCR + OpenCV + python-docx*
