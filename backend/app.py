"""
AI Notes Digitizer — Flask Backend
====================================
Main application entry point. Handles all HTTP routes,
file uploads, job queuing, SSE streaming, and session management.
"""

import os
import uuid
import json
import time
import threading
import traceback
from pathlib import Path
from flask import (
    Flask, request, jsonify, send_file,
    Response, stream_with_context, render_template
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import Config
from modules.image_enhancer import ImageEnhancer
from modules.ocr_engine import OCREngine
from modules.pdf_handler import PDFHandler
from modules.formula_detector import FormulaDetector
from modules.ai_formatter import AIFormatter
from modules.docx_builder import DOCXBuilder
from modules.subject_detector import SubjectDetector
from modules.session_manager import SessionManager

# ─── App Bootstrap ────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = Config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
CORS(app, supports_credentials=True)

# ─── Module Singletons ────────────────────────────────────────────────────────
image_enhancer   = ImageEnhancer()
ocr_engine       = OCREngine()
pdf_handler      = PDFHandler()
formula_detector = FormulaDetector()
ai_formatter     = AIFormatter()
docx_builder     = DOCXBuilder()
subject_detector = SubjectDetector()
session_manager  = SessionManager(Config.SESSIONS_DIR)

# ─── In-memory job store  {job_id: {status, progress, pages, error}} ──────────
JOBS: dict = {}
JOBS_LOCK = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  Internal Helpers
# ══════════════════════════════════════════════════════════════════════════════

def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext in Config.ALLOWED_EXTENSIONS


def _job_update(job_id: str, **kwargs):
    """Thread-safe job state update."""
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def _process_pipeline(job_id: str, file_paths: list, settings: dict):
    """
    Full AI pipeline executed in a background thread.

    Stages per page:
      1. Image enhancement  (OpenCV: deskew, denoise, threshold)
      2. OCR               (EasyOCR + Tesseract ensemble)
      3. Formula detection  (regex + pix2tex / LaTeX-OCR)
      4. Subject detection  (keyword heuristics + Claude)
      5. Claude AI formatting (structure, bullets, headings, summary)
      6. DOCX section build

    After all pages:
      7. Assemble final DOCX (python-docx)
      8. Optional PDF export (LibreOffice headless)
    """
    try:
        all_pages   = []
        total_files = len(file_paths)

        _job_update(job_id,
                    status="running",
                    progress=0,
                    stage="Starting pipeline",
                    total_pages=0,
                    processed_pages=0)

        # ── Expand PDFs into individual page images ───────────────────────
        image_list = []   # [{path, origin, page_num}]

        for fi, fp in enumerate(file_paths):
            ext = Path(fp).suffix.lower()
            if ext == ".pdf":
                _job_update(job_id, stage=f"Splitting PDF {fi+1}/{total_files}")
                pages = pdf_handler.split_to_images(fp, Config.UPLOADS_DIR)
                image_list.extend(pages)
            else:
                image_list.append({
                    "path"    : fp,
                    "origin"  : Path(fp).name,
                    "page_num": 1
                })

        total_pages = len(image_list)
        _job_update(job_id, total_pages=total_pages, stage="Images ready")

        # ── Per-page processing ───────────────────────────────────────────
        for idx, img_info in enumerate(image_list):
            img_path   = img_info["path"]
            origin     = img_info["origin"]
            page_num   = img_info["page_num"]
            page_label = f"{origin} — page {page_num}"

            _job_update(job_id,
                        stage=f"Enhancing image [{idx+1}/{total_pages}]",
                        progress=int((idx / total_pages) * 85))

            # 1. Image enhancement
            enhanced_path = image_enhancer.enhance(
                img_path,
                output_dir=Config.UPLOADS_DIR,
                mode=settings.get("enhance_mode", "auto")
            )

            # 2. OCR
            _job_update(job_id, stage=f"Running OCR [{idx+1}/{total_pages}]")
            ocr_result  = ocr_engine.extract(enhanced_path)
            raw_text    = ocr_result["text"]
            confidence  = ocr_result["confidence"]
            word_boxes  = ocr_result["word_boxes"]

            # 3. Formula detection
            _job_update(job_id, stage=f"Detecting formulas [{idx+1}/{total_pages}]")
            formula_result       = formula_detector.detect(enhanced_path, raw_text)
            raw_text_with_formulas = formula_result["annotated_text"]
            formulas               = formula_result["formulas"]

            # 4. Subject detection
            _job_update(job_id, stage=f"Detecting subject [{idx+1}/{total_pages}]")
            subject = subject_detector.detect(raw_text_with_formulas)

            # 5. Claude AI formatting (Vision: reads image directly for best accuracy)
            _job_update(job_id, stage=f"AI formatting [{idx+1}/{total_pages}]")
            formatted = ai_formatter.format_page(
                raw_text   = raw_text_with_formulas,
                subject    = subject,
                page_label = page_label,
                formulas   = formulas,
                settings   = settings,
                image_path = img_path,   # original image for Claude Vision
            )

            page_data = {
                "page_index"    : idx,
                "page_label"    : page_label,
                "origin_file"   : origin,
                "page_num"      : page_num,
                "raw_text"      : raw_text,
                "confidence"    : confidence,
                "subject"       : subject,
                "formulas"      : formulas,
                "word_boxes"    : word_boxes,
                "formatted"     : formatted,
                "enhanced_image": enhanced_path,
            }
            all_pages.append(page_data)
            _job_update(job_id, processed_pages=idx + 1)

        # ── Build DOCX ────────────────────────────────────────────────────
        _job_update(job_id, stage="Building DOCX document", progress=90)
        docx_path = docx_builder.build(
            pages=all_pages,
            output_dir=Config.OUTPUTS_DIR,
            session_id=job_id,
            settings=settings
        )

        # ── Optional PDF export ───────────────────────────────────────────
        pdf_path = None
        if settings.get("export_pdf", False):
            _job_update(job_id, stage="Exporting to PDF", progress=96)
            pdf_path = docx_builder.export_pdf(docx_path)

        # ── Persist session ───────────────────────────────────────────────
        session_manager.save(job_id, {
            "job_id"    : job_id,
            "pages"     : all_pages,
            "docx_path" : docx_path,
            "pdf_path"  : pdf_path,
            "settings"  : settings,
            "created_at": time.time(),
        })

        _job_update(job_id,
                    status="done",
                    progress=100,
                    stage="Complete",
                    docx_path=docx_path,
                    pdf_path=pdf_path,
                    pages=all_pages)

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[PIPELINE ERROR] job={job_id}\n{tb}")
        _job_update(job_id, status="error", stage="Failed", error=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
#  Routes — Frontend
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ══════════════════════════════════════════════════════════════════════════════
#  Routes — API
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/upload", methods=["POST"])
def upload():
    """
    Accept one or more files (images / PDFs).
    Returns: {session_id, files: [{file_id, filename, type, size, dest}]}
    """
    if "files" not in request.files:
        return jsonify({"error": "No files field in request"}), 400

    uploaded = request.files.getlist("files")
    if not uploaded or all(f.filename == "" for f in uploaded):
        return jsonify({"error": "Empty file list"}), 400

    sess_id = str(uuid.uuid4())
    saved   = []

    for f in uploaded:
        if not f.filename:
            continue
        if not allowed_file(f.filename):
            return jsonify({"error": f"File type not allowed: {f.filename}"}), 400

        file_id = str(uuid.uuid4())
        safe_fn = secure_filename(f.filename)
        ext     = Path(safe_fn).suffix.lower()
        dest    = os.path.join(Config.UPLOADS_DIR, f"{file_id}{ext}")
        f.save(dest)

        saved.append({
            "file_id" : file_id,
            "filename": safe_fn,
            "dest"    : dest,
            "type"    : "pdf" if ext == ".pdf" else "image",
            "size"    : os.path.getsize(dest),
        })

    return jsonify({"session_id": sess_id, "files": saved}), 200


@app.route("/api/process", methods=["POST"])
def process():
    """
    Kick off the full AI pipeline in a background thread.
    Body: {session_id?, file_paths: [...], settings: {...}}
    Returns: {job_id}
    """
    body       = request.get_json(force=True)
    session_id = body.get("session_id") or str(uuid.uuid4())
    file_paths = body.get("file_paths", [])
    settings   = body.get("settings", {})

    if not file_paths:
        return jsonify({"error": "No file_paths provided"}), 400

    for fp in file_paths:
        if not os.path.isfile(fp):
            return jsonify({"error": f"File not found on server: {fp}"}), 400

    job_id = session_id
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status"         : "queued",
            "progress"       : 0,
            "stage"          : "Queued",
            "total_pages"    : 0,
            "processed_pages": 0,
            "pages"          : [],
            "error"          : None,
        }

    t = threading.Thread(
        target=_process_pipeline,
        args=(job_id, file_paths, settings),
        daemon=True
    )
    t.start()

    return jsonify({"job_id": job_id}), 202


@app.route("/api/status/<job_id>", methods=["GET"])
def status(job_id: str):
    """Poll-based job status — lightweight, pages excluded while running."""
    with JOBS_LOCK:
        job = dict(JOBS.get(job_id, {}))

    if not job:
        sess = session_manager.load(job_id)
        if sess:
            return jsonify({
                "status"         : "done",
                "progress"       : 100,
                "stage"          : "Complete (cached)",
                "total_pages"    : len(sess.get("pages", [])),
                "processed_pages": len(sess.get("pages", [])),
            })
        return jsonify({"error": "Job not found"}), 404

    resp = {
        "status"         : job["status"],
        "progress"       : job["progress"],
        "stage"          : job["stage"],
        "total_pages"    : job["total_pages"],
        "processed_pages": job["processed_pages"],
        "error"          : job.get("error"),
    }
    if job["status"] == "done":
        resp["docx_path"] = job.get("docx_path")
        resp["pdf_path"]  = job.get("pdf_path")

    return jsonify(resp), 200


@app.route("/api/stream/<job_id>", methods=["GET"])
def stream(job_id: str):
    """
    Server-Sent Events endpoint.
    Frontend: const es = new EventSource('/api/stream/<job_id>')
    """
    def generate():
        last_progress = -1
        while True:
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {}))

            if not job:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break

            progress = job.get("progress", 0)
            if progress != last_progress:
                payload = {
                    "status"         : job.get("status"),
                    "progress"       : progress,
                    "stage"          : job.get("stage"),
                    "total_pages"    : job.get("total_pages"),
                    "processed_pages": job.get("processed_pages"),
                    "error"          : job.get("error"),
                }
                yield f"data: {json.dumps(payload)}\n\n"
                last_progress = progress

            if job.get("status") in ("done", "error"):
                final = {
                    "status"  : job.get("status"),
                    "progress": 100,
                    "done"    : True,
                }
                yield f"data: {json.dumps(final)}\n\n"
                break

            time.sleep(0.4)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.route("/api/preview/<job_id>", methods=["GET"])
def preview(job_id: str):
    """
    Return structured page data for live preview panel.
    ?page=N   → single page (0-indexed)
    No param  → all pages
    """
    page_idx = request.args.get("page", type=int)

    with JOBS_LOCK:
        job = JOBS.get(job_id)

    pages = []
    if job:
        pages = job.get("pages", [])
    else:
        sess = session_manager.load(job_id)
        if sess:
            pages = sess.get("pages", [])
        else:
            return jsonify({"error": "Job not found"}), 404

    # Strip heavy word_boxes from response
    slim = [{k: v for k, v in p.items() if k != "word_boxes"} for p in pages]

    if page_idx is not None:
        if 0 <= page_idx < len(slim):
            return jsonify({"page": slim[page_idx]}), 200
        return jsonify({"error": "Page index out of range"}), 404

    return jsonify({"pages": slim, "total": len(slim)}), 200


@app.route("/api/edit/<job_id>/<int:page_idx>", methods=["POST"])
def edit_page(job_id: str, page_idx: int):
    """
    User manually corrects a page's formatted content.
    Body: {formatted: {title, summary, sections:[{heading, content}]}}
    Rebuilds DOCX automatically.
    """
    body          = request.get_json(force=True)
    new_formatted = body.get("formatted")

    if not new_formatted:
        return jsonify({"error": "No formatted data provided"}), 400

    sess = session_manager.load(job_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404

    pages = sess.get("pages", [])
    if page_idx >= len(pages):
        return jsonify({"error": "Page index out of range"}), 404

    pages[page_idx]["formatted"] = new_formatted
    docx_path = docx_builder.build(
        pages=pages,
        output_dir=Config.OUTPUTS_DIR,
        session_id=job_id,
        settings=sess.get("settings", {})
    )
    sess.update({"pages": pages, "docx_path": docx_path})
    session_manager.save(job_id, sess)

    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["pages"]     = pages
            JOBS[job_id]["docx_path"] = docx_path

    return jsonify({"message": "Page updated and DOCX rebuilt"}), 200


@app.route("/api/reprocess/<job_id>", methods=["POST"])
def reprocess(job_id: str):
    """
    Re-run the AI pipeline for a specific page.
    Body: {page_idx, settings}
    """
    body     = request.get_json(force=True)
    page_idx = body.get("page_idx", 0)
    settings = body.get("settings", {})

    sess = session_manager.load(job_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404

    pages = sess.get("pages", [])
    if page_idx >= len(pages):
        return jsonify({"error": "Page index out of range"}), 404

    page     = pages[page_idx]
    img_path = page.get("enhanced_image") or page.get("original_image")

    if not img_path or not os.path.isfile(img_path):
        return jsonify({"error": "Source image not found for reprocessing"}), 404

    ocr_result   = ocr_engine.extract(img_path)
    raw_text     = ocr_result["text"]
    fml_result   = formula_detector.detect(img_path, raw_text)
    raw_w_formulas = fml_result["annotated_text"]
    subject      = subject_detector.detect(raw_w_formulas)
    formatted    = ai_formatter.format_page(
        raw_text   = raw_w_formulas,
        subject    = subject,
        page_label = page["page_label"],
        formulas   = fml_result["formulas"],
        settings   = settings,
        image_path = img_path,
    )

    pages[page_idx].update({
        "raw_text" : raw_text,
        "formulas" : fml_result["formulas"],
        "subject"  : subject,
        "formatted": formatted,
        "confidence": ocr_result["confidence"],
    })

    docx_path = docx_builder.build(
        pages=pages,
        output_dir=Config.OUTPUTS_DIR,
        session_id=job_id,
        settings=sess.get("settings", {})
    )
    sess.update({"pages": pages, "docx_path": docx_path})
    session_manager.save(job_id, sess)

    return jsonify({
        "message": "Page reprocessed",
        "page"   : {k: v for k, v in pages[page_idx].items() if k != "word_boxes"},
    }), 200


@app.route("/api/download/<job_id>", methods=["GET"])
def download_docx(job_id: str):
    """Download the generated DOCX."""
    docx_path = None
    sess = session_manager.load(job_id)
    if sess:
        docx_path = sess.get("docx_path")

    if not docx_path:
        with JOBS_LOCK:
            job = JOBS.get(job_id, {})
        docx_path = job.get("docx_path")

    if not docx_path or not os.path.isfile(docx_path):
        return jsonify({"error": "DOCX not found — process not yet complete?"}), 404

    return send_file(
        docx_path,
        as_attachment=True,
        download_name="AI_Notes.docx",
        mimetype=(
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        )
    )


@app.route("/api/download-pdf/<job_id>", methods=["GET"])
def download_pdf(job_id: str):
    """Download as PDF (auto-generates from DOCX if not cached)."""
    sess = session_manager.load(job_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404

    pdf_path  = sess.get("pdf_path")
    docx_path = sess.get("docx_path")

    if not pdf_path or not os.path.isfile(str(pdf_path)):
        if not docx_path or not os.path.isfile(str(docx_path)):
            return jsonify({"error": "DOCX not found, cannot export PDF"}), 404
        pdf_path = docx_builder.export_pdf(docx_path)
        sess["pdf_path"] = pdf_path
        session_manager.save(job_id, sess)

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name="AI_Notes.pdf",
        mimetype="application/pdf"
    )


@app.route("/api/merge", methods=["POST"])
def merge():
    """
    Merge multiple session outputs into one DOCX.
    Body: {job_ids: [...], settings: {...}}
    """
    body     = request.get_json(force=True)
    job_ids  = body.get("job_ids", [])
    settings = body.get("settings", {})

    if len(job_ids) < 2:
        return jsonify({"error": "Provide at least 2 job_ids to merge"}), 400

    all_pages = []
    for jid in job_ids:
        sess = session_manager.load(jid)
        if sess:
            all_pages.extend(sess.get("pages", []))

    if not all_pages:
        return jsonify({"error": "No pages found in provided sessions"}), 404

    merged_id = str(uuid.uuid4())
    docx_path = docx_builder.build(
        pages=all_pages,
        output_dir=Config.OUTPUTS_DIR,
        session_id=merged_id,
        settings=settings
    )
    session_manager.save(merged_id, {
        "job_id"    : merged_id,
        "pages"     : all_pages,
        "docx_path" : docx_path,
        "settings"  : settings,
        "created_at": time.time(),
    })

    return jsonify({
        "merged_job_id": merged_id,
        "total_pages"  : len(all_pages),
    }), 200


@app.route("/api/history", methods=["GET"])
def history():
    """Return metadata for recent sessions."""
    limit   = request.args.get("limit", 10, type=int)
    records = session_manager.list_recent(limit)
    result  = []
    for r in records:
        pages = r.get("pages", [])
        result.append({
            "job_id"      : r.get("job_id"),
            "created_at"  : r.get("created_at"),
            "total_pages" : len(pages),
            "subjects"    : list({p.get("subject", "Unknown") for p in pages}),
        })
    return jsonify({"history": result}), 200


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status"       : "ok",
        "ocr_ready"    : ocr_engine.is_ready(),
        "formula_ready": formula_detector.is_ready(),
        "claude_ready" : ai_formatter.is_ready(),
        "active_jobs"  : len([j for j in JOBS.values()
                               if j.get("status") == "running"]),
    }), 200


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(Config.UPLOADS_DIR,  exist_ok=True)
    os.makedirs(Config.OUTPUTS_DIR,  exist_ok=True)
    os.makedirs(Config.SESSIONS_DIR, exist_ok=True)
    print("🧠 AI Notes Digitizer — backend starting")
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG, threaded=True)