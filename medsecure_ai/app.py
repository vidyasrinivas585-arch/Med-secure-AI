"""
app.py
MedSecure AI — Flask web application entry point.
Counterfeit Medicine Detection System.

Routes:
    /               → Home page
    /upload         → Image upload page
    /analyze        → POST: run full analysis pipeline
    /result/<id>    → Show result for a specific report ID
    /history        → Show all past reports
    /search         → AJAX search endpoint
    /download/<id>  → Generate and download PDF report
    /api/stats      → JSON statistics for dashboard
"""

import os
import uuid
import logging
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, send_file, flash, session
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "medsecure_dev_secret")

UPLOAD_FOLDER  = os.path.join("static", "uploads")
ALLOWED_EXTS   = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_MB = 10

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join("static", "reports"), exist_ok=True)

app.config["UPLOAD_FOLDER"]    = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── Module imports (lazy where heavy) ─────────────────────────────────────────
from database.mongodb       import db
from ocr.preprocessor       import preprocess_for_model, analyze_packaging
from ocr.extractor          import extract_medicine_info
from model.decision_engine  import full_analysis
from model.translator       import translate_results
from reports.pdf_generator  import generate_pdf_report


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def save_uploaded_file(file) -> str:
    """Save uploaded file with a unique name; return the saved path."""
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Home / landing page."""
    stats = db.get_statistics()
    return render_template("index.html", stats=stats)


@app.route("/upload")
def upload():
    """Upload page — user selects image + language."""
    return render_template("upload.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main analysis pipeline:
        1. Validate & save uploaded image
        2. Preprocess with OpenCV
        3. OCR extraction (EasyOCR)
        4. Packaging analysis (OpenCV)
        5. AI model prediction (MobileNetV2)
        6. Decision engine → score, risk, recommendation
        7. Language translation (if Kannada)
        8. MongoDB Atlas storage
        9. Redirect to results page
    """
    # ── Validate file ─────────────────────────────────────────────────────
    if "image" not in request.files:
        flash("No image file provided.", "danger")
        return redirect(url_for("upload"))

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        flash("Invalid file. Please upload a JPG, PNG, or WebP image.", "danger")
        return redirect(url_for("upload"))

    language = request.form.get("language", "en")

    # ── Save image ────────────────────────────────────────────────────────
    image_path = save_uploaded_file(file)
    logger.info(f"Image saved: {image_path}")

    try:
        # ── Preprocessing ─────────────────────────────────────────────────
        preprocessed = preprocess_for_model(image_path)
        if preprocessed is None:
            flash("Image preprocessing failed. Please try a clearer image.", "warning")
            return redirect(url_for("upload"))

        # ── OCR ───────────────────────────────────────────────────────────
        ocr_info = extract_medicine_info(image_path)
        logger.info(f"OCR extracted: {ocr_info}")

        # ── Packaging analysis ────────────────────────────────────────────
        pkg_info = analyze_packaging(image_path)

        # ── AI + Decision ─────────────────────────────────────────────────
        analysis = full_analysis(preprocessed, ocr_info, pkg_info)

        # ── Build result document ─────────────────────────────────────────
        report = {
            # OCR fields
            "medicine_name":      ocr_info.get("medicine_name", "Not Detected"),
            "manufacturer":       ocr_info.get("manufacturer", "Not Detected"),
            "batch_number":       ocr_info.get("batch_number", "Not Detected"),
            "manufacturing_date": ocr_info.get("manufacturing_date", "N/A"),
            "expiry_date":        ocr_info.get("expiry_date", "N/A"),
            "expiry_status":      ocr_info.get("expiry_status", "Unknown"),
            "days_remaining":     ocr_info.get("days_remaining", 0),
            "months_remaining":   ocr_info.get("months_remaining", 0),
            "raw_text":           ocr_info.get("raw_text", ""),
            # Analysis fields
            "prediction":         analysis["prediction"],
            "ai_confidence":      analysis["ai_confidence"],
            "ocr_confidence":     analysis["ocr_confidence"],
            "packaging_score":    analysis["packaging_score"],
            "authenticity_score": analysis["authenticity_score"],
            "risk_level":         analysis["risk_level"],
            "recommendation":     analysis["recommendation"],
            # Metadata
            "language":           language,
            "image_path":         image_path,
        }

        # ── Language translation ───────────────────────────────────────────
        translated_report = translate_results(report, language)

        # ── Store in MongoDB ──────────────────────────────────────────────
        report_id = db.insert_report(translated_report)
        if report_id is None:
            # Store in session as fallback if DB is down
            session["last_report"] = translated_report
            session["last_report_id"] = "session"
        else:
            session["last_report_id"] = report_id

        return redirect(url_for("result", report_id=report_id or "session"))

    except Exception as e:
        logger.error(f"Analysis pipeline error: {e}", exc_info=True)
        flash(f"Analysis failed: {str(e)}", "danger")
        return redirect(url_for("upload"))


@app.route("/result/<report_id>")
def result(report_id: str):
    """Display the analysis result for a given report."""
    if report_id == "session":
        report = session.get("last_report", {})
    else:
        report = db.get_report_by_id(report_id)

    if not report:
        flash("Report not found.", "warning")
        return redirect(url_for("index"))

    return render_template("result.html", report=report, report_id=report_id)


@app.route("/history")
def history():
    """Display all past analysis reports."""
    reports = db.get_all_reports(limit=50)
    stats   = db.get_statistics()
    return render_template("history.html", reports=reports, stats=stats)


@app.route("/search")
def search():
    """AJAX search endpoint — returns JSON list of matching reports."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    results = db.search_reports(query)
    return jsonify(results)


@app.route("/download/<report_id>")
def download_report(report_id: str):
    """Generate and serve a PDF report for a given analysis."""
    if report_id == "session":
        report = session.get("last_report", {})
    else:
        report = db.get_report_by_id(report_id)

    if not report:
        flash("Report not found.", "warning")
        return redirect(url_for("history"))

    image_path = report.get("image_path", None)
    pdf_path   = generate_pdf_report(report, image_path)

    if not pdf_path or not os.path.exists(pdf_path):
        flash("PDF generation failed. Please try again.", "danger")
        return redirect(url_for("result", report_id=report_id))

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=f"MedSecure_Report_{report_id[:8]}.pdf",
        mimetype="application/pdf",
    )


@app.route("/api/stats")
def api_stats():
    """JSON endpoint for live dashboard statistics."""
    return jsonify(db.get_statistics())


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", error="Page not found."), 404


@app.errorhandler(413)
def too_large(e):
    flash(f"File too large. Maximum size is {MAX_CONTENT_MB} MB.", "danger")
    return redirect(url_for("upload"))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=os.getenv("FLASK_DEBUG", "True") == "True",
    )
