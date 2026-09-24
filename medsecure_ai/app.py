"""
app.py
MedSecure AI - Flask Application
FIXED: rec_dict passed directly in report, not via session
UPDATED: Location latitude and longitude support
"""

import os
import uuid
import logging
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, send_file, flash, session
)
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "medsecure_dev_secret")

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_MB = 10

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join("static", "reports"), exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

from database.mongodb import db
from ocr.preprocessor import preprocess_for_model, analyze_packaging
from ocr.extractor import extract_medicine_info
from model.decision_engine import full_analysis
from model.translator import translate_results
from reports.pdf_generator import generate_pdf_report
from ocr.qr_scanner import scan_qr_code, check_expiry_from_qr


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rec_to_str(rec) -> str:
    """Extract plain string from recommendation dict or string."""
    if isinstance(rec, dict):
        return rec.get("message", str(rec))
    return str(rec) if rec else ""


def allowed_file(filename: str) -> bool:
    """Check whether uploaded file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS
    )


def save_uploaded_file(file) -> str:
    """Save uploaded image with a unique filename."""
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"

    save_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )

    file.save(save_path)

    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Home
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():

    stats = db.get_statistics()

    return render_template(
        "index.html",
        stats=stats
    )


# ─────────────────────────────────────────────────────────────────────────────
# Upload Page
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/upload")
def upload():

    return render_template("upload.html")


# ─────────────────────────────────────────────────────────────────────────────
# Medicine Image Analysis
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():

    # Check image
    if "image" not in request.files:

        flash(
            "No image file provided.",
            "danger"
        )

        return redirect(url_for("upload"))


    file = request.files["image"]


    # Validate image
    if (
        file.filename == ""
        or not allowed_file(file.filename)
    ):

        flash(
            "Invalid file. Please upload a JPG, PNG, or WebP image.",
            "danger"
        )

        return redirect(url_for("upload"))


    # ─────────────────────────────────────────────────────────────────────
    # Form data
    # ─────────────────────────────────────────────────────────────────────

    language = request.form.get(
        "language",
        "en"
    )


    # Location received from upload.html
    latitude = request.form.get(
        "latitude",
        ""
    )

    longitude = request.form.get(
        "longitude",
        ""
    )


    logger.info(
        f"Location received: latitude={latitude}, longitude={longitude}"
    )


    # Save image
    image_path = save_uploaded_file(file)

    logger.info(
        f"Image saved: {image_path}"
    )


    try:

        # ─────────────────────────────────────────────────────────────────
        # Image preprocessing
        # ─────────────────────────────────────────────────────────────────

        preprocessed = preprocess_for_model(
            image_path
        )


        if preprocessed is None:

            flash(
                "Image preprocessing failed.",
                "warning"
            )

            return redirect(
                url_for("upload")
            )


        # ─────────────────────────────────────────────────────────────────
        # OCR
        # ─────────────────────────────────────────────────────────────────

        ocr_info = extract_medicine_info(
            image_path
        )


        # ─────────────────────────────────────────────────────────────────
        # Packaging analysis
        # ─────────────────────────────────────────────────────────────────

        pkg_info = analyze_packaging(
            image_path
        )


        # ─────────────────────────────────────────────────────────────────
        # AI analysis
        # ─────────────────────────────────────────────────────────────────

        analysis = full_analysis(
            preprocessed,
            ocr_info,
            pkg_info
        )


        # ─────────────────────────────────────────────────────────────────
        # Recommendation
        # ─────────────────────────────────────────────────────────────────

        raw_rec = analysis.get(
            "recommendation",
            ""
        )


        rec_dict = (
            raw_rec
            if isinstance(raw_rec, dict)
            else None
        )


        rec_str = _rec_to_str(
            raw_rec
        )


        # If recommendation is not available,
        # generate it using recommendation engine.

        if rec_dict is None:

            try:

                from model.recommendation_engine import (
                    generate_recommendation
                )


                medicine_name = (
                    ocr_info.get(
                        "medicine_name",
                        "Medicine"
                    )
                    or "Medicine"
                )


                rec_dict = generate_recommendation(

                    prediction=analysis.get(
                        "prediction",
                        "Unknown"
                    ),

                    ai_confidence=analysis.get(
                        "ai_confidence",
                        0
                    ),

                    authenticity_score=analysis.get(
                        "authenticity_score",
                        0
                    ),

                    expiry_status=ocr_info.get(
                        "expiry_status",
                        "Unknown"
                    ),

                    ocr_confidence=analysis.get(
                        "ocr_confidence",
                        0
                    ),

                    packaging_score=analysis.get(
                        "packaging_score",
                        0
                    ),

                    medicine_name=medicine_name,
                )


                rec_str = rec_dict.get(
                    "message",
                    rec_str
                )


                logger.info(
                    f"Generated rec_dict: "
                    f"type={rec_dict.get('type')} "
                    f"alt={rec_dict.get('alternative_medicine')}"
                )


            except Exception as e:

                logger.error(
                    f"Recommendation engine failed: {e}"
                )

                rec_dict = None


        # ─────────────────────────────────────────────────────────────────
        # Build report
        # ─────────────────────────────────────────────────────────────────

        report = {

            # OCR fields

            "medicine_name": ocr_info.get(
                "medicine_name",
                "Not Detected"
            ),

            "manufacturer": ocr_info.get(
                "manufacturer",
                "Not Detected"
            ),

            "batch_number": ocr_info.get(
                "batch_number",
                "Not Detected"
            ),

            "manufacturing_date": ocr_info.get(
                "manufacturing_date",
                "N/A"
            ),

            "expiry_date": ocr_info.get(
                "expiry_date",
                "N/A"
            ),

            "expiry_status": ocr_info.get(
                "expiry_status",
                "Unknown"
            ),

            "days_remaining": ocr_info.get(
                "days_remaining",
                0
            ),

            "months_remaining": ocr_info.get(
                "months_remaining",
                0
            ),

            "raw_text": ocr_info.get(
                "raw_text",
                ""
            ),


            # AI analysis fields

            "prediction": analysis.get(
                "prediction",
                "Unknown"
            ),

            "raw_prediction": analysis.get(
                "raw_prediction",
                "Unknown"
            ),

            "ai_confidence": round(
                float(
                    analysis.get(
                        "ai_confidence",
                        0
                    )
                ),
                2
            ),

            "ocr_confidence": round(
                float(
                    analysis.get(
                        "ocr_confidence",
                        0
                    )
                ),
                2
            ),

            "packaging_score": round(
                float(
                    analysis.get(
                        "packaging_score",
                        0
                    )
                ),
                2
            ),

            "authenticity_score": round(
                float(
                    analysis.get(
                        "authenticity_score",
                        0
                    )
                ),
                2
            ),

            "risk_level": analysis.get(
                "risk_level",
                "High"
            ),

            "recommendation": rec_str,


            # Rich fields

            "safety_guidance": analysis.get(
                "safety_guidance",
                {}
            ),

            "explanation": analysis.get(
                "explanation",
                {}
            ),

            "show_pharmacy_locator": analysis.get(
                "show_pharmacy_locator",
                False
            ),


            # Metadata

            "language": language,

            "image_path": image_path,


            # ─────────────────────────────────────────────────────────────
            # Location
            # ─────────────────────────────────────────────────────────────

            "latitude": latitude,

            "longitude": longitude,
        }


        # ─────────────────────────────────────────────────────────────────
        # Translate report
        # ─────────────────────────────────────────────────────────────────

        translated_report = translate_results(
            report,
            language
        )


        # ─────────────────────────────────────────────────────────────────
        # Store report in MongoDB
        # ─────────────────────────────────────────────────────────────────

        report_id = db.insert_report(
            translated_report
        )


        if report_id is None:

            session["last_report"] = translated_report

            report_id = "session"

        else:

            session["last_report_id"] = report_id


        # ─────────────────────────────────────────────────────────────────
        # Store recommendation in session
        # ─────────────────────────────────────────────────────────────────

        if rec_dict:

            session["rec_dict"] = rec_dict

            session.modified = True


        logger.info(
            f"rec_dict stored in session: "
            f"{rec_dict is not None}"
        )


        logger.info(
            f"alt medicine: "
            f"{rec_dict.get('alternative_medicine') if rec_dict else 'None'}"
        )


        logger.info(
            f"Analysis completed successfully. "
            f"Report ID: {report_id}"
        )


        return redirect(
            url_for(
                "result",
                report_id=report_id
            )
        )


    except Exception as e:

        logger.error(
            f"Analysis error: {e}",
            exc_info=True
        )


        flash(
            f"Analysis failed: {str(e)}",
            "danger"
        )


        return redirect(
            url_for("upload")
        )


# ─────────────────────────────────────────────────────────────────────────────
# Result Page
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/result/<report_id>")
def result(report_id: str):

    if report_id == "session":

        report = session.get(
            "last_report",
            {}
        )

    else:

        report = db.get_report_by_id(
            report_id
        )


    if not report:

        flash(
            "Report not found.",
            "warning"
        )

        return redirect(
            url_for("index")
        )


    # Get recommendation from session

    rec_dict = session.get(
        "rec_dict",
        None
    )


    # Regenerate recommendation if missing

    if rec_dict is None:

        try:

            from model.recommendation_engine import (
                generate_recommendation
            )


            medicine_name = (
                report.get(
                    "medicine_name",
                    "Medicine"
                )
                or "Medicine"
            )


            rec_dict = generate_recommendation(

                prediction=report.get(
                    "prediction",
                    "Unknown"
                ),

                ai_confidence=report.get(
                    "ai_confidence",
                    0
                ),

                authenticity_score=report.get(
                    "authenticity_score",
                    0
                ),

                expiry_status=report.get(
                    "expiry_status",
                    "Unknown"
                ),

                ocr_confidence=report.get(
                    "ocr_confidence",
                    0
                ),

                packaging_score=report.get(
                    "packaging_score",
                    0
                ),

                medicine_name=medicine_name,
            )


            logger.info(
                f"Regenerated rec_dict for report {report_id}"
            )


        except Exception as e:

            logger.error(
                f"Could not regenerate rec_dict: {e}"
            )

            rec_dict = None


    logger.info(
        f"Result page — rec_dict: "
        f"{rec_dict is not None}, "
        f"alt: "
        f"{rec_dict.get('alternative_medicine') if rec_dict else 'None'}"
    )


    return render_template(
        "result.html",
        report=report,
        report_id=report_id,
        rec_dict=rec_dict
    )


# ─────────────────────────────────────────────────────────────────────────────
# Update Report Field
# ─────────────────────────────────────────────────────────────────────────────

@app.route(
    "/report/<report_id>/update-field",
    methods=["POST"]
)
def update_report_field(report_id: str):

    data = request.get_json(
        silent=True
    ) or {}


    field = data.get(
        "field"
    )

    value = data.get(
        "value"
    )


    if not field or value is None:

        return jsonify({
            "success": False,
            "error": "Missing field/value"
        }), 400


    ok = db.update_report(
        report_id,
        {
            field: value
        }
    )


    return jsonify({
        "success": ok
    })


# ─────────────────────────────────────────────────────────────────────────────
# History
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/history")
def history():

    prediction_filter = request.args.get(
        "prediction",
        ""
    ).strip()


    risk_filter = request.args.get(
        "risk",
        ""
    ).strip()


    reports = db.get_all_reports(
        limit=50
    )


    if prediction_filter:

        reports = [
            r for r in reports
            if r.get("prediction") == prediction_filter
        ]


    if risk_filter:

        reports = [
            r for r in reports
            if r.get("risk_level") == risk_filter
        ]


    stats = db.get_statistics()


    return render_template(
        "history.html",
        reports=reports,
        stats=stats,
        prediction_filter=prediction_filter,
        risk_filter=risk_filter
    )


# ─────────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/search")
def search():

    query = request.args.get(
        "q",
        ""
    ).strip()


    if not query:

        return jsonify([])


    return jsonify(
        db.search_reports(query)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Download PDF Report
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/download/<report_id>")
def download_report(report_id: str):

    if report_id == "session":

        report = session.get(
            "last_report",
            {}
        )

    else:

        report = db.get_report_by_id(
            report_id
        )


    if not report:

        flash(
            "Report not found.",
            "warning"
        )

        return redirect(
            url_for("history")
        )


    pdf_path = generate_pdf_report(
        report,
        report.get("image_path")
    )


    if (
        not pdf_path
        or not os.path.exists(pdf_path)
    ):

        flash(
            "PDF generation failed.",
            "danger"
        )

        return redirect(
            url_for(
                "result",
                report_id=report_id
            )
        )


    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=(
            f"MedSecure_Report_"
            f"{report_id[:8]}.pdf"
        ),
        mimetype="application/pdf"
    )


# ─────────────────────────────────────────────────────────────────────────────
# API Statistics
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():

    return jsonify(
        db.get_statistics()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Camera Scan
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/camera-scan")
def camera_scan():

    return render_template(
        "camera_scan.html"
    )


# ─────────────────────────────────────────────────────────────────────────────
# QR Scan Page
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/scan-qr")
def scan_qr_page():

    return render_template(
        "scan_qr.html"
    )


# ─────────────────────────────────────────────────────────────────────────────
# QR Analysis
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/analyze-qr", methods=["POST"])
def analyze_qr():

    if "image" not in request.files:

        flash(
            "No image file provided.",
            "danger"
        )

        return redirect(
            url_for("scan_qr_page")
        )


    file = request.files["image"]


    if (
        file.filename == ""
        or not allowed_file(file.filename)
    ):

        flash(
            "Invalid file.",
            "danger"
        )

        return redirect(
            url_for("scan_qr_page")
        )


    language = request.form.get(
        "language",
        "en"
    )


    image_path = save_uploaded_file(
        file
    )


    try:

        # QR scan

        qr_result = scan_qr_code(
            image_path
        )


        # Image preprocessing

        preprocessed = preprocess_for_model(
            image_path
        )


        # Packaging

        pkg_info = analyze_packaging(
            image_path
        )


        # Expiry

        expiry_info = check_expiry_from_qr(
            qr_result["parsed"]
        )


        # Default values

        ai_prediction = "Unknown"

        ai_confidence = 50.0

        risk_level = "Medium"

        authenticity = 50.0

        rec_str = (
            "Unable to determine. "
            "Please verify manually."
        )

        rec_dict = None


        # AI prediction

        if preprocessed is not None:

            from model.decision_engine import (
                predict_image,
                compute_authenticity_score,
                assess_risk
            )


            ai_prediction, ai_confidence = predict_image(
                preprocessed
            )


            packaging_score = pkg_info.get(
                "packaging_score",
                50.0
            )


            ocr_conf = (
                95.0
                if qr_result["found"]
                else 40.0
            )


            authenticity = compute_authenticity_score(

                ai_confidence,
                ocr_conf,
                packaging_score,
                ai_prediction
            )


            risk_level = assess_risk(

                authenticity,
                ai_prediction,
                expiry_info.get(
                    "expiry_status",
                    "Unknown"
                )
            )


        # Recommendation

        try:

            from model.recommendation_engine import (
                generate_recommendation as ext_rec
            )


            medicine_name = (
                qr_result["parsed"].get(
                    "medicine_name",
                    "Medicine"
                )
                or "Medicine"
            )


            rec_dict = ext_rec(

                prediction=ai_prediction,

                ai_confidence=ai_confidence,

                authenticity_score=authenticity,

                expiry_status=expiry_info.get(
                    "expiry_status",
                    "Unknown"
                ),

                ocr_confidence=(
                    95.0
                    if qr_result["found"]
                    else 40.0
                ),

                packaging_score=pkg_info.get(
                    "packaging_score",
                    50.0
                ),

                medicine_name=medicine_name,
            )


            if isinstance(
                rec_dict,
                dict
            ):

                rec_str = rec_dict.get(
                    "message",
                    rec_str
                )

            else:

                rec_str = str(
                    rec_dict
                )


        except Exception as e:

            logger.error(
                f"QR recommendation engine failed: {e}"
            )


        # Parsed QR data

        parsed = qr_result["parsed"]


        # Build QR report

        report = {

            "medicine_name": parsed.get(
                "medicine_name",
                "Not Detected"
            ),

            "manufacturer": parsed.get(
                "manufacturer",
                "Not Detected"
            ),

            "batch_number": parsed.get(
                "batch_number",
                "Not Detected"
            ),

            "manufacturing_date": parsed.get(
                "manufacturing_date",
                "N/A"
            ),

            "expiry_date": parsed.get(
                "expiry_date",
                "N/A"
            ),

            "expiry_status": expiry_info.get(
                "expiry_status",
                "Unknown"
            ),

            "days_remaining": expiry_info.get(
                "days_remaining",
                0
            ),

            "months_remaining": expiry_info.get(
                "months_remaining",
                0
            ),

            "serial_number": parsed.get(
                "serial_number",
                "N/A"
            ),

            "gtin": parsed.get(
                "gtin",
                "N/A"
            ),

            "raw_text": qr_result["raw_data"],

            "qr_found": qr_result["found"],

            "qr_type": qr_result["qr_type"],

            "qr_confidence": qr_result["confidence"],

            "qr_format": parsed.get(
                "format",
                "Unknown"
            ),

            "prediction": ai_prediction,

            "ai_confidence": round(
                ai_confidence,
                2
            ),

            "ocr_confidence": (
                95.0
                if qr_result["found"]
                else 40.0
            ),

            "packaging_score": round(
                pkg_info.get(
                    "packaging_score",
                    50.0
                ),
                2
            ),

            "authenticity_score": round(
                authenticity,
                2
            ),

            "risk_level": risk_level,

            "recommendation": rec_str,

            "language": language,

            "image_path": image_path,

            "scan_type": "QR Code Scan",

            # Location fields
            "latitude": request.form.get(
                "latitude",
                ""
            ),

            "longitude": request.form.get(
                "longitude",
                ""
            ),
        }


        # Store QR report

        report_id = db.insert_report(
            report
        )


        if report_id is None:

            session["last_report"] = report

            report_id = "session"


        if rec_dict:

            session["rec_dict"] = rec_dict

            session.modified = True


        return render_template(
            "qr_result.html",
            report=report,
            qr_result=qr_result,
            report_id=report_id,
            rec_dict=rec_dict
        )


    except Exception as e:

        logger.error(
            f"QR analysis error: {e}",
            exc_info=True
        )


        flash(
            f"QR analysis failed: {str(e)}",
            "danger"
        )


        return redirect(
            url_for("scan_qr_page")
        )


# ─────────────────────────────────────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):

    return render_template(
        "index.html",
        stats=db.get_statistics()
    ), 404


@app.errorhandler(413)
def too_large(e):

    flash(
        f"File too large. Max {MAX_CONTENT_MB} MB.",
        "danger"
    )

    return redirect(
        url_for("upload")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run Application
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )