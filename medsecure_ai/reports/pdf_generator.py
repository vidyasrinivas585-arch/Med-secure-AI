"""
reports/pdf_generator.py
Generates downloadable PDF analysis reports using ReportLab.
Supports English and Kannada content.
"""

import os
import logging
from datetime import datetime
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE_DIR, "static", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# Brand colours
BRAND_BLUE = colors.HexColor("#1A73E8")
BRAND_GREEN = colors.HexColor("#0F9D58")
BRAND_RED = colors.HexColor("#D93025")
BRAND_AMBER = colors.HexColor("#F9AB00")
LIGHT_GRAY = colors.HexColor("#F8F9FA")
DARK_GRAY = colors.HexColor("#212529")


def _risk_colour(risk: str):
    mapping = {
        "Low": BRAND_GREEN,
        "Medium": BRAND_AMBER,
        "High": BRAND_RED
    }

    return mapping.get(risk, BRAND_BLUE)


def _prediction_colour(pred: str):
    return BRAND_GREEN if pred == "Genuine" else BRAND_RED


def generate_pdf_report(
    data: dict,
    image_path: Optional[str] = None
) -> Optional[str]:
    """
    Generate a PDF report for a medicine analysis result.

    Args:
        data (dict): Full analysis result (from DB or decision engine).
        image_path (str): Path to the uploaded medicine image.

    Returns:
        str: Path to the generated PDF file, or None on failure.
    """

    try:
        timestamp_str = datetime.utcnow().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = f"medsecure_report_{timestamp_str}.pdf"

        pdf_path = os.path.join(
            REPORT_DIR,
            filename
        )

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Title"],
            fontSize=22,
            textColor=BRAND_BLUE,
            spaceAfter=6,
            alignment=TA_CENTER,
        )

        subtitle_style = ParagraphStyle(
            "Subtitle",
            fontSize=11,
            textColor=colors.gray,
            alignment=TA_CENTER,
            spaceAfter=12,
        )

        section_style = ParagraphStyle(
            "Section",
            fontSize=13,
            textColor=BRAND_BLUE,
            fontName="Helvetica-Bold",
            spaceBefore=14,
            spaceAfter=6,
        )

        body_style = ParagraphStyle(
            "Body",
            fontSize=10,
            textColor=DARK_GRAY,
            leading=14,
        )

        content = []

        # ── Header ──────────────────────────────────────
        content.append(
            Paragraph(
                "MedSecure AI",
                title_style
            )
        )

        content.append(
            Paragraph(
                "Medicine Authenticity Analysis Report",
                subtitle_style
            )
        )

        content.append(
            Paragraph(
                f"Generated: "
                f"{datetime.utcnow().strftime('%d %B %Y, %H:%M UTC')}",
                subtitle_style,
            )
        )

        content.append(
            HRFlowable(
                width="100%",
                thickness=2,
                color=BRAND_BLUE
            )
        )

        content.append(
            Spacer(
                1,
                0.4 * cm
            )
        )

        # ── Medicine Image ──────────────────────────────
        if image_path and os.path.exists(image_path):

            try:
                img = RLImage(
                    image_path,
                    width=6 * cm,
                    height=6 * cm
                )

                img.hAlign = "CENTER"

                content.append(img)

                content.append(
                    Spacer(
                        1,
                        0.3 * cm
                    )
                )

            except Exception as e:

                logger.warning(
                    f"Could not embed image in PDF: {e}"
                )

        # ── Prediction Banner ───────────────────────────
        pred = data.get(
            "prediction",
            "Unknown"
        )

        pred_colour = _prediction_colour(pred)

        pred_table_data = [[
            Paragraph(
                f'<font color="white">'
                f'<b>PREDICTION: {pred.upper()}</b>'
                f'</font>',
                ParagraphStyle(
                    "pred",
                    fontSize=14,
                    alignment=TA_CENTER
                ),
            )
        ]]

        pred_table = Table(
            pred_table_data,
            colWidths=["100%"]
        )

        pred_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    pred_colour
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    10
                ),
                (
                    "ROUNDEDCORNERS",
                    [8]
                ),
            ])
        )

        content.append(pred_table)

        content.append(
            Spacer(
                1,
                0.4 * cm
            )
        )

        # ── Authenticity Score + Risk ───────────────────
        score = data.get(
            "authenticity_score",
            0
        )

        risk = data.get(
            "risk_level",
            "Unknown"
        )

        risk_col = _risk_colour(risk)

        score_data = [[
            Paragraph(
                "<b>Authenticity Score</b>",
                body_style
            ),
            Paragraph(
                f"<b>{score}%</b>",
                body_style
            ),
            Paragraph(
                "<b>Risk Level</b>",
                body_style
            ),
            Paragraph(
                f'<font color="{risk_col.hexval()}">'
                f'<b>{risk}</b>'
                f'</font>',
                body_style
            ),
        ]]

        score_table = Table(
            score_data,
            colWidths=[
                "28%",
                "22%",
                "25%",
                "25%"
            ]
        )

        score_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    LIGHT_GRAY
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.lightgrey
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
            ])
        )

        content.append(score_table)

        content.append(
            Spacer(
                1,
                0.3 * cm
            )
        )

        # ── Medicine Information ─────────────────────────
        content.append(
            Paragraph(
                "Medicine Information",
                section_style
            )
        )

        info_fields = [
            (
                "Medicine Name",
                data.get(
                    "medicine_name",
                    "N/A"
                )
            ),
            (
                "Manufacturer",
                data.get(
                    "manufacturer",
                    "N/A"
                )
            ),
            (
                "Batch Number",
                data.get(
                    "batch_number",
                    "N/A"
                )
            ),
            (
                "Manufacturing Date",
                data.get(
                    "manufacturing_date",
                    "N/A"
                )
            ),
            (
                "Expiry Date",
                data.get(
                    "expiry_date",
                    "N/A"
                )
            ),
            (
                "Expiry Status",
                data.get(
                    "expiry_status",
                    "N/A"
                )
            ),
            (
                "Days Remaining",
                str(
                    data.get(
                        "days_remaining",
                        0
                    )
                )
            ),
        ]

        for label, value in info_fields:

            row_data = [[
                Paragraph(
                    f"<b>{label}</b>",
                    body_style
                ),
                Paragraph(
                    str(value),
                    body_style
                ),
            ]]

            t = Table(
                row_data,
                colWidths=[
                    "40%",
                    "60%"
                ]
            )

            t.setStyle(
                TableStyle([
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.lightgrey
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        LIGHT_GRAY
                    ),
                ])
            )

            content.append(t)

        # ── Analysis Scores ─────────────────────────────
        content.append(
            Spacer(
                1,
                0.2 * cm
            )
        )

        content.append(
            Paragraph(
                "Analysis Scores",
                section_style
            )
        )

        scores_data = [
            ["Metric", "Score"],
            [
                "AI Confidence",
                f"{data.get('ai_confidence', 0):.1f}%"
            ],
            [
                "OCR Confidence",
                f"{data.get('ocr_confidence', 0):.1f}%"
            ],
            [
                "Packaging Score",
                f"{data.get('packaging_score', 0):.1f}%"
            ],
            [
                "Authenticity Score",
                f"{score:.1f}%"
            ],
        ]

        scores_table = Table(
            scores_data,
            colWidths=[
                "60%",
                "40%"
            ]
        )

        scores_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    BRAND_BLUE
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.lightgrey
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        LIGHT_GRAY
                    ]
                ),
            ])
        )

        content.append(scores_table)

        # ── Recommendation ──────────────────────────────
        content.append(
            Spacer(
                1,
                0.4 * cm
            )
        )

        content.append(
            Paragraph(
                "Recommendation",
                section_style
            )
        )

        rec = data.get(
            "recommendation",
            "N/A"
        )

        content.append(
            Paragraph(
                rec,
                body_style
            )
        )

        # ── Safety Guidance ─────────────────────────────
        safety = data.get(
            "safety_guidance",
            {}
        )

        if safety and not safety.get(
            "is_safe",
            True
        ):

            content.append(
                Spacer(
                    1,
                    0.3 * cm
                )
            )

            content.append(
                Paragraph(
                    "Safety Guidance",
                    section_style
                )
            )

            content.append(
                Paragraph(
                    f"<b>{safety.get('title', '')}</b>",
                    body_style
                )
            )

            for action in safety.get(
                "actions",
                []
            ):

                content.append(
                    Paragraph(
                        f"• {action}",
                        body_style
                    )
                )

        # ── AI Explainability ───────────────────────────
        explanation = data.get(
            "explanation",
            {}
        )

        if (
            explanation
            and explanation.get("reasons")
        ):

            content.append(
                Spacer(
                    1,
                    0.3 * cm
                )
            )

            content.append(
                Paragraph(
                    "Reason for Prediction",
                    section_style
                )
            )

            for reason in explanation["reasons"]:

                icon = (
                    "✗"
                    if reason.get("flagged")
                    else "✓"
                )

                content.append(
                    Paragraph(
                        f"{icon} "
                        f"{reason.get('text', '')}",
                        body_style
                    )
                )

            content.append(
                Paragraph(
                    f"<b>Overall Confidence: "
                    f"{explanation.get('overall_confidence', 0)}%"
                    f"</b>",
                    body_style,
                )
            )

        # ── Nearby Pharmacies ───────────────────────────
        pharmacies = data.get(
            "nearby_pharmacies",
            []
        )

        if pharmacies:

            content.append(
                Spacer(
                    1,
                    0.3 * cm
                )
            )

            content.append(
                Paragraph(
                    "Nearby Licensed Pharmacies",
                    section_style
                )
            )

            pharm_data = [
                [
                    "Name",
                    "Distance",
                    "Rating",
                    "Address"
                ]
            ]

            for p in pharmacies[:10]:

                pharm_data.append([
                    p.get(
                        "name",
                        "N/A"
                    ),
                    f"{p.get('distance_km', '?')} km",
                    str(
                        p.get(
                            "rating",
                            "N/A"
                        )
                    ),
                    p.get(
                        "address",
                        "N/A"
                    ),
                ])

            pharm_table = Table(
                pharm_data,
                colWidths=[
                    "30%",
                    "12%",
                    "12%",
                    "46%"
                ]
            )

            pharm_table.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        BRAND_BLUE
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.lightgrey
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [
                            colors.white,
                            LIGHT_GRAY
                        ]
                    ),
                ])
            )

            content.append(pharm_table)

        # ── Language ────────────────────────────────────
        lang = data.get(
            "language",
            "en"
        )

        lang_display = (
            "English"
            if lang == "en"
            else "Kannada (ಕನ್ನಡ)"
        )

        content.append(
            Spacer(
                1,
                0.3 * cm
            )
        )

        content.append(
            Paragraph(
                f"<i>Report Language: "
                f"{lang_display}</i>",
                subtitle_style
            )
        )

        # ── Footer ──────────────────────────────────────
        content.append(
            Spacer(
                1,
                0.5 * cm
            )
        )

        content.append(
            HRFlowable(
                width="100%",
                thickness=1,
                color=colors.lightgrey
            )
        )

        content.append(
            Paragraph(
                "⚠️ This report is generated by an AI system "
                "for screening purposes only. "
                "Always consult a licensed pharmacist or "
                "healthcare professional before making "
                "decisions based on this report.",
                ParagraphStyle(
                    "disclaimer",
                    fontSize=8,
                    textColor=colors.gray,
                    alignment=TA_CENTER
                ),
            )
        )

        # Build PDF
        doc.build(content)

        logger.info(
            f"PDF report generated: {pdf_path}"
        )

        return pdf_path

    except Exception as e:

        logger.error(
            f"PDF generation failed: {e}"
        )

        return None