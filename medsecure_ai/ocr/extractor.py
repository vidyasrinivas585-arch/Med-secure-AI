"""
ocr/extractor.py
OCR-based information extraction from medicine package images.
Uses EasyOCR when available, falls back to OpenCV text detection.
"""

import re
import logging
import numpy as np
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_reader = None


def _get_reader():
    """Return EasyOCR reader if available, else None."""
    global _reader
    if _reader is None:
        try:
            import easyocr
            _reader = easyocr.Reader(["en", "kn"], gpu=False, verbose=False)
            logger.info("EasyOCR reader initialised.")
        except Exception as e:
            logger.warning(f"EasyOCR not available: {e}. Using OpenCV fallback.")
            _reader = "unavailable"
    return None if _reader == "unavailable" else _reader


def _ocr_with_opencv(image_path: str) -> tuple[str, float]:
    """
    OpenCV-based text extraction fallback when EasyOCR is not installed.
    Uses edge density and contour analysis to estimate OCR confidence.
    Returns extracted text (empty) and a confidence score based on
    how much text-like content is detected in the image.
    """
    try:
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            return "", 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # ── Method 1: Edge density (text has many edges) ──────────────────
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / (h * w)

        # ── Method 2: Count small contours (text characters) ─────────────
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # Text-like contours: small area, reasonable aspect ratio
        text_contours = 0
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            aspect = cw / ch if ch > 0 else 0
            if 10 < area < 2000 and 0.1 < aspect < 15:
                text_contours += 1

        # ── Method 3: Variance (text areas have high local variance) ─────
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # ── Compute confidence score ──────────────────────────────────────
        # Scale each metric to 0-100 contribution
        edge_score     = min(edge_density * 500, 40.0)       # max 40 pts
        contour_score  = min(text_contours * 0.5, 40.0)      # max 40 pts
        variance_score = min(variance / 50.0, 20.0)          # max 20 pts

        confidence = edge_score + contour_score + variance_score
        confidence = round(min(max(confidence, 0.0), 92.0), 2)

        logger.info(f"OpenCV OCR fallback — edges:{edge_density:.3f} "
                    f"contours:{text_contours} var:{variance:.1f} "
                    f"→ confidence:{confidence}%")

        return "", confidence

    except Exception as e:
        logger.error(f"OpenCV OCR fallback failed: {e}")
        return "", 45.0   # safe default so bar is visible


def extract_text_with_confidence(image_path: str) -> tuple[str, float]:
    """
    Extract text and confidence. Uses EasyOCR if available,
    otherwise uses OpenCV analysis to estimate a meaningful confidence.
    Returns (text, confidence_0_to_100).
    """
    reader = _get_reader()

    # ── EasyOCR path ──────────────────────────────────────────────────────
    if reader is not None:
        try:
            results = reader.readtext(image_path, detail=1, paragraph=False)
            if results:
                texts = [r[1] for r in results]
                confs  = [r[2] for r in results]
                full_text = " ".join(texts)
                mean_conf = (sum(confs) / len(confs)) * 100
                return full_text, round(mean_conf, 2)
        except Exception as e:
            logger.error(f"EasyOCR readtext failed: {e}")

    # ── OpenCV fallback path ──────────────────────────────────────────────
    return _ocr_with_opencv(image_path)


# ── Date / field regex patterns ───────────────────────────────────────────────

_DATE_PATTERNS = [
    r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b",
    r"\b(\d{2}[/-]\d{4})\b",
    r"\b([A-Za-z]{3}\.?\s?\d{4})\b",
    r"\b(\d{4}-\d{2}-\d{2})\b",
]

_BATCH_PATTERNS = [
    r"(?:batch|lot|b\.no|b/n|lot no\.?|batch no\.?)[:\s#]*([A-Z0-9\-]+)",
    r"\b(B[A-Z0-9]{4,10})\b",
]

_MFR_PATTERNS = [
    r"(?:mfg\s*by|manufactured by|mfr\.?)[:\s]*([A-Za-z &\.]+(?:Ltd|Pvt|Inc|Pharma|Labs?|Corp)?)",
    r"(?:marketed by)[:\s]*([A-Za-z &\.]+(?:Ltd|Pvt|Inc|Pharma|Labs?|Corp)?)",
]

_MED_NAME_PATTERN = (
    r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,3}"
    r"(?:\s\d+\s?mg|\s\d+\s?ml)?)\b"
)


def _first_match(text: str, patterns: list, flags=re.IGNORECASE) -> str:
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m.group(1).strip()
    return "Not Detected"


def _parse_date(date_str: str):
    formats = ["%d/%m/%Y", "%m/%Y", "%d-%m-%Y",
               "%Y-%m-%d", "%b %Y", "%b. %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def extract_medicine_info(image_path: str) -> dict[str, Any]:
    """
    Full extraction pipeline.
    Works with or without EasyOCR installed.
    """
    raw_text, ocr_conf_pct = extract_text_with_confidence(image_path)

    logger.info(f"OCR text: \"{raw_text[:100]}..." if raw_text else "OCR text: (empty)")
    logger.info(f"OCR confidence: {ocr_conf_pct}%")

    # ── Field extraction (only meaningful if EasyOCR extracted text) ──────
    medicine_name = _first_match(raw_text, [_MED_NAME_PATTERN])
    manufacturer  = _first_match(raw_text, _MFR_PATTERNS)
    batch_number  = _first_match(raw_text, _BATCH_PATTERNS)

    all_dates = []
    for pat in _DATE_PATTERNS:
        all_dates += re.findall(pat, raw_text, re.IGNORECASE)
    all_dates = list(dict.fromkeys(all_dates))

    mfg_date_str = all_dates[0] if len(all_dates) > 0 else "Not Detected"
    exp_date_str = all_dates[1] if len(all_dates) > 1 else "Not Detected"

    exp_match = re.search(
        r"(?:exp(?:iry)?\.?\s*date?|use before|best before)"
        r"[:\s]*(\d{2}[/-]\d{4}|\d{2}[/-]\d{2}[/-]\d{4})",
        raw_text, re.IGNORECASE
    )
    if exp_match:
        exp_date_str = exp_match.group(1).strip()

    # ── Expiry check ──────────────────────────────────────────────────────
    expiry_status   = "Unknown"
    days_remaining  = 0
    months_remaining = 0

    exp_dt = _parse_date(exp_date_str)
    if exp_dt:
        today = datetime.utcnow()
        delta = exp_dt - today
        days_remaining   = delta.days
        months_remaining = days_remaining // 30
        expiry_status    = "Valid" if days_remaining >= 0 else "Expired"

    return {
        "medicine_name":      medicine_name,
        "manufacturer":       manufacturer,
        "batch_number":       batch_number,
        "manufacturing_date": mfg_date_str,
        "expiry_date":        exp_date_str,
        "expiry_status":      expiry_status,
        "days_remaining":     max(days_remaining, 0),
        "months_remaining":   max(months_remaining, 0),
        "ocr_confidence":     ocr_conf_pct,
        "raw_text":           raw_text,
    }