"""
ocr/extractor.py
OCR-based information extraction from medicine package images.
Uses EasyOCR with English + Kannada language support.
"""

import re
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-load EasyOCR reader (heavy to initialise; singleton pattern)
_reader = None


def _get_reader():
    """Return a cached EasyOCR reader supporting English and Kannada."""
    global _reader
    if _reader is None:
        try:
            import easyocr
            # 'en' = English, 'kn' = Kannada
            _reader = easyocr.Reader(["en", "kn"], gpu=False, verbose=False)
            logger.info("EasyOCR reader initialised (en + kn).")
        except Exception as e:
            logger.error(f"EasyOCR init failed: {e}")
            _reader = None
    return _reader


# ─────────────────────────────────────────────────────────────────────────────
# Raw OCR
# ─────────────────────────────────────────────────────────────────────────────

def run_ocr(image_input) -> list[tuple]:
    """
    Run EasyOCR on an image file path or a numpy array.

    Args:
        image_input: File path (str) or preprocessed numpy array.

    Returns:
        list[tuple]: List of (bbox, text, confidence) tuples.
    """
    reader = _get_reader()
    if reader is None:
        return []
    try:
        results = reader.readtext(image_input, detail=1, paragraph=False)
        return results  # [(bbox, text, confidence), ...]
    except Exception as e:
        logger.error(f"OCR readtext failed: {e}")
        return []


def extract_text_with_confidence(image_path: str) -> tuple[str, float]:
    """
    Extract full raw text from an image and compute mean OCR confidence.

    Args:
        image_path (str): Path to the image.

    Returns:
        tuple[str, float]: (concatenated_text, mean_confidence 0–1)
    """
    results = run_ocr(image_path)
    if not results:
        return "", 0.0

    texts = [r[1] for r in results]
    confidences = [r[2] for r in results]

    full_text = " ".join(texts)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return full_text, mean_conf


# ─────────────────────────────────────────────────────────────────────────────
# Structured field extraction via regex
# ─────────────────────────────────────────────────────────────────────────────

# Date patterns: DD/MM/YYYY, MM/YYYY, DD-MM-YYYY, MMM YYYY, etc.
_DATE_PATTERNS = [
    r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b",
    r"\b(\d{2}[/-]\d{4})\b",
    r"\b([A-Za-z]{3}\.?\s?\d{4})\b",
    r"\b(\d{4}-\d{2}-\d{2})\b",
]

# Batch / lot number patterns
_BATCH_PATTERNS = [
    r"(?:batch|lot|b\.no|b/n|lot no\.?|batch no\.?)[:\s#]*([A-Z0-9\-]+)",
    r"\b(B[A-Z0-9]{4,10})\b",
]

# Manufacturer patterns
_MFR_PATTERNS = [
    r"(?:mfg\s*by|manufactured by|mfr\.?)[:\s]*([A-Za-z &\.]+(?:Ltd|Pvt|Inc|Pharma|Labs?|Corp)?)",
    r"(?:marketed by)[:\s]*([A-Za-z &\.]+(?:Ltd|Pvt|Inc|Pharma|Labs?|Corp)?)",
]

# Medicine name — first capitalised multi-word cluster (heuristic)
_MED_NAME_PATTERN = r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,3}(?:\s\d+\s?mg|\s\d+\s?ml)?)\b"


def _first_match(text: str, patterns: list[str], flags=re.IGNORECASE) -> str:
    """Return the first capture group match across a list of patterns."""
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m.group(1).strip()
    return "Not Detected"


def _parse_date(date_str: str) -> datetime | None:
    """Try multiple format strings to parse a date."""
    formats = ["%d/%m/%Y", "%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%b %Y", "%b. %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def extract_medicine_info(image_path: str) -> dict[str, Any]:
    """
    Full extraction pipeline: OCR → regex field parsing → expiry check.

    Args:
        image_path (str): Path to the medicine package image.

    Returns:
        dict with keys:
            medicine_name, manufacturer, batch_number,
            manufacturing_date, expiry_date, expiry_status,
            days_remaining, months_remaining, ocr_confidence,
            raw_text
    """
    raw_text, ocr_confidence = extract_text_with_confidence(image_path)

    logger.info(f"Raw OCR text: {raw_text[:200]}...")

    # ── Field extraction ──────────────────────────────────────────────────
    medicine_name = _first_match(raw_text, [_MED_NAME_PATTERN])
    manufacturer  = _first_match(raw_text, _MFR_PATTERNS)
    batch_number  = _first_match(raw_text, _BATCH_PATTERNS)

    # Date extraction — find all dates, heuristically assign mfg vs exp
    all_dates = []
    for pat in _DATE_PATTERNS:
        all_dates += re.findall(pat, raw_text, re.IGNORECASE)
    all_dates = list(dict.fromkeys(all_dates))  # deduplicate, preserve order

    mfg_date_str = all_dates[0] if len(all_dates) > 0 else "Not Detected"
    exp_date_str = all_dates[1] if len(all_dates) > 1 else "Not Detected"

    # Check for explicit "Exp" / "Expiry" label to pick the right date
    exp_match = re.search(
        r"(?:exp(?:iry)?\.?\s*date?|use before|best before)[:\s]*(" + "|".join(_DATE_PATTERNS) + ")",
        raw_text, re.IGNORECASE
    )
    if exp_match:
        exp_date_str = exp_match.group(1).strip()

    # ── Expiry check ──────────────────────────────────────────────────────
    expiry_status = "Unknown"
    days_remaining = 0
    months_remaining = 0

    exp_dt = _parse_date(exp_date_str)
    if exp_dt:
        today = datetime.utcnow()
        delta = exp_dt - today
        days_remaining = delta.days
        months_remaining = days_remaining // 30
        expiry_status = "Valid" if days_remaining >= 0 else "Expired"

    return {
        "medicine_name":     medicine_name,
        "manufacturer":      manufacturer,
        "batch_number":      batch_number,
        "manufacturing_date": mfg_date_str,
        "expiry_date":       exp_date_str,
        "expiry_status":     expiry_status,
        "days_remaining":    max(days_remaining, 0),
        "months_remaining":  max(months_remaining, 0),
        "ocr_confidence":    round(ocr_confidence * 100, 2),
        "raw_text":          raw_text,
    }
