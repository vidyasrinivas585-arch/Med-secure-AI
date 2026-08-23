"""
ocr/qr_scanner.py
QR Code and Barcode scanner for MedSecure AI.
Uses OpenCV's built-in QRCodeDetector — no extra packages needed.
Also attempts barcode detection using contour analysis.

Extracts from QR data:
    - Medicine name
    - Manufacturer
    - Batch number
    - Expiry date
    - Serial number
    - Any other key:value pairs
"""

import cv2
import numpy as np
import re
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ✅ WeChatQRCode only exists in opencv-contrib-python, not the
#    opencv-python-headless build in requirements.txt. Check once at
#    import time instead of raising/catching an exception on every
#    single scan.
_WECHAT_QR_AVAILABLE = hasattr(cv2, "wechat_qrcode_WeChatQRCode")
if not _WECHAT_QR_AVAILABLE:
    logger.info(
        "cv2.wechat_qrcode_WeChatQRCode not available "
        "(needs opencv-contrib-python) — skipping that detector, "
        "falling back to cv2.QRCodeDetector + barcode region detection."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORE QR DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def scan_qr_code(image_path: str) -> dict:
    """
    Main function — scan image for QR codes and barcodes.

    Tries multiple preprocessing strategies to maximise detection rate.

    Args:
        image_path (str): Path to the medicine package image.

    Returns:
        dict: {
            "found":        bool,
            "raw_data":     str,
            "qr_type":      str,   # "QR Code", "Barcode", "None"
            "bbox":         list,  # bounding box corners
            "parsed":       dict,  # extracted medicine fields
            "confidence":   float, # 0-100
        }
    """
    result = {
        "found":      False,
        "raw_data":   "",
        "qr_type":    "None",
        "bbox":       [],
        "parsed":     {},
        "confidence": 0.0,
    }

    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Cannot load image: {image_path}")
        return result

    # ── Try QR detection with multiple preprocessing strategies ──────────
    strategies = [
        ("original",    img.copy()),
        ("upscaled",    cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
        ("grayscale",   _to_gray_bgr(img)),
        ("sharpened",   _sharpen(img)),
        ("denoised",    cv2.fastNlMeansDenoisingColored(img, h=10)),
        ("high_contrast", _enhance_contrast(img)),
    ]

    qr_detector = cv2.QRCodeDetector()

    for strategy_name, processed in strategies:
        data, bbox, _ = qr_detector.detectAndDecode(processed)
        if data and data.strip():
            logger.info(f"QR detected with strategy '{strategy_name}': {data[:80]}")
            result["found"]      = True
            result["raw_data"]   = data.strip()
            result["qr_type"]    = "QR Code"
            result["confidence"] = 95.0
            if bbox is not None:
                result["bbox"] = bbox.tolist()
            result["parsed"] = _parse_qr_data(data.strip())
            return result

    # ── Try WeChatQRCode detector (more powerful, available in OpenCV 4.5+) ─
    if _WECHAT_QR_AVAILABLE:
        try:
            wechat_qr = cv2.wechat_qrcode_WeChatQRCode()
            texts, points = wechat_qr.detectAndDecode(img)
            if texts:
                data = texts[0]
                logger.info(f"WeChatQR detected: {data[:80]}")
                result["found"]      = True
                result["raw_data"]   = data.strip()
                result["qr_type"]    = "QR Code"
                result["confidence"] = 92.0
                result["parsed"]     = _parse_qr_data(data.strip())
                return result
        except Exception as e:
            logger.debug(f"WeChatQRCode detection failed: {e}")

    # ── Barcode region detection (visual, no decode) ──────────────────────
    barcode_found, barcode_conf = _detect_barcode_region(img)
    if barcode_found:
        result["found"]      = True
        result["raw_data"]   = "BARCODE_DETECTED"
        result["qr_type"]    = "Barcode"
        result["confidence"] = barcode_conf
        result["parsed"]     = {
            "note": "Barcode detected but could not be decoded. "
                    "Use a dedicated barcode scanner for full data."
        }
        return result

    logger.info("No QR code or barcode found in image.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _to_gray_bgr(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale then back to BGR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _sharpen(img: np.ndarray) -> np.ndarray:
    """Apply sharpening kernel."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _detect_barcode_region(img: np.ndarray) -> tuple[bool, float]:
    """
    Detect barcode-like regions using gradient analysis.
    Barcodes have strong horizontal gradients in a rectangular region.
    Returns (found, confidence).
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.subtract(np.abs(grad_x), np.abs(grad_y))
        gradient = cv2.convertScaleAbs(gradient)
        # Blur and threshold
        blurred  = cv2.blur(gradient, (9, 9))
        _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
        # Morphological close to connect barcode bars
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed   = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # Erode + dilate
        eroded   = cv2.erode(closed, None, iterations=4)
        dilated  = cv2.dilate(eroded, None, iterations=4)
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h if h > 0 else 0
            area   = w * h
            # Barcodes are wide rectangles with significant area
            if aspect > 2.0 and area > 3000:
                confidence = min(50.0 + aspect * 5, 80.0)
                return True, round(confidence, 1)
        return False, 0.0
    except Exception as e:
        logger.error(f"Barcode region detection failed: {e}")
        return False, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# QR DATA PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _parse_qr_data(raw: str) -> dict:
    """
    Parse QR code data into structured medicine fields.

    Handles multiple formats:
        1. JSON  : {"name":"Paracetamol","batch":"BT123","exp":"06/2026"}
        2. URL   : https://verify.cipla.com/?batch=BT123&exp=2026-06
        3. Key:Value pairs: NAME:Paracetamol|BATCH:BT123|EXP:06/2026
        4. Pipe-separated: Paracetamol|Cipla|BT123|06/2026
        5. GS1 standard: (01)07290105705006(17)260630(10)BT123
        6. Plain text  : Any free-form text
    """
    parsed = {"raw": raw}

    # ── Format 1: JSON ───────────────────────────────────────────────────
    if raw.startswith("{") or raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                parsed.update(_map_json_fields(data))
                parsed["format"] = "JSON"
                return parsed
        except json.JSONDecodeError:
            pass

    # ── Format 2: URL with query parameters ──────────────────────────────
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed["url"]    = raw
        parsed["format"] = "URL"
        # Extract query params
        params = _parse_url_params(raw)
        parsed.update(_map_json_fields(params))
        return parsed

    # ── Format 3: GS1 standard barcodes ──────────────────────────────────
    if "(" in raw and ")" in raw:
        gs1 = _parse_gs1(raw)
        if gs1:
            parsed.update(gs1)
            parsed["format"] = "GS1"
            return parsed

    # ── Format 4: Key:Value pipe-separated ───────────────────────────────
    if "|" in raw and ":" in raw:
        pairs = {}
        for part in raw.split("|"):
            if ":" in part:
                k, _, v = part.partition(":")
                pairs[k.strip().lower()] = v.strip()
        if pairs:
            parsed.update(_map_json_fields(pairs))
            parsed["format"] = "Key-Value"
            return parsed

    # ── Format 5: Pipe-separated plain values ────────────────────────────
    if "|" in raw:
        parts = [p.strip() for p in raw.split("|")]
        field_names = ["medicine_name", "manufacturer",
                       "batch_number", "expiry_date", "serial_number"]
        for i, val in enumerate(parts):
            if i < len(field_names) and val:
                parsed[field_names[i]] = val
        parsed["format"] = "Pipe-Separated"
        return parsed

    # ── Format 6: Plain text — extract using regex ────────────────────────
    parsed.update(_regex_extract(raw))
    parsed["format"] = "Plain Text"
    return parsed


def _map_json_fields(data: dict) -> dict:
    """Map various field name variants to standard field names."""
    mapping = {
        # Medicine name variants
        "name":         "medicine_name",
        "medicine":     "medicine_name",
        "drug":         "medicine_name",
        "product":      "medicine_name",
        "medicine_name":"medicine_name",
        "productname":  "medicine_name",
        # Manufacturer variants
        "manufacturer": "manufacturer",
        "mfr":          "manufacturer",
        "company":      "manufacturer",
        "brand":        "manufacturer",
        "mfg":          "manufacturer",
        # Batch variants
        "batch":        "batch_number",
        "batch_no":     "batch_number",
        "lot":          "batch_number",
        "lot_no":       "batch_number",
        "batchno":      "batch_number",
        # Expiry variants
        "exp":          "expiry_date",
        "expiry":       "expiry_date",
        "expiry_date":  "expiry_date",
        "expdate":      "expiry_date",
        "best_before":  "expiry_date",
        "use_before":   "expiry_date",
        # Manufacturing date
        "mfg_date":     "manufacturing_date",
        "manufacture":  "manufacturing_date",
        "manufactured": "manufacturing_date",
        # Serial / other
        "serial":       "serial_number",
        "serial_no":    "serial_number",
        "gtin":         "gtin",
        "ndc":          "ndc",
        "dosage":       "dosage",
        "strength":     "dosage",
    }
    result = {}
    for key, value in data.items():
        std_key = mapping.get(key.lower().replace(" ", "_"), key.lower())
        result[std_key] = str(value)
    return result


def _parse_url_params(url: str) -> dict:
    """Extract query parameters from a URL."""
    params = {}
    if "?" in url:
        query = url.split("?", 1)[1]
        for pair in query.split("&"):
            if "=" in pair:
                k, _, v = pair.partition("=")
                params[k] = v
    return params


def _parse_gs1(raw: str) -> dict:
    """
    Parse GS1 Application Identifier format.
    Common AIs: (01)=GTIN, (17)=Expiry YYMMDD, (10)=Batch/Lot
    """
    result = {}
    # Find all (AI)value pairs
    pattern = r'\((\d{2,4})\)([^(]+)'
    matches  = re.findall(pattern, raw)
    ai_map   = {
        "01": "gtin",
        "02": "gtin",
        "10": "batch_number",
        "11": "manufacturing_date",
        "17": "expiry_date",
        "21": "serial_number",
        "30": "quantity",
    }
    for ai, value in matches:
        field = ai_map.get(ai)
        if field:
            value = value.strip()
            # Format GS1 date YYMMDD → MM/YYYY
            if field in ("expiry_date", "manufacturing_date") and len(value) == 6:
                try:
                    yy, mm = value[:2], value[2:4]
                    year   = "20" + yy
                    value  = f"{mm}/{year}"
                except Exception:
                    pass
            result[field] = value
    return result


def _regex_extract(text: str) -> dict:
    """Extract medicine fields from plain text using regex."""
    result = {}
    # Batch number
    m = re.search(r"(?:batch|lot|b\.?no)[:\s#]*([A-Z0-9\-]+)", text, re.IGNORECASE)
    if m:
        result["batch_number"] = m.group(1)
    # Expiry date
    m = re.search(r"(?:exp|expiry|use before|best before)[:\s]*"
                  r"(\d{2}[/-]\d{4}|\d{2}[/-]\d{2}[/-]\d{4})", text, re.IGNORECASE)
    if m:
        result["expiry_date"] = m.group(1)
    # Dates in general
    dates = re.findall(r"\b(\d{2}[/-]\d{2}[/-]\d{4}|\d{2}[/-]\d{4})\b", text)
    if dates and "expiry_date" not in result:
        result["expiry_date"] = dates[-1]  # last date is usually expiry
    # Medicine name (first capitalised phrase)
    m = re.search(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?(?:\s\d+\s?mg|\s\d+\s?ml)?)\b", text)
    if m:
        result["medicine_name"] = m.group(1)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# EXPIRY CHECK FROM QR DATA
# ─────────────────────────────────────────────────────────────────────────────

def check_expiry_from_qr(parsed: dict) -> dict:
    """
    Check expiry status from QR-parsed data.
    Returns expiry_status, days_remaining, months_remaining.
    """
    exp_str = parsed.get("expiry_date", "")
    if not exp_str:
        return {"expiry_status": "Unknown", "days_remaining": 0, "months_remaining": 0}

    formats = ["%m/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%Y", "%b %Y"]
    exp_dt  = None
    for fmt in formats:
        try:
            exp_dt = datetime.strptime(exp_str.strip(), fmt)
            break
        except ValueError:
            continue

    if exp_dt is None:
        return {"expiry_status": "Unknown", "days_remaining": 0, "months_remaining": 0}

    today  = datetime.utcnow()
    delta  = exp_dt - today
    days   = delta.days
    months = days // 30
    status = "Valid" if days >= 0 else "Expired"

    return {
        "expiry_status":   status,
        "days_remaining":  max(days, 0),
        "months_remaining": max(months, 0),
    }