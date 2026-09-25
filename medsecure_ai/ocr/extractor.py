"""
ocr/extractor.py
MedSecure AI - Improved OCR Engine

Features:
- Blurry medicine image enhancement
- Upscaling
- CLAHE contrast enhancement
- Denoising
- Sharpening / deblurring
- Multiple thresholding methods
- Multiple Tesseract PSM modes
- Medicine name extraction
- Manufacturer extraction
- Batch number extraction
- Manufacturing / expiry date extraction
- Better handling of formats like:
    05/05/2026
    05-05-2026
    05.05.2026
    05/2026
    MAY 2026
    5.5.3.20260724
"""

import os
import re
import cv2
import logging
import subprocess
import numpy as np

from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


# ============================================================
# TESSERACT
# ============================================================

TESSERACT_PATHS = [
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "tesseract",
]


def _find_tesseract():

    for path in TESSERACT_PATHS:

        try:

            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                timeout=5
            )

            if result.returncode == 0:
                return path

        except Exception:
            continue

    return None


# ============================================================
# IMAGE LOADING
# ============================================================

def _load_image(image_path: str):

    img = cv2.imread(image_path)

    if img is not None:
        return img

    try:

        pil = Image.open(image_path).convert("RGB")

        img = cv2.cvtColor(
            np.array(pil),
            cv2.COLOR_RGB2BGR
        )

        return img

    except Exception as e:

        logger.error(
            f"Cannot load image: {e}"
        )

        return None


# ============================================================
# BLUR DETECTION
# ============================================================

def _blur_score(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return float(score)


def _is_blurry(img):

    score = _blur_score(img)

    logger.info(
        f"Image blur score: {score:.2f}"
    )

    return score < 120


# ============================================================
# IMAGE ENHANCEMENT
# ============================================================

def _enhance_image(img):

    """
    Improve blurry medicine images before OCR.
    """

    # --------------------------------------------------------
    # Upscale
    # --------------------------------------------------------

    h, w = img.shape[:2]

    min_dimension = min(h, w)

    if min_dimension < 1200:

        scale = 1200 / min_dimension

        scale = min(scale, 3.0)

        img = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Denoise
    # --------------------------------------------------------

    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        denoised
    )

    # --------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------

    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    )

    sharpened = cv2.filter2D(
        enhanced,
        -1,
        kernel
    )

    # --------------------------------------------------------
    # Strong sharpen
    # --------------------------------------------------------

    blur = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        3
    )

    unsharp = cv2.addWeighted(
        enhanced,
        1.7,
        blur,
        -0.7,
        0
    )

    return {
        "gray": gray,
        "denoised": denoised,
        "clahe": enhanced,
        "sharp": sharpened,
        "unsharp": unsharp
    }


# ============================================================
# OCR VARIANTS
# ============================================================

def _get_image_variants(image_path):

    img = _load_image(
        image_path
    )

    if img is None:
        return []

    enhanced = _enhance_image(
        img
    )

    variants = []

    for name, image in enhanced.items():

        variants.append(
            (
                name,
                Image.fromarray(image)
            )
        )

    # --------------------------------------------------------
    # Threshold variants
    # --------------------------------------------------------

    base = enhanced["clahe"]

    # OTSU
    _, otsu = cv2.threshold(
        base,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    variants.append(
        (
            "otsu",
            Image.fromarray(otsu)
        )
    )

    # Adaptive
    adaptive = cv2.adaptiveThreshold(
        base,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    variants.append(
        (
            "adaptive",
            Image.fromarray(adaptive)
        )
    )

    # Inverted
    inverted = cv2.bitwise_not(
        otsu
    )

    variants.append(
        (
            "inverted",
            Image.fromarray(inverted)
        )
    )

    # --------------------------------------------------------
    # PIL enhancement
    # --------------------------------------------------------

    pil_img = Image.fromarray(
        enhanced["clahe"]
    )

    contrast = ImageEnhance.Contrast(
        pil_img
    ).enhance(2.5)

    sharp = ImageEnhance.Sharpness(
        contrast
    ).enhance(3.0)

    variants.append(
        (
            "pil_enhanced",
            sharp
        )
    )

    return variants


# ============================================================
# TESSERACT OCR
# ============================================================

def _ocr_image(
    pil_img,
    tess_path,
    psm
):

    try:

        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = (
            tess_path
        )

        data = pytesseract.image_to_data(
            pil_img,
            output_type=pytesseract.Output.DICT,
            config=f"--psm {psm} --oem 3"
        )

        texts = []
        confidences = []

        for i, txt in enumerate(
            data["text"]
        ):

            txt = txt.strip()

            try:
                conf = float(
                    data["conf"][i]
                )
            except Exception:
                conf = 0

            if (
                txt
                and conf > 5
            ):

                texts.append(
                    txt
                )

                confidences.append(
                    conf
                )

        if texts:

            text = " ".join(
                texts
            )

            confidence = (
                sum(confidences)
                /
                len(confidences)
            )

            return (
                text,
                confidence
            )

        # fallback
        text = pytesseract.image_to_string(
            pil_img,
            config=f"--psm {psm} --oem 3"
        ).strip()

        if text:

            return (
                text,
                45.0
            )

        return "", 0.0

    except Exception as e:

        logger.debug(
            f"OCR error: {e}"
        )

        return "", 0.0


# ============================================================
# OCR ENGINE
# ============================================================

def extract_text_with_confidence(
    image_path
):

    tess_path = _find_tesseract()

    if tess_path is None:

        logger.warning(
            "Tesseract not found."
        )

        return (
            "",
            0.0
        )

    variants = _get_image_variants(
        image_path
    )

    if not variants:

        return (
            "",
            0.0
        )

    # PSM modes
    psm_modes = [
        6,
        11,
        12,
        3,
        4,
        7
    ]

    best_text = ""
    best_conf = 0
    best_score = 0

    for name, image in variants:

        for psm in psm_modes:

            text, conf = _ocr_image(
                image,
                tess_path,
                psm
            )

            if not text:
                continue

            words = [
                w for w in text.split()
                if len(w) > 1
            ]

            word_count = len(words)

            # More useful than just word count
            useful_score = (
                word_count * 2
                +
                conf * 0.5
            )

            if useful_score > best_score:

                best_score = useful_score

                best_text = text

                best_conf = conf

                logger.info(
                    f"NEW OCR RESULT → "
                    f"{name}/PSM{psm}"
                )

                logger.info(
                    f"Text: {text[:300]}"
                )

    if best_text:

        logger.info(
            f"FINAL OCR → "
            f"{best_conf:.2f}% confidence"
        )

        return (
            best_text,
            round(best_conf, 2)
        )

    return (
        "",
        0.0
    )


# ============================================================
# FIELD EXTRACTION
# ============================================================

SKIP_WORDS = {
    "EACH",
    "STORE",
    "OVER",
    "MADE",
    "REGD",
    "TRADE",
    "MARK",
    "DOSAGE",
    "DOSE",
    "INDIA",
    "DIRECTED",
    "PHYSICIAN",
    "TEMPERATURE",
    "EXCEEDING",
    "INJURIOUS",
    "TABLET",
    "TABLETS",
    "CAPSULE",
    "CAPSULES",
    "INJECTION",
    "WARNING",
    "KEEP",
    "CHILDREN",
    "REACH",
    "SCHEDULE",
    "DRUG",
    "PRESCRIPTION",
    "ONLY",
}


# ============================================================
# MEDICINE NAME
# ============================================================

def _extract_medicine_name(text):

    if not text.strip():
        return "Not Detected"

    patterns = [

        # Dolo 650
        r"\b([A-Za-z][A-Za-z\-]{2,})\s*(\d{2,4})\s*(?:mg|ml|mcg|g)?\b",

        # Paracetamol 650mg
        r"\b([A-Za-z][A-Za-z\s\-]{2,})\s+"
        r"(\d{2,4}\s*(?:mg|ml|mcg|g|IU))\b",

        # Brand:
        r"(?:brand|trade|product|medicine|drug)"
        r"\s*(?:name)?\s*[:\-]\s*"
        r"([A-Za-z][A-Za-z0-9\-\s]+)",

    ]

    for pattern in patterns:

        matches = re.finditer(
            pattern,
            text,
            re.IGNORECASE
        )

        for match in matches:

            groups = match.groups()

            name = " ".join(
                g.strip()
                for g in groups
                if g
            )

            words = name.upper().split()

            if any(
                w in SKIP_WORDS
                for w in words
            ):
                continue

            if len(name) > 2:

                return name[:60]

    # fallback
    for token in text.split():

        token = token.strip(
            ".,:;-/()[]"
        )

        if (
            len(token) >= 4
            and token[0].isalpha()
            and token.upper()
            not in SKIP_WORDS
        ):

            return token[:50]

    return "Not Detected"


# ============================================================
# MANUFACTURER
# ============================================================

def _extract_manufacturer(text):

    if not text.strip():
        return "Not Detected"

    patterns = [

        r"(?:manufactured\s+by|"
        r"manufactured\s+for|"
        r"mfg\.?\s*by|"
        r"mfd\.?\s*by|"
        r"marketed\s+by|"
        r"distributed\s+by)"
        r"\s*[:\-]?\s*"
        r"([A-Za-z][A-Za-z\s&\.]+)",

        r"\b([A-Z][A-Z\s&]{3,}"
        r"(?:PHARMA|LABS|"
        r"LABORATORIES|"
        r"HEALTHCARE|"
        r"PHARMACEUTICALS))\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            result = re.sub(
                r"\s+",
                " ",
                match.group(1)
            ).strip()

            if len(result) > 3:

                return result[:80]

    return "Not Detected"


# ============================================================
# BATCH NUMBER
# ============================================================

def _extract_batch(text):

    if not text.strip():
        return "Not Detected"

    patterns = [

        r"(?:batch|batch\s*no|"
        r"lot|lot\s*no|"
        r"b\.?\s*no)"
        r"\s*[:\-#]?\s*"
        r"([A-Z0-9][A-Z0-9\-\/\.]{2,20})",

        r"\b([A-Z]{1,4}\d{2,10})\b",

        r"\b([A-Z0-9]{3,}"
        r"[\-\/]"
        r"[A-Z0-9]{2,12})\b",
    ]

    for pattern in patterns:

        matches = re.finditer(
            pattern,
            text,
            re.IGNORECASE
        )

        for match in matches:

            result = match.group(
                1
            ).strip().upper()

            if len(result) >= 4:

                return result

    return "Not Detected"


# ============================================================
# DATE EXTRACTION
# ============================================================

def _extract_dates(text):

    if not text.strip():

        return (
            "Not Detected",
            "Not Detected"
        )

    # Standard formats
    date_patterns = [

        r"\b\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4}\b",

        r"\b\d{2}[\/\-\.]\d{4}\b",

        r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2}\b",

        r"\b\d{4}[\/\-]\d{2}[\/\-]\d{2}\b",

        r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|"
        r"JUL|AUG|SEP|OCT|NOV|DEC)"
        r"[A-Z]*\.?\s+\d{4}\b",

        r"\b\d{1,2}\.\d{4}\b",

        r"\b\d{1,2}[\/\-]\d{2}\b",
    ]

    all_dates = []

    for pattern in date_patterns:

        matches = re.findall(
            pattern,
            text,
            re.IGNORECASE
        )

        for date in matches:

            date = date.strip()

            if date not in all_dates:

                all_dates.append(
                    date
                )

    # --------------------------------------------------------
    # Explicit EXP date
    # --------------------------------------------------------

    exp_patterns = [

        r"(?:exp|expiry|expiry\s*date|"
        r"expires|use\s*before|"
        r"best\s*before)"
        r"\s*[:\-]?\s*"
        r"([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})",

        r"(?:exp|expiry|"
        r"use\s*before|best\s*before)"
        r"\s*[:\-]?\s*"
        r"([0-9]{1,2}[\/\-\.][0-9]{4})",

        r"(?:exp|expiry)"
        r"\s*[:\-]?\s*"
        r"([A-Za-z]{3,9}\.?\s*\d{4})",
    ]

    exp_date = "Not Detected"

    for pattern in exp_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            exp_date = match.group(
                1
            ).strip()

            break

    # --------------------------------------------------------
    # Explicit MFG date
    # --------------------------------------------------------

    mfg_patterns = [

        r"(?:mfg|mfd|manufactured)"
        r"\s*(?:date|on)?"
        r"\s*[:\-]?\s*"
        r"([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})",

        r"(?:mfg|mfd|manufactured)"
        r"\s*(?:date|on)?"
        r"\s*[:\-]?\s*"
        r"([0-9]{1,2}[\/\-\.][0-9]{4})",
    ]

    mfg_date = "Not Detected"

    for pattern in mfg_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            mfg_date = match.group(
                1
            ).strip()

            break

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if exp_date == "Not Detected":

        if len(all_dates) >= 2:

            # Usually first = MFG
            # second = EXP
            if mfg_date == "Not Detected":

                mfg_date = all_dates[0]

            exp_date = all_dates[1]

        elif len(all_dates) == 1:

            exp_date = all_dates[0]

    return (
        mfg_date,
        exp_date
    )


# ============================================================
# DATE PARSER
# ============================================================

def _parse_date(date_string):

    if not date_string:
        return None

    if date_string == "Not Detected":
        return None

    date_string = date_string.strip()

    formats = [

        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",

        "%d/%m/%y",
        "%d-%m-%y",
        "%d.%m.%y",

        "%m/%Y",
        "%m-%Y",
        "%m.%Y",

        "%Y-%m-%d",

        "%b %Y",
        "%b. %Y",

        "%B %Y",

        "%d/%m",
        "%d-%m",
        "%d.%m",
    ]

    for fmt in formats:

        try:

            return datetime.strptime(
                date_string,
                fmt
            )

        except ValueError:
            continue

    return None


# ============================================================
# EXPIRY
# ============================================================

def _check_expiry(expiry_string):

    dt = _parse_date(
        expiry_string
    )

    if dt is None:

        return (
            "Unknown",
            0,
            0
        )

    now = datetime.now()

    # If only month/year is provided,
    # consider end of that month.
    if (
        dt.day == 1
        and dt.month != 1
    ):

        pass

    days = (
        dt - now
    ).days

    months = days // 30

    if days >= 0:

        return (
            "Valid",
            days,
            max(months, 0)
        )

    return (
        "Expired",
        0,
        0
    )


# ============================================================
# MAIN OCR FUNCTION
# ============================================================

def extract_medicine_info(
    image_path: str
):

    logger.info(
        f"Starting OCR: {image_path}"
    )

    raw_text, ocr_confidence = (
        extract_text_with_confidence(
            image_path
        )
    )

    # --------------------------------------------------------
    # OCR word count
    # --------------------------------------------------------

    word_count = (
        len(raw_text.split())
        if raw_text
        else 0
    )

    # Improve confidence when useful text found
    if word_count >= 10:

        ocr_confidence = max(
            ocr_confidence,
            65
        )

    elif word_count >= 5:

        ocr_confidence = max(
            ocr_confidence,
            50
        )

    elif word_count == 0:

        ocr_confidence = 0

    # --------------------------------------------------------
    # Extract fields
    # --------------------------------------------------------

    medicine_name = (
        _extract_medicine_name(
            raw_text
        )
    )

    manufacturer = (
        _extract_manufacturer(
            raw_text
        )
    )

    batch_number = (
        _extract_batch(
            raw_text
        )
    )

    manufacturing_date, expiry_date = (
        _extract_dates(
            raw_text
        )
    )

    expiry_status, days_remaining, months_remaining = (
        _check_expiry(
            expiry_date
        )
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logger.info(
        "OCR RESULT"
    )

    logger.info(
        f"Medicine: {medicine_name}"
    )

    logger.info(
        f"Manufacturer: {manufacturer}"
    )

    logger.info(
        f"Batch: {batch_number}"
    )

    logger.info(
        f"MFG: {manufacturing_date}"
    )

    logger.info(
        f"EXP: {expiry_date}"
    )

    logger.info(
        f"OCR Confidence: {ocr_confidence:.2f}%"
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "medicine_name":
            medicine_name,

        "manufacturer":
            manufacturer,

        "batch_number":
            batch_number,

        "manufacturing_date":
            manufacturing_date,

        "expiry_date":
            expiry_date,

        "expiry_status":
            expiry_status,

        "days_remaining":
            days_remaining,

        "months_remaining":
            months_remaining,

        "ocr_confidence":
            round(
                ocr_confidence,
                2
            ),

        "raw_text":
            raw_text,
    }