"""
model/translator.py
Multi-language support for MedSecure AI.
Translates key result fields between English and Kannada.
Uses deep-translator (Google Translate backend, free tier).
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# ── Static translation dictionary (fallback when API is unavailable) ──────────
# Pre-translated common medicine analysis terms (English → Kannada)
STATIC_KN = {
    # Predictions
    "Genuine":      "ಅಸಲಿ",
    "Counterfeit":  "ನಕಲಿ",
    "Suspicious":   "ಸಂಶಯಾಸ್ಪದ",
    "Expired":      "ಅವಧಿ ಮೀರಿದ",
    "Unknown":      "ಅಜ್ಞಾತ",
    # Risk levels
    "Low":          "ಕಡಿಮೆ",
    "Medium":       "ಮಧ್ಯಮ",
    "High":         "ಹೆಚ್ಚು",
    # Expiry
    "Valid":        "ಮಾನ್ಯ",
    "Expired":      "ಅವಧಿ ಮೀರಿದ",
    "Not Detected": "ಪತ್ತೆಯಾಗಿಲ್ಲ",
    # Risk descriptions
    "Low Risk":     "ಕಡಿಮೆ ಅಪಾಯ",
    "Medium Risk":  "ಮಧ್ಯಮ ಅಪಾಯ",
    "High Risk":    "ಹೆಚ್ಚಿನ ಅಪಾಯ",
    # UI labels
    "Medicine Name":    "ಔಷಧದ ಹೆಸರು",
    "Manufacturer":     "ತಯಾರಕ",
    "Batch Number":     "ಬ್ಯಾಚ್ ಸಂಖ್ಯೆ",
    "Expiry Date":      "ಮುಕ್ತಾಯ ದಿನಾಂಕ",
    "Authenticity Score": "ಸ್ವಾಯತ್ತತೆ ಅಂಕ",
    "Risk Level":       "ಅಪಾಯ ಮಟ್ಟ",
    "Recommendation":   "ಶಿಫಾರಸು",
    "Prediction":       "ಭವಿಷ್ಯ",
    "Expiry Status":    "ಮುಕ್ತಾಯ ಸ್ಥಿತಿ",
}


@lru_cache(maxsize=512)
def translate_to_kannada(text: str) -> str:
    """
    Translate a string to Kannada.
    First tries the static dictionary, then deep-translator API.

    Args:
        text (str): English input string.

    Returns:
        str: Kannada translation, or original text on failure.
    """
    if not text or not text.strip():
        return text

    # Static lookup first (fast + offline)
    if text in STATIC_KN:
        return STATIC_KN[text]

    # API translation
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="en", target="kn").translate(text)
        return translated if translated else text
    except Exception as e:
        logger.warning(f"Translation failed for '{text}': {e}")
        return text  # Graceful fallback


def translate_results(results: dict, language: str) -> dict:
    """
    Translate all user-facing fields in the results dict to the requested language.

    Args:
        results (dict): Analysis result dictionary.
        language (str): "en" or "kn".

    Returns:
        dict: Results with translated string fields.
    """
    if language != "kn":
        return results  # English is the base language

    translated = results.copy()

    fields_to_translate = [
        "medicine_name",
        "manufacturer",
        "prediction",
        "expiry_status",
        "risk_level",
        "recommendation",
    ]

    for field in fields_to_translate:
        value = results.get(field, "")
        if isinstance(value, str) and value:
            translated[field] = translate_to_kannada(value)

    # Translate safety guidance
    safety = results.get("safety_guidance", {})
    if safety:
        translated_safety = safety.copy()
        for key in ("title", "message"):
            if safety.get(key):
                translated_safety[key] = translate_to_kannada(safety[key])
        if safety.get("actions"):
            translated_safety["actions"] = [
                translate_to_kannada(a) for a in safety["actions"]
            ]
        translated["safety_guidance"] = translated_safety

    # Translate explanation reasons
    explanation = results.get("explanation", {})
    if explanation and explanation.get("reasons"):
        translated_exp = explanation.copy()
        translated_exp["reasons"] = [
            {**r, "text": translate_to_kannada(r.get("text", ""))}
            for r in explanation["reasons"]
        ]
        translated["explanation"] = translated_exp

    return translated


def get_ui_labels(language: str) -> dict:
    """
    Return UI label strings for the given language.

    Args:
        language (str): "en" or "kn".

    Returns:
        dict: Mapping of label keys to display strings.
    """
    labels_en = {
        "medicine_name":      "Medicine Name",
        "manufacturer":       "Manufacturer",
        "batch_number":       "Batch Number",
        "expiry_date":        "Expiry Date",
        "authenticity_score": "Authenticity Score",
        "risk_level":         "Risk Level",
        "recommendation":     "Recommendation",
        "prediction":         "Prediction",
        "expiry_status":      "Expiry Status",
        "safety_guidance":    "Safety Guidance",
        "nearby_pharmacies":  "Nearby Pharmacies",
        "reason_prediction":  "Reason for Prediction",
        "overall_confidence": "Overall Confidence",
        "download_pdf":       "Download PDF Report",
        "analyze_another":    "Analyze Another",
    }

    if language != "kn":
        return labels_en

    return {k: translate_to_kannada(v) for k, v in labels_en.items()}
