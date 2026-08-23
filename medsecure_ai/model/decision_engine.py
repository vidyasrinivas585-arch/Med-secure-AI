"""
model/decision_engine.py
MedSecure AI - Final Decision Engine
Combines AI, OCR, Packaging, Expiry into final results.
FIXED: recommendation is always a plain string for MongoDB + Jinja2 compatibility.
"""

import os
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join("trained_model", "medicine_classifier.keras")
MODEL_PATH_H5  = os.path.join("trained_model", "medicine_classifier.h5")
CLASS_IDX_PATH = os.path.join("trained_model", "class_indices.json")

W_AI        = 0.70
W_OCR       = 0.15
W_PACKAGING = 0.15

_model         = None
_class_indices = None


# ── Model loader ──────────────────────────────────────────────────────────────
def _load_model():
    global _model, _class_indices
    if _model is not None:
        return _model

    try:
        import tensorflow as tf

        # Try .keras first, then .h5
        path = MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_PATH_H5
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        _model = tf.keras.models.load_model(path)
        logger.info(f"✅ Model loaded: {path}")
        print(f"✅ Model loaded: {path}")

    except Exception as e:
        logger.error(f"Model load failed: {e}")
        _model = None
        raise RuntimeError(f"Could not load model: {e}")

    try:
        if os.path.exists(CLASS_IDX_PATH):
            with open(CLASS_IDX_PATH, "r") as f:
                _class_indices = json.load(f)
            print(f"✅ Class mapping: {_class_indices}")
        else:
            _class_indices = {"counterfeit": 0, "genuine": 1}
            print("⚠️  Using default class mapping")
    except Exception as e:
        _class_indices = {"counterfeit": 0, "genuine": 1}
        logger.warning(f"Class mapping load failed: {e}")

    return _model


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_image(preprocessed_image: np.ndarray) -> tuple:
    """Run AI model. Returns (label, confidence_0_to_100)."""
    model = _load_model()

    image = np.asarray(preprocessed_image, dtype=np.float32)
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    output = model.predict(image, verbose=0)
    print(f"DEBUG → Raw output: {output} shape: {output.shape}")

    # Binary sigmoid
    if output.shape[-1] == 1:
        prob = float(output[0][0])
        genuine_idx = (_class_indices or {}).get("genuine", 1)
        if genuine_idx == 1:
            label      = "Genuine"      if prob >= 0.5 else "Counterfeit"
            confidence = prob * 100      if prob >= 0.5 else (1 - prob) * 100
        else:
            label      = "Counterfeit"  if prob >= 0.5 else "Genuine"
            confidence = prob * 100      if prob >= 0.5 else (1 - prob) * 100

    # Softmax 2-class
    elif output.shape[-1] == 2:
        probs   = output[0]
        ci      = _class_indices or {"counterfeit": 0, "genuine": 1}
        i2c     = {int(v): k for k, v in ci.items()}
        pred_i  = int(np.argmax(probs))
        cls     = i2c.get(pred_i, "genuine" if pred_i == 1 else "counterfeit")
        label      = "Genuine" if cls.lower() == "genuine" else "Counterfeit"
        confidence = float(probs[pred_i]) * 100

    else:
        raise ValueError(f"Unexpected output shape: {output.shape}")

    confidence = round(confidence, 2)
    print(f"DEBUG → {label} | {confidence}%")
    return label, confidence


# ── Authenticity score ────────────────────────────────────────────────────────
def compute_authenticity_score(ai_confidence: float, ocr_confidence: float,
                                packaging_score: float, prediction: str) -> float:
    ai_score = (100.0 - ai_confidence) if prediction == "Counterfeit" else ai_confidence
    score = (ai_score * W_AI +
             ocr_confidence  * W_OCR +
             packaging_score * W_PACKAGING)
    return round(min(max(score, 0.0), 100.0), 2)


# ── Risk ──────────────────────────────────────────────────────────────────────
def assess_risk(authenticity_score: float, prediction: str,
                expiry_status: str) -> str:
    if expiry_status == "Expired":
        return "High"
    if prediction in ("Counterfeit", "Expired"):
        return "High"
    if prediction == "Suspicious":
        return "Medium" if authenticity_score >= 70 else "High"
    if authenticity_score >= 75:
        return "Low"
    if authenticity_score >= 45:
        return "Medium"
    return "High"


# ── Refine prediction ─────────────────────────────────────────────────────────
def refine_prediction(raw_prediction: str, ai_confidence: float,
                      authenticity_score: float, expiry_status: str) -> str:
    if expiry_status == "Expired":
        return "Expired"
    if raw_prediction == "Genuine" and ai_confidence >= 70:
        return "Genuine"
    if raw_prediction == "Counterfeit" and ai_confidence >= 70:
        return "Counterfeit"
    return "Suspicious"


# ── Recommendation — ALWAYS returns a plain string ────────────────────────────
def generate_recommendation(prediction: str, risk_level: str = "High",
                             expiry_status: str = "Unknown",
                             medicine_name: str = "This medicine",
                             **kwargs) -> str:
    """
    Always returns a plain string.
    Accepts extra kwargs so it is compatible with the old
    recommendation_engine signature without crashing.
    """
    name = medicine_name if medicine_name and medicine_name != "Medicine" \
           else "This medicine"

    if expiry_status == "Expired":
        return (f"⚠️ EXPIRED: {name} has expired. "
                "Do not consume. Dispose safely and buy a fresh supply.")

    if prediction in ("Counterfeit", "Expired"):
        return (f"🚨 HIGH RISK: {name} appears counterfeit. "
                "Stop use immediately, keep the packaging, and report to "
                "your nearest pharmacist or health authority.")

    if prediction == "Suspicious":
        return (f"⚠️ UNCERTAIN: {name} could not be fully verified. "
                "Upload a clearer image or consult a licensed pharmacist "
                "before consuming.")

    if prediction == "Genuine":
        if risk_level == "Low":
            return (f"✅ LOW RISK: {name} appears genuine. "
                    "Safe to use as prescribed by your physician.")
        if risk_level == "Medium":
            return (f"⚠️ MEDIUM RISK: {name} appears likely genuine but some "
                    "indicators are unclear. Purchase from a verified pharmacy "
                    "for complete assurance.")
        return (f"⚠️ HIGH RISK: {name} shows warning signs. "
                "Do not consume without pharmacist verification.")

    return ("Unable to determine authenticity. "
            "Consult a licensed pharmacist before use.")


# ── Safety guidance ───────────────────────────────────────────────────────────
def generate_safety_guidance(prediction: str) -> dict:
    """Returns structured safety guidance dict (used by templates)."""
    if prediction == "Genuine":
        return {
            "is_safe": True,
            "title": "Medicine Appears Genuine",
            "message": "Based on AI analysis, this medicine appears authentic.",
            "actions": [
                "Purchase medicines from licensed pharmacies.",
                "Verify batch number and expiry date.",
                "Consult a doctor or pharmacist if concerned.",
            ]
        }
    if prediction == "Expired":
        return {
            "is_safe": False,
            "title": "Medicine Has Expired",
            "message": "This medicine appears to be expired. Do not consume.",
            "actions": [
                "Do not consume the medicine.",
                "Dispose of expired medicine safely.",
                "Purchase a fresh supply from a licensed pharmacy.",
            ]
        }
    if prediction == "Counterfeit":
        return {
            "is_safe": False,
            "title": "Possible Counterfeit Medicine",
            "message": "The medicine shows potential counterfeit indicators.",
            "actions": [
                "Do not consume this medicine.",
                "Keep the medicine and packaging.",
                "Verify the batch number and manufacturer.",
                "Contact a licensed pharmacist.",
                "Purchase a verified medicine from a licensed pharmacy.",
            ]
        }
    return {
        "is_safe": False,
        "title": "Medicine Requires Verification",
        "message": "Analysis produced uncertain results.",
        "actions": [
            "Do not use the medicine until verified.",
            "Upload a clearer image and try again.",
            "Consult a licensed pharmacist.",
        ]
    }


# ── Explainability ────────────────────────────────────────────────────────────
def generate_explanation(ocr_info: dict, packaging_info: dict,
                          prediction: str, ai_confidence: float,
                          ocr_confidence: float, packaging_score: float,
                          qr_found: bool = False) -> dict:
    reasons = []

    reasons.append({"text": "QR code detected and readable" if qr_found
                    else "QR code missing or not detected", "flagged": not qr_found})

    exp = ocr_info.get("expiry_date", "N/A")
    if exp in ("N/A", "Not Detected", ""):
        reasons.append({"text": "Expiry date unreadable", "flagged": True})
    elif ocr_info.get("expiry_status") == "Expired":
        reasons.append({"text": "Medicine has expired", "flagged": True})
    else:
        reasons.append({"text": "Expiry date readable and valid", "flagged": False})

    batch = ocr_info.get("batch_number", "Not Detected")
    reasons.append({"text": "Batch number extracted" if batch not in
                    ("Not Detected", "N/A", "") else "Batch number not detected",
                    "flagged": batch in ("Not Detected", "N/A", "")})

    reasons.append({"text": f"Packaging quality: {'good' if packaging_score >= 70 else 'low'}",
                    "flagged": packaging_score < 70})

    reasons.append({"text": f"OCR confidence: {ocr_confidence:.0f}%",
                    "flagged": ocr_confidence < 50})

    mfr = ocr_info.get("manufacturer", "Not Detected")
    reasons.append({"text": "Manufacturer detected" if mfr not in
                    ("Not Detected", "N/A", "") else "Manufacturer not identified",
                    "flagged": mfr in ("Not Detected", "N/A", "")})

    flagged = prediction in ("Counterfeit", "Expired", "Suspicious")
    reasons.append({"text": f"AI model: {prediction} ({ai_confidence:.0f}% confidence)",
                    "flagged": flagged})

    return {
        "reasons": reasons,
        "overall_confidence": round(ai_confidence*.5 + ocr_confidence*.25 + packaging_score*.25, 1),
        "flagged_count": sum(1 for r in reasons if r["flagged"])
    }


def needs_pharmacy_locator(prediction: str) -> bool:
    return prediction in ("Counterfeit", "Suspicious", "Expired")


# ── Full analysis ─────────────────────────────────────────────────────────────
def full_analysis(preprocessed_image: np.ndarray,
                  ocr_info: dict, packaging_info: dict) -> dict:
    """
    Master analysis. All returned values are plain Python types
    safe for MongoDB storage and Jinja2 rendering.
    """
    prediction, ai_confidence = predict_image(preprocessed_image)
    ocr_confidence  = float(ocr_info.get("ocr_confidence", 0.0))
    packaging_score = float(packaging_info.get("packaging_score", 50.0))
    expiry_status   = str(ocr_info.get("expiry_status", "Unknown"))
    medicine_name   = str(ocr_info.get("medicine_name", "Medicine") or "Medicine").strip() or "Medicine"

    authenticity_score = compute_authenticity_score(
        ai_confidence, ocr_confidence, packaging_score, prediction)
    final_prediction = refine_prediction(
        prediction, ai_confidence, authenticity_score, expiry_status)
    risk_level = assess_risk(
        authenticity_score, final_prediction, expiry_status)

    # ── Recommendation — ALWAYS a plain string ────────────────────────────
    # Try importing the recommendation engine if it exists
    recommendation_str = ""
    try:
        from model.recommendation_engine import generate_recommendation as ext_rec
        raw_rec = ext_rec(
            prediction=final_prediction,
            ai_confidence=ai_confidence,
            authenticity_score=authenticity_score,
            expiry_status=expiry_status,
            ocr_confidence=ocr_confidence,
            packaging_score=packaging_score,
            medicine_name=medicine_name,
        )
        # If it returned a dict, extract the message
        if isinstance(raw_rec, dict):
            recommendation_str = raw_rec.get("message", str(raw_rec))
        else:
            recommendation_str = str(raw_rec)
    except Exception:
        # Fallback to built-in
        recommendation_str = generate_recommendation(
            final_prediction, risk_level, expiry_status, medicine_name)

    # Safety net — must be a string
    if not isinstance(recommendation_str, str):
        recommendation_str = str(recommendation_str)

    safety_guidance = generate_safety_guidance(final_prediction)
    qr_found        = bool(ocr_info.get("qr_found", False))
    explanation     = generate_explanation(
        ocr_info, packaging_info, final_prediction,
        ai_confidence, ocr_confidence, packaging_score, qr_found)

    print(f"DEBUG → Final: {final_prediction} | Score: {authenticity_score} | Risk: {risk_level}")

    return {
        "prediction":          final_prediction,
        "raw_prediction":      prediction,
        "ai_confidence":       round(ai_confidence, 2),
        "ocr_confidence":      round(ocr_confidence, 2),
        "packaging_score":     round(packaging_score, 2),
        "authenticity_score":  round(authenticity_score, 2),
        "risk_level":          risk_level,
        "recommendation":      recommendation_str,       # ← always str
        "medicine_name":       medicine_name,
        "safety_guidance":     safety_guidance,
        "explanation":         explanation,
        "show_pharmacy_locator": needs_pharmacy_locator(final_prediction),
    }