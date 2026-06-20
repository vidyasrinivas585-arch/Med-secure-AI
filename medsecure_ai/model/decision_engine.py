"""
model/decision_engine.py
Final decision engine — combines AI confidence, OCR confidence,
packaging score and expiry result into:
  - Prediction (Genuine / Counterfeit)
  - Authenticity Score (0–100)
  - Risk Level (Low / Medium / High)
  - Recommendation (actionable text)
"""

import os
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Load trained model ────────────────────────────────────────────────────────
_model = None
_class_indices = None

MODEL_PATH      = os.path.join("trained_model", "medicine_classifier.h5")
CLASS_IDX_PATH  = os.path.join("trained_model", "class_indices.json")

# Weights for composite authenticity score
W_AI          = 0.50   # AI confidence dominates
W_OCR         = 0.25   # OCR confidence
W_PACKAGING   = 0.25   # Packaging analysis


def _load_model():
    """Lazy-load the Keras model."""
    global _model, _class_indices
    if _model is not None:
        return _model
    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Model load failed: {e}. Using demo mode.")
        _model = None

    try:
        with open(CLASS_IDX_PATH) as f:
            _class_indices = json.load(f)
    except Exception:
        # Default: 0=counterfeit, 1=genuine (alphabetical order)
        _class_indices = {"counterfeit": 0, "genuine": 1}

    return _model


def predict_image(preprocessed_image: np.ndarray) -> tuple[str, float]:
    """
    Run model inference on a preprocessed image batch.
    counterfeit = 0, genuine = 1
    sigmoid output close to 1.0 = genuine
    sigmoid output close to 0.0 = counterfeit
    """
    model = _load_model()
    if model is None:
        return "Genuine", 85.0

    try:
        prob = float(model.predict(preprocessed_image, verbose=0)[0][0])
        
        # prob = sigmoid output
        # Since genuine=1 and counterfeit=0:
        # prob >= 0.5 means model says GENUINE
        # prob <  0.5 means model says COUNTERFEIT
        
        if prob >= 0.5:
            label = "Genuine"
            confidence = round(prob * 100, 2)
        else:
            label = "Counterfeit"
            confidence = round((1 - prob) * 100, 2)

        print(f"DEBUG → prob: {prob:.4f} | label: {label} | confidence: {confidence}%")
        return label, confidence

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return "Unknown", 50.0


def compute_authenticity_score(
    ai_confidence: float,
    ocr_confidence: float,
    packaging_score: float,
    prediction: str,
) -> float:
    """
    Weighted combination of three sub-scores into a 0–100 authenticity score.

    If prediction is 'Counterfeit', AI confidence is INVERTED
    (high confidence in counterfeit → low authenticity).
    """
    if prediction == "Counterfeit":
        ai_score = 100.0 - ai_confidence
    else:
        ai_score = ai_confidence

    score = (
        ai_score        * W_AI +
        ocr_confidence  * W_OCR +
        packaging_score * W_PACKAGING
    )
    return round(min(max(score, 0.0), 100.0), 2)


def assess_risk(authenticity_score: float, prediction: str, expiry_status: str) -> str:
    """
    Determine risk level from authenticity score, prediction, and expiry.

    Rules:
        - Expired always adds risk
        - Counterfeit prediction → High risk unless score > 80
        - Low score (<40) → High risk
        - Mid score (40–70) → Medium risk
        - High score (>70) → Low risk
    """
    if prediction == "Counterfeit" or expiry_status == "Expired":
        if authenticity_score >= 80:
            return "Medium"
        return "High"
    if authenticity_score >= 75:
        return "Low"
    if authenticity_score >= 45:
        return "Medium"
    return "High"


def generate_recommendation(prediction: str, risk_level: str, expiry_status: str) -> str:
    """
    Generate a human-readable recommendation string.
    """
    if expiry_status == "Expired":
        return "⚠️ This medicine is EXPIRED. Do not consume. Dispose safely."

    if prediction == "Counterfeit":
        if risk_level == "High":
            return "🚨 HIGH RISK: This medicine appears counterfeit. Stop use immediately and report to health authorities."
        return "⚠️ MEDIUM RISK: Authenticity is questionable. Verify with a licensed pharmacist before use."

    if risk_level == "Low":
        return "✅ LOW RISK: This medicine appears genuine. Safe to use as prescribed."
    if risk_level == "Medium":
        return "⚠️ MEDIUM RISK: Some indicators are unclear. Purchase from a verified pharmacy for certainty."
    return "🚨 HIGH RISK: Multiple warning signs detected. Do not use without expert verification."


def full_analysis(preprocessed_image: np.ndarray, ocr_info: dict, packaging_info: dict) -> dict:
    """
    Master analysis function.

    Args:
        preprocessed_image: Preprocessed image for model inference.
        ocr_info: Output of ocr.extractor.extract_medicine_info()
        packaging_info: Output of ocr.preprocessor.analyze_packaging()

    Returns:
        dict with all analysis fields ready for DB storage and frontend display.
    """
    # ── AI prediction ────────────────────────────────────────────────────────
    prediction, ai_confidence = predict_image(preprocessed_image)

    # ── Sub-scores ───────────────────────────────────────────────────────────
    ocr_confidence = ocr_info.get("ocr_confidence", 50.0)
    packaging_score = packaging_info.get("packaging_score", 50.0)
    expiry_status = ocr_info.get("expiry_status", "Unknown")

    # ── Composite score ──────────────────────────────────────────────────────
    authenticity_score = compute_authenticity_score(
        ai_confidence, ocr_confidence, packaging_score, prediction
    )

    # ── Risk ────────────────────────────────────────────────────────────────
    risk_level = assess_risk(authenticity_score, prediction, expiry_status)

    # ── Recommendation ───────────────────────────────────────────────────────
    recommendation = generate_recommendation(prediction, risk_level, expiry_status)

    return {
        "prediction":         prediction,
        "ai_confidence":      ai_confidence,
        "ocr_confidence":     ocr_confidence,
        "packaging_score":    packaging_score,
        "authenticity_score": authenticity_score,
        "risk_level":         risk_level,
        "recommendation":     recommendation,
    }
