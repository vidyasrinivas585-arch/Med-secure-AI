"""
ocr/preprocessor.py
Image preprocessing pipeline using OpenCV.
Prepares medicine package images for OCR and model inference.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Target size for MobileNetV2
TARGET_SIZE = (224, 224)


def preprocess_for_model(image_path: str) -> np.ndarray | None:
    """
    Preprocess image exactly the same way as MobileNetV2
    training preprocessing.

    IMPORTANT:
    train.py uses:
        tensorflow.keras.applications.mobilenet_v2.preprocess_input

    Therefore inference must use the same preprocessing.
    """

    try:
        img = cv2.imread(image_path)

        if img is None:
            logger.error(
                f"Failed to load image: {image_path}"
            )
            return None

        # OpenCV BGR -> RGB
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # Resize exactly as training
        img = cv2.resize(
            img,
            TARGET_SIZE,
            interpolation=cv2.INTER_AREA
        )

        # Convert to float32
        img = img.astype(np.float32)

        # IMPORTANT:
        # MobileNetV2 preprocess_input converts:
        # [0, 255] -> [-1, 1]
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        img = preprocess_input(img)

        # Add batch dimension
        img = np.expand_dims(
            img,
            axis=0
        )

        return img

    except Exception as e:

        logger.error(
            f"Preprocessing for model failed: {e}"
        )

        return None

def preprocess_for_ocr(image_path: str) -> np.ndarray | None:
    """
    Preprocessing pipeline optimised for OCR accuracy.

    Steps:
        1. Load in grayscale
        2. Denoise
        3. Contrast enhancement (CLAHE)
        4. Adaptive thresholding → binary image
        5. Resize (upscale small images for better OCR)

    Args:
        image_path (str): Path to the input image file.

    Returns:
        np.ndarray: Processed grayscale image ready for EasyOCR.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # CLAHE – Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Adaptive thresholding for clean text extraction
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Upscale small images so EasyOCR reads fine print better
        h, w = binary.shape
        if w < 800:
            scale = 800 / w
            binary = cv2.resize(binary, (800, int(h * scale)), interpolation=cv2.INTER_CUBIC)

        return binary

    except Exception as e:
        logger.error(f"OCR preprocessing failed: {e}")
        return None


def analyze_packaging(image_path: str) -> dict:
    """
    Analyse packaging quality features to contribute to the authenticity score.

    Checks:
        - Colour consistency (variance in HSV saturation)
        - Edge sharpness (Laplacian variance — blurry = fake)
        - Text region density (ratio of text-like contours to image area)

    Args:
        image_path (str): Path to the input image.

    Returns:
        dict: {
            "packaging_score": float (0–100),
            "sharpness": float,
            "color_consistency": float,
            "text_density": float
        }
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"packaging_score": 50, "sharpness": 50, "color_consistency": 50, "text_density": 50}

        # ── 1. Sharpness (Laplacian variance) ──────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalise: score 100 at variance ≥ 500
        sharpness_score = min(100.0, (laplacian_var / 500.0) * 100.0)

        # ── 2. Colour consistency (low saturation variance → consistent branding) ─
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat_std = float(np.std(hsv[:, :, 1]))
        # Lower std → more consistent palette → higher score
        color_score = max(0.0, 100.0 - sat_std)

        # ── 3. Text density (contour-based) ──────────────────────────────
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = img.shape[0] * img.shape[1]
        text_area = sum(cv2.contourArea(c) for c in contours if 5 < cv2.contourArea(c) < 5000)
        text_density = min(100.0, (text_area / total_area) * 1000.0)

        # ── Composite packaging score (weighted average) ──────────────────
        packaging_score = (
            sharpness_score * 0.40 +
            color_score     * 0.35 +
            text_density    * 0.25
        )

        return {
            "packaging_score": round(packaging_score, 2),
            "sharpness": round(sharpness_score, 2),
            "color_consistency": round(color_score, 2),
            "text_density": round(text_density, 2),
        }

    except Exception as e:
        logger.error(f"Packaging analysis failed: {e}")
        return {"packaging_score": 50, "sharpness": 50, "color_consistency": 50, "text_density": 50}
