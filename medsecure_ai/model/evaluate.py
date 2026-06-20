"""
model/evaluate.py
Evaluates the trained MobileNetV2 model.
Generates accuracy, precision, recall, F1, confusion matrix, and ROC curve.

Usage:
    python model/evaluate.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    classification_report,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset"
MODEL_PATH  = "trained_model/medicine_classifier.h5"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
REPORT_DIR  = "trained_model/evaluation"


def evaluate():
    os.makedirs(REPORT_DIR, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading model from {MODEL_PATH} ...")
    if not os.path.exists(MODEL_PATH):
        print("❌ Trained model not found. Run model/train.py first.")
        return
    model = load_model(MODEL_PATH)
    print("✅ Model loaded.")

    # ── Data generator (no augmentation) ───────────────────────────────────
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
    )
    test_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )

    class_indices = test_gen.class_indices
    print(f"Class indices: {class_indices}")

    # ── Predictions ─────────────────────────────────────────────────────────
    print("Running predictions ...")
    y_prob = model.predict(test_gen, verbose=1).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = test_gen.classes

    # Determine which index is 'Genuine' and which is 'Counterfeit'
    labels = ["Genuine", "Counterfeit"]
    if class_indices.get("genuine", 0) == 1:
        labels = ["Counterfeit", "Genuine"]

    # ── Classification Report ───────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    report = classification_report(y_true, y_pred, target_names=labels)
    print(report)

    # Save report as text
    with open(f"{REPORT_DIR}/classification_report.txt", "w") as f:
        f.write(report)

    # ── Scalar metrics ──────────────────────────────────────────────────────
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    with open(f"{REPORT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Confusion Matrix ────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix – Medicine Authenticity Classifier")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"\n✅ Confusion matrix saved.")

    # ── ROC Curve ───────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="royalblue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Medicine Authenticity Classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/roc_curve.png", dpi=150)
    plt.close()
    print(f"✅ ROC curve saved. AUC = {roc_auc:.4f}")

    print(f"\nAll evaluation artifacts saved to: {REPORT_DIR}/")


if __name__ == "__main__":
    evaluate()
