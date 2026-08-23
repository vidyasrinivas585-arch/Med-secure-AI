"""
model/evaluate.py
MedSecure AI - Model Evaluation

Evaluates the trained MobileNetV2 model using the
separate TEST dataset.

Dataset structure:

dataset/
├── train/
│   ├── counterfeit/
│   └── genuine/
├── val/
│   ├── counterfeit/
│   └── genuine/
└── test/
    ├── counterfeit/
    └── genuine/

Usage:
python model/evaluate.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset"

TEST_DIR = os.path.join(
    DATASET_DIR,
    "test"
)

MODEL_PATH = os.path.join(
    "trained_model",
    "medicine_classifier.keras"
)

CLASS_INDEX_PATH = os.path.join(
    "trained_model",
    "class_indices.json"
)

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

REPORT_DIR = os.path.join(
    "trained_model",
    "evaluation"
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    print("\n")
    print("=" * 60)

    print(
        "MedSecure AI - Model Evaluation"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Create report directory
    # --------------------------------------------------------

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    print(
        f"\nLoading model from: {MODEL_PATH}"
    )


    if not os.path.exists(MODEL_PATH):

        print(
            "❌ Trained model not found."
        )

        print(
            "Run: python model/train.py"
        )

        return


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        MODEL_PATH
    )


    print(
        "✅ Model loaded."
    )


    # --------------------------------------------------------
    # Check TEST directory
    # --------------------------------------------------------

    if not os.path.exists(TEST_DIR):

        print(
            f"❌ Test directory not found: {TEST_DIR}"
        )

        return


    # --------------------------------------------------------
    # Test data generator
    # --------------------------------------------------------

    datagen = ImageDataGenerator(

        preprocessing_function=preprocess_input
    )


    test_gen = datagen.flow_from_directory(

        TEST_DIR,

        target_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        class_mode="binary",

        shuffle=False
    )


    # --------------------------------------------------------
    # Class indices
    # --------------------------------------------------------

    class_indices = test_gen.class_indices


    print("\n")
    print("=" * 60)

    print(
        "CLASS INDICES"
    )

    print("=" * 60)

    print(
        class_indices
    )


    expected_mapping = {
        "counterfeit": 0,
        "genuine": 1
    }


    if class_indices != expected_mapping:

        print(
            "\n⚠️ WARNING:"
        )

        print(
            "Unexpected class mapping detected."
        )

    else:

        print(
            "✅ Correct mapping:"
        )

        print(
            "counterfeit = 0"
        )

        print(
            "genuine = 1"
        )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)

    print(
        "RUNNING TEST PREDICTIONS"
    )

    print("=" * 60)


    y_prob = model.predict(

        test_gen,

        verbose=1
    ).flatten()


    # --------------------------------------------------------
    # Convert probability to class
    # --------------------------------------------------------

    y_pred = (

        y_prob >= 0.5

    ).astype(int)


    y_true = test_gen.classes


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = [
        "Counterfeit",
        "Genuine"
    ]


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n")
    print("=" * 60)

    print(
        "CLASSIFICATION REPORT"
    )

    print("=" * 60)


    report = classification_report(

        y_true,

        y_pred,

        target_names=labels,

        zero_division=0
    )


    print(
        report
    )


    # Save report

    report_path = os.path.join(

        REPORT_DIR,

        "classification_report.txt"
    )


    with open(

        report_path,

        "w"
    ) as f:

        f.write(report)


    # ========================================================
    # METRICS
    # ========================================================

    acc = accuracy_score(

        y_true,

        y_pred
    )


    precision = precision_score(

        y_true,

        y_pred,

        zero_division=0
    )


    recall = recall_score(

        y_true,

        y_pred,

        zero_division=0
    )


    f1 = f1_score(

        y_true,

        y_pred,

        zero_division=0
    )


    metrics = {

        "accuracy": float(acc),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1)

    }


    print("\n")
    print("=" * 60)

    print(
        "FINAL METRICS"
    )

    print("=" * 60)


    print(
        f"Accuracy:  {acc:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )


    # Save metrics

    metrics_path = os.path.join(

        REPORT_DIR,

        "metrics.json"
    )


    with open(

        metrics_path,

        "w"
    ) as f:

        json.dump(

            metrics,

            f,

            indent=4
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        y_true,

        y_pred
    )


    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=labels
    )


    fig, ax = plt.subplots(

        figsize=(6, 5)
    )


    disp.plot(

        ax=ax,

        colorbar=False,

        cmap="Blues"
    )


    ax.set_title(

        "Confusion Matrix - Medicine Authenticity Classifier"
    )


    plt.tight_layout()


    confusion_path = os.path.join(

        REPORT_DIR,

        "confusion_matrix.png"
    )


    plt.savefig(

        confusion_path,

        dpi=150
    )


    plt.close()


    print(
        "\n✅ Confusion matrix saved."
    )


    # ========================================================
    # ROC CURVE
    # ========================================================

    fpr, tpr, _ = roc_curve(

        y_true,

        y_prob
    )


    roc_auc = auc(

        fpr,

        tpr
    )


    plt.figure(

        figsize=(6, 5)
    )


    plt.plot(

        fpr,

        tpr,

        color="royalblue",

        lw=2,

        label=(
            f"ROC curve "
            f"(AUC = {roc_auc:.3f})"
        )
    )


    plt.plot(

        [0, 1],

        [0, 1],

        color="gray",

        linestyle="--"
    )


    plt.xlim(

        [0, 1]
    )


    plt.ylim(

        [0, 1.02]
    )


    plt.xlabel(
        "False Positive Rate"
    )


    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(

        "ROC Curve - Medicine Authenticity Classifier"
    )


    plt.legend(

        loc="lower right"
    )


    plt.tight_layout()


    roc_path = os.path.join(

        REPORT_DIR,

        "roc_curve.png"
    )


    plt.savefig(

        roc_path,

        dpi=150
    )


    plt.close()


    print(
        f"✅ ROC curve saved. "
        f"AUC = {roc_auc:.4f}"
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 60)

    print(
        "EVALUATION COMPLETED"
    )

    print("=" * 60)


    print(
        f"Accuracy  : {acc * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : {f1 * 100:.2f}%"
    )

    print(
        f"ROC AUC   : {roc_auc:.4f}"
    )


    print("\nEvaluation files saved to:")

    print(
        REPORT_DIR
    )


    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate()