"""
model/train.py
MobileNetV2 Transfer Learning trainer for counterfeit medicine detection.

Usage:
    python model/train.py

Dataset expected at:
    dataset/
        genuine/      ← Class 0
        counterfeit/  ← Class 1
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ── TensorFlow / Keras imports ──────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)

# ── Scikit-learn ─────────────────────────────────────────────────────────────
from sklearn.utils.class_weight import compute_class_weight

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS_FROZEN = 10   # Train only top layers first
EPOCHS_FINE   = 20   # Then unfreeze last N base layers
FINE_TUNE_AT  = 100  # Unfreeze from this layer index onward
LEARNING_RATE = 1e-4
DATASET_DIR   = "dataset"
MODEL_DIR     = "trained_model"
MODEL_PATH    = os.path.join(MODEL_DIR, "medicine_classifier.h5")


def build_data_generators():
    """
    Create ImageDataGenerators with augmentation for train/val/test splits.
    80% train | 10% val | 10% test  (val_split=0.2 → 80 train / 20 val/test)
    """

    # Training augmentation — mirrors real-world capture variations
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
        validation_split=0.2,
    )

    # Validation / test — only normalise, no augmentation
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = test_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    print(f"Classes: {train_gen.class_indices}")   # {counterfeit: 0/1, genuine: 0/1}
    return train_gen, val_gen


def build_model() -> Model:
    """
    Build transfer-learning model on top of MobileNetV2 pre-trained on ImageNet.

    Architecture:
        MobileNetV2 (frozen) → GlobalAveragePooling → BN → Dropout → Dense(256)
        → BN → Dropout → Dense(1, sigmoid)
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = False  # Freeze all base layers initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)  # Binary: 0=Genuine, 1=Counterfeit

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model


def get_callbacks(phase: str) -> list:
    """Return training callbacks."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    return [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir=f"logs/{phase}"),
    ]


def plot_history(history, phase: str):
    """Save accuracy / loss plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title(f"{phase} – Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title(f"{phase} – Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/{phase}_history.png")
    print(f"Saved training plot: {MODEL_DIR}/{phase}_history.png")
    plt.close()


def train():
    print("=" * 60)
    print("  MedSecure AI — MobileNetV2 Training")
    print("=" * 60)

    # Validate dataset
    for cls in ["genuine", "counterfeit"]:
        path = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(path) or len(os.listdir(path)) == 0:
            print(f"❌ Missing or empty dataset folder: {path}")
            print("   Please add images before training.")
            sys.exit(1)

    train_gen, val_gen = build_data_generators()
    model, base_model = build_model()

    # Compute class weights to handle imbalanced datasets
    labels = train_gen.classes
    class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")

    # ── Phase 1: Train top layers (base frozen) ───────────────────────────
    print("\n📌 Phase 1: Training top layers with frozen MobileNetV2 base...")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FROZEN,
        callbacks=get_callbacks("phase1"),
        class_weight=class_weight_dict,
    )
    plot_history(history1, "phase1")

    # ── Phase 2: Fine-tune — unfreeze last layers of base ─────────────────
    print(f"\n📌 Phase 2: Fine-tuning from layer {FINE_TUNE_AT} onward...")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),  # Lower LR for fine-tune
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINE,
        callbacks=get_callbacks("phase2"),
        class_weight=class_weight_dict,
    )
    plot_history(history2, "phase2")

    # ── Save final model ──────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")

    # Save class indices for inference
    import json
    with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
        json.dump(train_gen.class_indices, f)
    print(f"✅ Class indices saved to: {MODEL_DIR}/class_indices.json")


if __name__ == "__main__":
    train()
