"""
model/train.py
MedSecure AI - Counterfeit vs Genuine Medicine Classifier

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

Model:
MobileNetV2 Transfer Learning

Class mapping:
counterfeit = 0
genuine    = 1
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS_FROZEN = 10
EPOCHS_FINE = 20

FINE_TUNE_AT = 100

LEARNING_RATE = 1e-4


# ============================================================
# DATASET PATHS
# ============================================================

DATASET_DIR = "dataset"

TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

VAL_DIR = os.path.join(
    DATASET_DIR,
    "val"
)

TEST_DIR = os.path.join(
    DATASET_DIR,
    "test"
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = "trained_model"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "medicine_classifier.keras"
)

CLASS_INDEX_PATH = os.path.join(
    MODEL_DIR,
    "class_indices.json"
)


# ============================================================
# REQUIRED CLASSES
# ============================================================

REQUIRED_CLASSES = [
    "counterfeit",
    "genuine"
]


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset():

    print("\nChecking dataset structure...")
    print("=" * 60)

    required_dirs = [
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR
    ]

    # --------------------------------------------------------
    # Check train / val / test folders
    # --------------------------------------------------------

    for directory in required_dirs:

        if not os.path.exists(directory):

            print(
                f"❌ Missing folder: {directory}"
            )

            return False


    # --------------------------------------------------------
    # Check class folders
    # --------------------------------------------------------

    for directory in required_dirs:

        for cls in REQUIRED_CLASSES:

            class_dir = os.path.join(
                directory,
                cls
            )

            if not os.path.exists(class_dir):

                print(
                    f"❌ Missing class folder: {class_dir}"
                )

                return False


            # ------------------------------------------------
            # Count images
            # ------------------------------------------------

            images = [
                f
                for f in os.listdir(class_dir)
                if f.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".webp"
                    )
                )
            ]


            print(
                f"{class_dir} → {len(images)} images"
            )


            if len(images) == 0:

                print(
                    f"❌ No images found in {class_dir}"
                )

                return False


    print("=" * 60)
    print("✅ Dataset structure is correct.")

    return True


# ============================================================
# DATA GENERATORS
# ============================================================

def build_data_generators():

    print("\nBuilding data generators...")
    print("=" * 60)


    # --------------------------------------------------------
    # Training augmentation
    # --------------------------------------------------------

    train_datagen = ImageDataGenerator(

        preprocessing_function=preprocess_input,

        rotation_range=15,

        zoom_range=0.15,

        width_shift_range=0.10,

        height_shift_range=0.10,

        brightness_range=[
            0.8,
            1.2
        ],

        horizontal_flip=True,

        fill_mode="nearest"
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_datagen = ImageDataGenerator(

        preprocessing_function=preprocess_input
    )


    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_datagen = ImageDataGenerator(

        preprocessing_function=preprocess_input
    )


    # --------------------------------------------------------
    # TRAIN GENERATOR
    # --------------------------------------------------------

    train_gen = train_datagen.flow_from_directory(

        TRAIN_DIR,

        target_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        class_mode="binary",

        shuffle=True,

        seed=42
    )


    # --------------------------------------------------------
    # VALIDATION GENERATOR
    # --------------------------------------------------------

    val_gen = val_datagen.flow_from_directory(

        VAL_DIR,

        target_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        class_mode="binary",

        shuffle=False
    )


    # --------------------------------------------------------
    # TEST GENERATOR
    # --------------------------------------------------------

    test_gen = test_datagen.flow_from_directory(

        TEST_DIR,

        target_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        class_mode="binary",

        shuffle=False
    )


    # --------------------------------------------------------
    # CLASS MAPPING
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("CLASS MAPPING")
    print("=" * 60)

    print(
        train_gen.class_indices
    )


    expected_mapping = {
        "counterfeit": 0,
        "genuine": 1
    }


    if train_gen.class_indices != expected_mapping:

        print(
            "\n⚠️ WARNING:"
        )

        print(
            "Unexpected class mapping detected:"
        )

        print(
            train_gen.class_indices
        )

    else:

        print(
            "\n✅ Correct class mapping:"
        )

        print(
            "counterfeit = 0"
        )

        print(
            "genuine = 1"
        )


    return (
        train_gen,
        val_gen,
        test_gen
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    print("\nLoading MobileNetV2...")
    print("=" * 60)


    # --------------------------------------------------------
    # MobileNetV2 base
    # --------------------------------------------------------

    base_model = MobileNetV2(

        weights="imagenet",

        include_top=False,

        input_shape=(
            224,
            224,
            3
        )
    )


    # Initially freeze base model

    base_model.trainable = False


    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    x = base_model.output


    x = GlobalAveragePooling2D()(x)


    x = BatchNormalization()(x)


    x = Dropout(
        0.3
    )(x)


    x = Dense(
        256,
        activation="relu"
    )(x)


    x = BatchNormalization()(x)


    x = Dropout(
        0.3
    )(x)


    output = Dense(
        1,
        activation="sigmoid"
    )(x)


    # --------------------------------------------------------
    # Complete model
    # --------------------------------------------------------

    model = Model(

        inputs=base_model.input,

        outputs=output
    )


    print(
        "✅ MobileNetV2 model created."
    )


    return (
        model,
        base_model
    )


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks(phase):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    return [

        ModelCheckpoint(

            MODEL_PATH,

            monitor="val_accuracy",

            save_best_only=True,

            verbose=1
        ),


        EarlyStopping(

            monitor="val_loss",

            patience=5,

            restore_best_weights=True,

            verbose=1
        ),


        ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=2,

            min_lr=1e-7,

            verbose=1
        ),


        TensorBoard(

            log_dir=f"logs/{phase}"
        )
    ]


# ============================================================
# PLOT TRAINING HISTORY
# ============================================================

def plot_history(
    history,
    phase
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Accuracy plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )


    plt.plot(

        history.history["accuracy"],

        label="Train Accuracy"
    )


    plt.plot(

        history.history["val_accuracy"],

        label="Validation Accuracy"
    )


    plt.title(
        f"{phase} Accuracy"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            MODEL_DIR,

            f"{phase}_accuracy.png"
        )
    )


    plt.close()


    # --------------------------------------------------------
    # Loss plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )


    plt.plot(

        history.history["loss"],

        label="Train Loss"
    )


    plt.plot(

        history.history["val_loss"],

        label="Validation Loss"
    )


    plt.title(
        f"{phase} Loss"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            MODEL_DIR,

            f"{phase}_loss.png"
        )
    )


    plt.close()


# ============================================================
# TRAIN MODEL
# ============================================================

def train():

    print("\n")
    print("=" * 60)

    print(
        "MedSecure AI"
    )

    print(
        "Counterfeit vs Genuine Medicine Training"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if not check_dataset():

        print(
            "\n❌ Dataset structure is incorrect."
        )

        return


    # --------------------------------------------------------
    # BUILD GENERATORS
    # --------------------------------------------------------

    (
        train_gen,
        val_gen,
        test_gen
    ) = build_data_generators()


    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    labels = train_gen.classes

    classes = np.unique(
        labels
    )


    class_weights = compute_class_weight(

        class_weight="balanced",

        classes=classes,

        y=labels
    )


    class_weight_dict = dict(

        zip(
            classes,
            class_weights
        )
    )


    print("\n")
    print("=" * 60)

    print(
        "CLASS WEIGHTS"
    )

    print("=" * 60)

    print(
        class_weight_dict
    )


    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    (
        model,
        base_model
    ) = build_model()


    # ========================================================
    # PHASE 1
    # ========================================================

    print("\n")
    print("=" * 60)

    print(
        "PHASE 1 - TRAINING CLASSIFIER"
    )

    print("=" * 60)


    model.compile(

        optimizer=Adam(

            learning_rate=LEARNING_RATE
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy"
        ]
    )


    history1 = model.fit(

        train_gen,

        validation_data=val_gen,

        epochs=EPOCHS_FROZEN,

        callbacks=get_callbacks(
            "phase1"
        ),

        class_weight=class_weight_dict
    )


    plot_history(

        history1,

        "phase1"
    )


    # ========================================================
    # PHASE 2 - FINE TUNING
    # ========================================================

    print("\n")
    print("=" * 60)

    print(
        "PHASE 2 - FINE TUNING MOBILENETV2"
    )

    print("=" * 60)


    base_model.trainable = True


    # --------------------------------------------------------
    # Freeze first layers
    # --------------------------------------------------------

    for layer in base_model.layers[
        :FINE_TUNE_AT
    ]:

        layer.trainable = False


    # --------------------------------------------------------
    # Freeze BatchNormalization
    # --------------------------------------------------------

    for layer in base_model.layers:

        if isinstance(

            layer,

            BatchNormalization
        ):

            layer.trainable = False


    # --------------------------------------------------------
    # Recompile model
    # --------------------------------------------------------

    model.compile(

        optimizer=Adam(

            learning_rate=1e-5
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy"
        ]
    )


    history2 = model.fit(

        train_gen,

        validation_data=val_gen,

        epochs=EPOCHS_FINE,

        callbacks=get_callbacks(
            "phase2"
        ),

        class_weight=class_weight_dict
    )


    plot_history(

        history2,

        "phase2"
    )


    # ========================================================
    # FINAL TEST
    # ========================================================

    print("\n")
    print("=" * 60)

    print(
        "FINAL TEST EVALUATION"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    if os.path.exists(
        MODEL_PATH
    ):

        print(
            "\nLoading best saved model..."
        )


        model = tf.keras.models.load_model(
            MODEL_PATH
        )


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    test_loss, test_accuracy = model.evaluate(

        test_gen,

        verbose=1
    )


    print("\n")
    print("=" * 60)

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )

    print("=" * 60)


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model.save(
        MODEL_PATH
    )


    print(
        f"\n✅ Model saved:"
    )

    print(
        MODEL_PATH
    )


    # ========================================================
    # SAVE CLASS INDICES
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    with open(

        CLASS_INDEX_PATH,

        "w"
    ) as f:

        json.dump(

            train_gen.class_indices,

            f,

            indent=4
        )


    print(
        "\n✅ Class mapping saved:"
    )

    print(
        CLASS_INDEX_PATH
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 60)

    print(
        "TRAINING COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)

    print(
        f"Final Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        "\nClass Mapping:"
    )

    print(
        train_gen.class_indices
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()