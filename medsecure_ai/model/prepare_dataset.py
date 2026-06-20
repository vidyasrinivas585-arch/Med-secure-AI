"""
model/prepare_dataset.py
Utility script to verify, balance, and augment the training dataset.
Run this before train.py to ensure your dataset is ready.

Usage:
    python model/prepare_dataset.py
"""

import os
import shutil
import random
import cv2
import numpy as np
import json

DATASET_DIR = "dataset"
CLASSES     = ["genuine", "counterfeit"]
MIN_IMAGES  = 50   # Minimum images per class recommended


def check_dataset():
    """Verify dataset structure and report counts."""
    print("=" * 50)
    print("Dataset Verification")
    print("=" * 50)
    all_ok = True
    for cls in CLASSES:
        path = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(path):
            print(f"❌ Missing folder: {path}")
            all_ok = False
            continue
        images = [f for f in os.listdir(path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
        status = "✅" if len(images) >= MIN_IMAGES else "⚠️"
        print(f"{status} {cls}: {len(images)} images")
        if len(images) < MIN_IMAGES:
            print(f"   → Recommend at least {MIN_IMAGES} images for reliable training.")
            all_ok = False
    return all_ok


def augment_image(image: np.ndarray) -> list[np.ndarray]:
    """
    Generate augmented variants of a single image.
    Returns 5 new images using different augmentations.
    """
    results = []
    h, w = image.shape[:2]

    # 1. Horizontal flip
    results.append(cv2.flip(image, 1))

    # 2. Rotation ±15°
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1.0)
    results.append(cv2.warpAffine(image, M, (w, h)))

    # 3. Brightness adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.3, beta=20)
    results.append(bright)

    # 4. Gaussian noise
    noise = np.random.normal(0, 15, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    results.append(noisy)

    # 5. Zoom crop (80% center)
    crop_h, crop_w = int(h * 0.8), int(w * 0.8)
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    cropped = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    results.append(cv2.resize(cropped, (w, h)))

    return results


def augment_class(cls: str, target_count: int = 200):
    """
    Augment a class folder until it has at least `target_count` images.
    Augmented images are saved alongside originals.
    """
    class_dir = os.path.join(DATASET_DIR, cls)
    existing  = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(existing) >= target_count:
        print(f"  {cls}: already has {len(existing)} images. No augmentation needed.")
        return

    print(f"  {cls}: augmenting from {len(existing)} → {target_count}+...")
    aug_count = 0
    while len(os.listdir(class_dir)) < target_count:
        src_file = random.choice(existing)
        img = cv2.imread(os.path.join(class_dir, src_file))
        if img is None:
            continue
        for i, aug_img in enumerate(augment_image(img)):
            save_name = f"aug_{aug_count:05d}_{i}.jpg"
            cv2.imwrite(os.path.join(class_dir, save_name), aug_img)
            aug_count += 1
            if len(os.listdir(class_dir)) >= target_count:
                break

    print(f"  {cls}: done. Total: {len(os.listdir(class_dir))} images.")


def create_sample_dataset():
    """
    Create a minimal placeholder dataset with synthetic colored images
    so the project can be tested immediately without real medicine photos.
    FOR DEVELOPMENT ONLY — replace with real data before actual use.
    """
    print("\n⚠️  Creating SYNTHETIC placeholder dataset for development...")
    for cls in CLASSES:
        path = os.path.join(DATASET_DIR, cls)
        os.makedirs(path, exist_ok=True)
        existing = [f for f in os.listdir(path) if f.endswith('.jpg')]
        for i in range(len(existing), MIN_IMAGES):
            # Generate a distinguishable synthetic image per class
            if cls == "genuine":
                img = np.full((224, 224, 3), [180, 230, 180], dtype=np.uint8)  # greenish
            else:
                img = np.full((224, 224, 3), [230, 180, 180], dtype=np.uint8)  # reddish
            # Add some noise to make images distinct
            noise = np.random.randint(0, 40, (224, 224, 3), dtype=np.uint8)
            img = cv2.add(img, noise)
            cv2.putText(img, f"{cls[:3].upper()} {i}", (20, 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imwrite(os.path.join(path, f"synthetic_{i:04d}.jpg"), img)
    print("✅ Synthetic dataset created. Replace with real images before production use.\n")


if __name__ == "__main__":
    print("MedSecure AI — Dataset Preparation\n")

    ok = check_dataset()
    if not ok:
        choice = input("\nCreate synthetic placeholder dataset for testing? [y/N]: ").strip().lower()
        if choice == "y":
            create_sample_dataset()
            check_dataset()

    if check_dataset():
        print("\n📦 Augmenting classes to 200+ images each...")
        for cls in CLASSES:
            augment_class(cls, target_count=200)
        print("\n✅ Dataset ready for training. Run: python model/train.py")
