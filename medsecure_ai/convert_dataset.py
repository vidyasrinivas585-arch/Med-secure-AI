import os
import shutil

SOURCE = os.path.expanduser("~/Downloads/counterfeit_drugs")
DEST = os.path.expanduser("~/Med-secure-AI/medsecure_ai/dataset")

SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test"
}


def get_class_from_label(label_file):
    """
    YOLO classes:
    0 = authentic
    1 = counterfeit

    If an image contains any counterfeit label (1),
    classify the image as counterfeit.
    """

    classes = set()

    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            try:
                class_id = int(parts[0])
                classes.add(class_id)
            except ValueError:
                continue

    if 1 in classes:
        return "counterfeit"

    if 0 in classes:
        return "genuine"

    return None


def process_split(source_split, destination_split):

    image_dir = os.path.join(
        SOURCE,
        source_split,
        "images"
    )

    label_dir = os.path.join(
        SOURCE,
        source_split,
        "labels"
    )

    genuine_dir = os.path.join(
        DEST,
        destination_split,
        "genuine"
    )

    counterfeit_dir = os.path.join(
        DEST,
        destination_split,
        "counterfeit"
    )

    os.makedirs(genuine_dir, exist_ok=True)
    os.makedirs(counterfeit_dir, exist_ok=True)

    genuine_count = 0
    counterfeit_count = 0
    skipped_count = 0

    if not os.path.exists(image_dir):
        print(f"Image folder not found: {image_dir}")
        return

    for filename in os.listdir(image_dir):

        image_path = os.path.join(
            image_dir,
            filename
        )

        if not os.path.isfile(image_path):
            continue

        base_name = os.path.splitext(filename)[0]

        label_path = os.path.join(
            label_dir,
            base_name + ".txt"
        )

        if not os.path.exists(label_path):
            skipped_count += 1
            continue

        image_class = get_class_from_label(label_path)

        if image_class is None:
            skipped_count += 1
            continue

        if image_class == "genuine":

            destination = os.path.join(
                genuine_dir,
                filename
            )

            genuine_count += 1

        else:

            destination = os.path.join(
                counterfeit_dir,
                filename
            )

            counterfeit_count += 1

        shutil.copy2(
            image_path,
            destination
        )

    print()
    print(f"{source_split} -> {destination_split}")
    print(f"Genuine: {genuine_count}")
    print(f"Counterfeit: {counterfeit_count}")
    print(f"Skipped: {skipped_count}")


print("=" * 50)
print("Converting Counterfeit Drugs 2.0 dataset")
print("=" * 50)

for source_split, destination_split in SPLITS.items():

    process_split(
        source_split,
        destination_split
    )

print()
print("Dataset conversion completed.")