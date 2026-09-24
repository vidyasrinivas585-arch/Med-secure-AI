import csv
import os
import zipfile
import shutil

BASE = os.path.expanduser("~/Med-secure-AI/medsecure_ai")
CSV_FILE = os.path.join(BASE, "rximage_data", "table.csv")
OUTPUT_DIR = os.path.join(BASE, "rximage_data", "images")
ZIP_FILE = os.path.expanduser("~/Downloads/rximage.zip")

os.makedirs(OUTPUT_DIR, exist_ok=True)

LIMIT = 100
count = 0

with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    with zipfile.ZipFile(ZIP_FILE, "r") as z:

        for row in reader:

            if count >= LIMIT:
                break

            filename = row["rxnavImageFileName"].strip()

            if not filename:
                continue

            zip_path = "image/images/gallery/original/" + filename

            try:
                data = z.read(zip_path)

                output_path = os.path.join(
                    OUTPUT_DIR,
                    filename
                )

                with open(output_path, "wb") as out:
                    out.write(data)

                count += 1

                print(f"{count}/100 extracted: {filename}")

            except KeyError:
                print(f"Image not found: {filename}")

print()
print(f"Completed. {count} images extracted.")