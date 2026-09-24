import os
import re
import time
import pandas as pd
import requests
from tqdm import tqdm

PARQUET_FILE = "medicine_data.parquet"
OUTPUT_DIR = "images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Reading medicine dataset...")

df = pd.read_parquet(PARQUET_FILE)

print("Total records:", len(df))
print("Columns:", list(df.columns))

def clean_name(name):
    name = str(name)
    name = re.sub(r'[\/:*?"<>|]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name[:100]

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

success = 0
failed = 0

for index, row in tqdm(df.iterrows(), total=len(df)):

    name = clean_name(row["name"])
    image_url = row["image_url"]

    if pd.isna(image_url) or not str(image_url).strip():
        failed += 1
        continue

    folder = os.path.join(OUTPUT_DIR, name)
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, f"{index}.jpg")

    if os.path.exists(file_path):
        success += 1
        continue

    try:
        response = session.get(
            str(image_url),
            timeout=20
        )

        if response.status_code == 200 and len(response.content) > 1000:

            with open(file_path, "wb") as f:
                f.write(response.content)

            success += 1

        else:
            failed += 1

    except Exception:
        failed += 1

    time.sleep(0.05)

print()
print("================================")
print("DOWNLOAD COMPLETED")
print("================================")
print("Successful:", success)
print("Failed:", failed)
print("Total:", len(df))
print("Images folder:", OUTPUT_DIR)
