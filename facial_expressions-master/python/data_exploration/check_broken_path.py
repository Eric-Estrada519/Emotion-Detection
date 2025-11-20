import os
import pandas as pd
from PIL import Image
from config import LEGEND_CSV, IMAGES_DIR

print("Checking image paths...")

df = pd.read_csv(LEGEND_CSV)

missing_files = []
corrupted_files = []
wrong_extension = []

ALLOWED_EXT = (".jpg", ".jpeg", ".png")

for idx, row in df.iterrows():
    fname = row["image"]
    fpath = os.path.join(IMAGES_DIR, fname)

    # --- Check extension ---
    if not fname.lower().endswith(ALLOWED_EXT):
        wrong_extension.append(fname)

    # --- Check existence ---
    if not os.path.exists(fpath):
        missing_files.append(fname)
        continue

    # --- Check readability (corrupt detection) ---
    try:
        with Image.open(fpath) as img:
            img.verify()   # PIL corruption check
    except Exception:
        corrupted_files.append(fname)


# ----------- RESULTS -----------
print("\n=== DATA INTEGRITY REPORT ===\n")

print("Missing image files:", len(missing_files))
print("Corrupted image files:", len(corrupted_files))
print("Wrong extension files:", len(wrong_extension))

# Save reports
report_dir = os.path.join(os.path.dirname(IMAGES_DIR), "integrity_reports")
os.makedirs(report_dir, exist_ok=True)

pd.DataFrame({"missing_file": missing_files}).to_csv(
    os.path.join(report_dir, "missing_files.csv"), index=False
)

pd.DataFrame({"corrupted_file": corrupted_files}).to_csv(
    os.path.join(report_dir, "corrupted_files.csv"), index=False
)

pd.DataFrame({"wrong_extension": wrong_extension}).to_csv(
    os.path.join(report_dir, "wrong_extension.csv"), index=False
)

print("\nReports saved to:", report_dir)
