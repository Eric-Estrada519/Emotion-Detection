import os
import pandas as pd
from config import LEGEND_CSV, IMAGES_DIR

df = pd.read_csv(LEGEND_CSV)

print("Original CSV rows:", len(df))

# Normalize labels
df["emotion"] = df["emotion"].str.lower()

# keep rows that image file still exists
df["exists"] = df["image"].apply(lambda f: os.path.exists(os.path.join(IMAGES_DIR, f)))
clean_df = df[df["exists"] == True].drop(columns=["exists"])

print("Rows after cleaning:", len(clean_df))

# Backup old CSV
backup_path = LEGEND_CSV.replace(".csv", "_backup_before_prune.csv")
df.to_csv(backup_path, index=False)

# Save cleaned CSV
clean_df.to_csv(LEGEND_CSV, index=False)

print("\nCleaned CSV saved.")
print("Backup saved at:", backup_path)
