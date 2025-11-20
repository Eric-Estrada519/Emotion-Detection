import os
from collections import Counter
from PIL import Image
from config import IMAGES_DIR
import pandas as pd


resolutions = []

# Loop through all image files in the images directory
for fname in os.listdir(IMAGES_DIR):
    if fname.lower().endswith((".jpg")):
        path = os.path.join(IMAGES_DIR, fname)
        try:
            with Image.open(path) as img:
                resolutions.append(img.size)  # (width, height)
        except Exception as e:
            print("Error reading image:", fname, e)

counts = Counter(resolutions)

df = pd.DataFrame(
    [(w, h, c) for (w, h), c in counts.items()],
    columns=["Width", "Height", "Count"]
).sort_values(by="Count", ascending=False)

print("\n=== IMAGE RESOLUTION STATS ===\n")
print(df.to_string(index=False))

print("\nTotal unique resolutions:", len(df))
print("Total images scanned:", len(resolutions))

"""Finds all the photos resolutions and actually shows you the unique resolutions in al the images and shows what needs to be pruned
Showed that most of the dataset is uniform
350 × 350 : 12,794 images  (≈93%)

Prune the ones that are small ,but resize large size images

Total unique resolutions: 855

Total images scanned: 13718


#"""