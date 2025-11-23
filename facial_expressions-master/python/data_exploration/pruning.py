import os
import shutil
from PIL import Image
from config import IMAGES_DIR

# Folder to store bad images (so you can review them)
BAD_DIR = os.path.join(os.path.dirname(IMAGES_DIR), "bad_images")
os.makedirs(BAD_DIR, exist_ok=True)

MIN_SIZE = 100  \


bad_images = []

for fname in os.listdir(IMAGES_DIR):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(IMAGES_DIR, fname)

        try:
            with Image.open(path) as img:
                w, h = img.size

                # Flag tiny or extremely weird images
                if w < MIN_SIZE or h < MIN_SIZE:
                    bad_images.append(fname)

        except Exception as e:
            print("Error reading:", fname, e)
            bad_images.append(fname)

# Move bad images out of training folder
for fname in bad_images:
    src = os.path.join(IMAGES_DIR, fname)
    dst = os.path.join(BAD_DIR, fname)

    if os.path.exists(src):
        shutil.move(src, dst)

print("Done!\n")
print(f"Total bad images moved: {len(bad_images)}")
print(f"Bad images are in: {BAD_DIR}")


"""This prunes out the images that have a low resolution that can trick our model"""