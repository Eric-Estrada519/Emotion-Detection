import os
import matplotlib.pyplot as plt
from model_code.data_loader import get_loader
from config import BASE_DIR

# Folder to save mean images
OUTPUT_DIR = os.path.join(BASE_DIR, "mean_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
loader = get_loader(batch_size=1, shuffle=True)
dataset = loader.dataset

# Classes
classes = sorted(set(dataset.data["emotion"].str.lower()))
print("Classes:", classes)

# Storage for sums and counts
sums = {cls: 0 for cls in classes}
counts = {cls: 0 for cls in classes}

print("Computing pixel sums...")

# Accumulate pixel values per class
for img, label in loader:
    cls = label[0].lower()  # single label in batch
    img = img[0].float()  # (3, H, W)

    if isinstance(sums[cls], int):
        sums[cls] = img.clone()
    else:
        sums[cls] += img

    counts[cls] += 1

print("Creating mean images...")

mean_images = {}

for cls in classes:
    if counts[cls] > 0:
        mean_img = sums[cls] / counts[cls]  # average pixel values
        mean_images[cls] = mean_img

        # Save mean face
        plt.imshow(mean_img.permute(1, 2, 0).numpy())
        plt.title(f"Mean {cls}")
        plt.axis("off")
        save_path = os.path.join(OUTPUT_DIR, f"mean_{cls}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

print(f"\nMean images saved to: {OUTPUT_DIR}")

# Plot all mean images in a grid
plt.figure(figsize=(4 * len(classes), 4))

for i, cls in enumerate(classes):
    img = mean_images[cls]
    plt.subplot(1, len(classes), i + 1)
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.title(cls)
    plt.axis("off")

plt.tight_layout()
plt.show()
