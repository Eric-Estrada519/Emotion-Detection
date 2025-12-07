import os
import matplotlib.pyplot as plt
from model_code.data_loader import get_all_data
from config import BASE_DIR


# Folder to save mean images
OUTPUT_DIR = os.path.join(BASE_DIR, "mean_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
loader = get_all_data(batch_size=1, shuffle=True)
dataset = loader.dataset

classes = sorted(set(dataset.data["emotion"].str.lower()))
print("Classes:", classes)

sums = {cls: None for cls in classes}
counts = {cls: 0 for cls in classes}

print("Computing mean images...")

for img, label in loader:
    cls = label[0].lower()
    img = img[0].float()

    if sums[cls] is None:
        sums[cls] = img.clone()
    else:
        sums[cls] += img

    counts[cls] += 1

mean_images = {cls: (sums[cls] / counts[cls]) for cls in classes}


num_classes = len(classes)
plt.figure(figsize=(4 * num_classes, 4))

for i, cls in enumerate(classes):
    mean_img = mean_images[cls]

    plt.subplot(1, num_classes, i + 1)
    plt.imshow(mean_img.permute(1, 2, 0).numpy())
    plt.title(cls)
    plt.axis("off")

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, "mean_faces_grid.png")
plt.savefig(save_path, bbox_inches='tight', dpi=200)
plt.show()

print(f"\nSaved combined mean grid to: {save_path}")
