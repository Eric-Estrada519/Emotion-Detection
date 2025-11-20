import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

# -----------------------------
# CUSTOM DATASET
# -----------------------------
class EmotionDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_name = row['image']
        label = row['emotion']

        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# -----------------------------
# CREATE DATALOADER
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = EmotionDataset(
    csv_path="../data/legend.csv",   # FIXED PATH
    img_dir="../images",             # FIXED PATH
    transform=transform
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

# -----------------------------
# DISPLAY FIRST 5 IMAGES
# -----------------------------
plt.figure(figsize=(12, 4))

for i, (img, label) in enumerate(loader):
    if i == 5:
        break

    img_np = img[0].permute(1, 2, 0).numpy()

    plt.subplot(1, 5, i + 1)
    plt.imshow(img_np)
    plt.title(label[0])
    plt.axis("off")

plt.tight_layout()
plt.show()


