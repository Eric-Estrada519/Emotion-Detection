import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import LEGEND_CSV, IMAGES_DIR


class EmotionDataset(Dataset):
    def __init__(self, transform=None):
        self.data = pd.read_csv(LEGEND_CSV)
        self.img_dir = IMAGES_DIR
        self.transform = transform

        # normalize labels
        self.data["emotion"] = self.data["emotion"].str.lower()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image"])

        img = Image.open(img_path).convert("RGB")
        label = row["emotion"]

        if self.transform:
            img = self.transform(img)

        return img, label


def get_loader(batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = EmotionDataset(transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)