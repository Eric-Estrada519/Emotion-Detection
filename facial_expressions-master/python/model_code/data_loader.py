import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from config import IMAGES_DIR

class EmotionDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        Dataset for emotion classification.

        csv_path: path to train.csv, val.csv, or test.csv
        transform: torchvision transforms to apply to each image
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        # normalize emotion labels
        self.data["emotion"] = self.data["emotion"].str.lower()

        # image directory (clean images only)
        self.img_dir = IMAGES_DIR

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_name = row["image"]
        label = row["emotion"]

        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image file: {img_path}")

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label




def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def get_train_loader(batch_size=32, shuffle=True):
    from config import TRAIN_CSV
    return DataLoader(
        EmotionDataset(TRAIN_CSV, transform=get_transforms()),
        batch_size=batch_size,
        shuffle=shuffle
    )

def get_val_loader(batch_size=32, shuffle=False):
    from config import VAL_CSV
    return DataLoader(
        EmotionDataset(VAL_CSV, transform=get_transforms()),
        batch_size=batch_size,
        shuffle=shuffle
    )

def get_test_loader(batch_size=32, shuffle=False):
    from config import TEST_CSV
    return DataLoader(
        EmotionDataset(TEST_CSV, transform=get_transforms()),
        batch_size=batch_size,
        shuffle=shuffle
    )
