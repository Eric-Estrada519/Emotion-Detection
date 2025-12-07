import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from config import IMAGES_DIR
from config import TEST_CSV, VAL_CSV, TRAIN_CSV, LEGEND_CSV

class EmotionDataset(Dataset):
    def __init__(self, csv_path, transform=None, cnn_mode=False):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.cnn_mode = cnn_mode

        # normalize
        self.data["emotion"] = self.data["emotion"].str.lower()

        # Only needed for CNN labels
        if self.cnn_mode:
            from sklearn.preprocessing import LabelEncoder
            self.encoder = LabelEncoder()
            self.data["label_id"] = self.encoder.fit_transform(self.data["emotion"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(IMAGES_DIR, row["image"])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.cnn_mode:
            # Return integer class ID for CNN
            label = int(row["label_id"])
        else:
            # Return raw emotion string for SVM, Logistic Regression
            label = row["emotion"]

        return img, label





def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def get_all_data(batch_size=32, shuffle=True):
    return DataLoader(
        EmotionDataset(LEGEND_CSV, transform=get_transforms()),
        batch_size=batch_size,
        shuffle=shuffle
    )

def get_train_loader(batch_size=32, shuffle=True):
    return DataLoader(
        EmotionDataset(TRAIN_CSV, transform=get_transforms()),
        batch_size=batch_size,
        shuffle=shuffle
    )

def get_val_loader(batch_size=32, shuffle=False):
    return DataLoader(
        EmotionDataset(VAL_CSV, transform=get_transforms()),
        batch_size=batch_size,
        shuffle=shuffle
    )

def get_test_loader(batch_size=32, shuffle=False):
    return DataLoader(
        EmotionDataset(TEST_CSV, transform=get_transforms()),
        batch_size=batch_size,
        shuffle=shuffle
    )

def get_train_loader_cnn(batch_size=32):
    return DataLoader(
        EmotionDataset(TRAIN_CSV, transform=get_transforms(), cnn_mode=True),
        batch_size=batch_size,
        shuffle=True
    )

def get_val_loader_cnn(batch_size=32):
    return DataLoader(
        EmotionDataset(VAL_CSV, transform=get_transforms(), cnn_mode=True),
        batch_size=batch_size,
        shuffle=False
    )

def get_test_loader_cnn(batch_size=32):
    return DataLoader(
        EmotionDataset(TEST_CSV, transform=get_transforms(), cnn_mode=True),
        batch_size=batch_size,
        shuffle=False
    )

