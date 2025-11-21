
import torch
import torch.nn as nn
from torchvision import models, transforms
from model_code.data_loader import get_train_loader, get_val_loader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np

# -------------------------------
# 1. Load ResNet18 backbone
# -------------------------------
print("Loading ResNet18 for feature extraction...")
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()
resnet.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

print("Loading train and validation sets...")
train_loader = get_train_loader(batch_size=1)
val_loader = get_val_loader(batch_size=1)

def extract_features(dataloader):
    features = []
    labels = []
    for img, label in dataloader:
        img = img.to(device)
        label = label[0].lower()
        with torch.no_grad():
            feat = resnet(img)[0].cpu().numpy()
        features.append(feat)
        labels.append(label)
    return np.array(features), np.array(labels)

print("Extracting train features...")
X_train, y_train = extract_features(train_loader)

print("Extracting val features...")
X_val, y_val = extract_features(val_loader)

# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)

pipeline = Pipeline([
    ("pca", PCA()),
    ("svm", LinearSVC(max_iter=10000))
])

param_grid = {
    "pca__n_components": [32, 64, 128, 256],
    "svm__C": [0.01, 0.1, 1, 10]
}


print("Running parameter search...")

best_acc = -1
best_params = None

for n_comp in param_grid["pca__n_components"]:
    for C in param_grid["svm__C"]:
        # Train pipeline on train set
        pipeline.set_params(pca__n_components=n_comp, svm__C=C)
        pipeline.fit(X_train, y_train_enc)

        # Eval on validation set
        acc = pipeline.score(X_val, y_val_enc)
        print(f"PCA={n_comp}, C={C} → Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_params = (n_comp, C)

print("\nBest parameters found:")
print(f"PCA components: {best_params[0]}")
print(f"SVM C: {best_params[1]}")
print(f"Best Validation Accuracy: {best_acc:.4f}")
"""

Loading train and validation sets...
Extracting train features...
Extracting val features...
Running parameter search...
PCA=32, C=0.01 → Val Acc: 0.6805
PCA=32, C=0.1 → Val Acc: 0.6800
PCA=32, C=1 → Val Acc: 0.6800
PCA=32, C=10 → Val Acc: 0.6800
PCA=64, C=0.01 → Val Acc: 0.7151
PCA=64, C=0.1 → Val Acc: 0.7161
PCA=64, C=1 → Val Acc: 0.7161
PCA=64, C=10 → Val Acc: 0.7146
PCA=128, C=0.01 → Val Acc: 0.7435
PCA=128, C=0.1 → Val Acc: 0.7435
PCA=128, C=1 → Val Acc: 0.7440
PCA=128, C=10 → Val Acc: 0.7435
PCA=256, C=0.01 → Val Acc: 0.7557
PCA=256, C=0.1 → Val Acc: 0.7567
PCA=256, C=1 → Val Acc: 0.7562
PCA=256, C=10 → Val Acc: 0.7557

Best parameters found:
PCA components: 256
SVM C: 0.1
Best Validation Accuracy: 0.7567
"""