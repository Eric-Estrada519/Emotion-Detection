import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from model_code.data_loader import (
    get_train_loader,
    get_val_loader,
    get_test_loader,
)

BEST_PCA_COMPONENTS = 256
BEST_C = 0.1

def get_feature_extractor():
    print("Loading ResNet18 backbone...")
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Identity()
    resnet.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = resnet.to(device)
    print(f"Using device: {device}")
    return resnet, device

def extract_features(dataloader, resnet, device):
    features = []
    labels = []

    for imgs, label_batch in dataloader:
        imgs = imgs.to(device).float()

        with torch.no_grad():
            feats = resnet(imgs)
        feats = feats.cpu().numpy()

        if isinstance(label_batch, (list, tuple)):
            labels.extend([str(l).lower() for l in label_batch])
        else:
            labels.append(str(label_batch).lower())

        features.append(feats)

    features = np.vstack(features)
    labels = np.array(labels)
    return features, labels

def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(this_dir, "../results_svm_final")
    os.makedirs(results_dir, exist_ok=True)

    print("Loading train/val/test loaders...")
    train_loader = get_train_loader(batch_size=32)
    val_loader = get_val_loader(batch_size=32)
    test_loader = get_test_loader(batch_size=32)

    resnet, device = get_feature_extractor()

    print("\nExtracting train features...")
    X_train, y_train_str = extract_features(train_loader, resnet, device)

    print("Extracting val features...")
    X_val, y_val_str = extract_features(val_loader, resnet, device)

    print("Extracting test features...")
    X_test, y_test_str = extract_features(test_loader, resnet, device)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_str)
    y_val = label_encoder.transform(y_val_str)
    y_test = label_encoder.transform(y_test_str)

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.hstack([y_train, y_val])

    print(f"\nFitting PCA with n_components={BEST_PCA_COMPONENTS}...")
    pca = PCA(n_components=BEST_PCA_COMPONENTS)
    X_trainval_pca = pca.fit_transform(X_trainval)

    print(f"Training final SVM with C={BEST_C}...")
    svm = LinearSVC(C=BEST_C, max_iter=10000)
    svm.fit(X_trainval_pca, y_trainval)

    print("\nEvaluating on test set...")
    X_test_pca = pca.transform(X_test)
    y_pred = svm.predict(X_test_pca)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")

    class_report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, digits=4
    )

    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(f"Final Test Accuracy: {accuracy:.4f}\n\n")
        f.write(class_report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Final SVM")
    plt.tight_layout()

    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.show()

    print(f"\nSaved confusion matrix and report to: {results_dir}")


if __name__ == "__main__":
    main()
