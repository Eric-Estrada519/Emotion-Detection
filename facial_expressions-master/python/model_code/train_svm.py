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

def get_feature_extractor():
    print("Loading ResNet18...")
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Identity()
    resnet.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet.to(device)
    print(f"Using device: {device}")
    return resnet, device

def extract_features(loader, model, device):
    feats, labels = [], []

    for imgs, lbls in loader:
        imgs = imgs.to(device).float()

        with torch.no_grad():
            f = model(imgs).cpu().numpy()

        feats.append(f)

        # handle batch labels
        if isinstance(lbls, (list, tuple)):
            labels.extend([x.lower() for x in lbls])
        else:
            labels.append(lbls.lower())

    return np.vstack(feats), np.array(labels)

def main():
    # Setup result directory
    results_dir = os.path.join(os.path.dirname(__file__), "../results_svm")
    os.makedirs(results_dir, exist_ok=True)

    # Load loaders
    train_loader = get_train_loader(batch_size=32)
    val_loader = get_val_loader(batch_size=32)
    test_loader = get_test_loader(batch_size=32)

    # Extract ResNet features
    resnet, device = get_feature_extractor()

    print("Extracting train features...")
    X_train, y_train_str = extract_features(train_loader, resnet, device)

    print("Extracting validation features...")
    X_val, y_val_str = extract_features(val_loader, resnet, device)

    print("Extracting test features...")
    X_test, y_test_str = extract_features(test_loader, resnet, device)

    # Encode labels
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_str)
    y_val   = encoder.transform(y_val_str)
    y_test  = encoder.transform(y_test_str)

    pca_sizes = [32, 64, 128, 256]
    C_values = [0.01, 0.1, 1.0, 10]

    best_acc = -1
    best_params = None
    best_model = None
    best_pca = None

    # Store (pca_size, C, acc) for the plot
    val_accuracy_record = []

    print("\n=== SVM Model Selection ===\n")

    for n_comp in pca_sizes:
        print(f"\nTesting PCA = {n_comp}")

        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca   = pca.transform(X_val)

        for C in C_values:
            svm = LinearSVC(C=C, max_iter=20000)
            svm.fit(X_train_pca, y_train)

            preds = svm.predict(X_val_pca)
            acc = accuracy_score(y_val, preds)

            print(f"PCA={n_comp}, C={C} → Val Acc: {acc:.4f}")
            val_accuracy_record.append((n_comp, C, acc))

            if acc > best_acc:
                best_acc = acc
                best_params = (n_comp, C)
                best_model = svm
                best_pca = pca

    print("\n=== Best SVM Parameters ===")
    print(f"PCA components: {best_params[0]}")
    print(f"SVM C: {best_params[1]}")
    print(f"Validation Accuracy: {best_acc:.4f}")

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.hstack([y_train, y_val])

    final_pca = PCA(n_components=best_params[0])
    X_trainval_pca = final_pca.fit_transform(X_trainval)
    X_test_pca = final_pca.transform(X_test)

    final_svm = LinearSVC(C=best_params[1], max_iter=20000)
    final_svm.fit(X_trainval_pca, y_trainval)

    preds = final_svm.predict(X_test_pca)
    acc = accuracy_score(y_test, preds)
    print(f"\nFinal SVM Test Accuracy: {acc:.4f}")

    report = classification_report(
        y_test, preds, target_names=encoder.classes_, digits=4
    )
    print(report)

    with open(os.path.join(results_dir, "svm_classification_report.txt"), "w") as f:
        f.write(report)


    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=encoder.classes_, yticklabels=encoder.classes_
    )
    plt.title("SVM Confusion Matrix (Final Model)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "svm_confusion_matrix.png"))
    plt.close()

    explained = final_pca.explained_variance_ratio_
    cumulative = explained.cumsum()

    plt.figure(figsize=(8,6))
    plt.plot(cumulative, marker="o")
    plt.axvline(best_params[0], color="red", linestyle="--")
    plt.title("PCA Variance Explained (SVM)")
    plt.xlabel("Components")
    plt.ylabel("Cumulative Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pca_variance_curve.png"))
    plt.close()

    plt.figure(figsize=(8,6))

    for C in C_values:
        # accuracies where C = that value
        accs = [acc for (pca_n, c_val, acc) in val_accuracy_record if c_val == C]
        plt.plot(pca_sizes, accs, marker="o", label=f"C={C}")

    plt.xlabel("PCA Components")
    plt.ylabel("Validation Accuracy")
    plt.title("SVM Model Selection — PCA Components vs Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "svm_model_selection.png"))
    plt.close()

    print("\nSaved:")
    print(" - svm_confusion_matrix.png")
    print(" - pca_variance_curve.png")
    print(" - svm_model_selection.png")
    print(" - svm_classification_report.txt")


if __name__ == "__main__":
    main()

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