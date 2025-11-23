import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from model_code.data_loader import (
    get_train_loader,
    get_val_loader,
    get_test_loader,
)


# -------------------------------------------------
# Extract ResNet features for an entire dataloader
# -------------------------------------------------
def extract_features(loader, feature_extractor, device):
    feature_extractor.eval()

    feats = []
    labels = []

    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            features = feature_extractor(images).cpu().numpy()

            feats.append(features)
            labels.extend(lbls)  # keep raw emotion strings

    feats = np.vstack(feats)
    return feats, labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torchvision import models

    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device)

    train_loader = get_train_loader(batch_size=32)
    val_loader = get_val_loader(batch_size=32)
    test_loader = get_test_loader(batch_size=32)

    print("Extracting training features...")
    X_train, y_train = extract_features(train_loader, resnet, device)

    print("Extracting validation features...")
    X_val, y_val = extract_features(val_loader, resnet, device)

    print("Extracting test features...")
    X_test, y_test = extract_features(test_loader, resnet, device)

    # Convert string labels to integers consistently
    class_names = sorted(list(set(y_train + y_val + y_test)))
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    y_train_idx = np.array([name_to_idx[y] for y in y_train])
    y_val_idx = np.array([name_to_idx[y] for y in y_val])
    y_test_idx = np.array([name_to_idx[y] for y in y_test])

    pca_options = [32, 64, 128, 256]
    C_options = [0.01, 0.1, 1.0, 10]

    results = []

    for pca_dim in pca_options:
        print(f"\nRunning PCA = {pca_dim}")

        pca = PCA(n_components=pca_dim)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        for C in C_options:
            print(f"  Logistic Regression (C={C})")

            logreg = LogisticRegression(
                C=C,
                max_iter=2000,
                solver="lbfgs",
                multi_class="multinomial"
            )
            logreg.fit(X_train_pca, y_train_idx)

            val_pred = logreg.predict(X_val_pca)
            val_acc = accuracy_score(y_val_idx, val_pred)

            results.append((pca_dim, C, val_acc))

    best_pca, best_C, best_acc = max(results, key=lambda x: x[2])

    print("\n=== BEST LOGISTIC REGRESSION MODEL ===")
    print(f"PCA components: {best_pca}")
    print(f"C value:        {best_C}")
    print(f"Val Accuracy:   {best_acc:.4f}")

    print("\nTraining final model on TRAIN+VAL...")

    # Combine datasets
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train_idx, y_val_idx])

    pca_final = PCA(n_components=best_pca)
    X_all_pca = pca_final.fit_transform(X_all)
    X_test_pca = pca_final.transform(X_test)

    final_model = LogisticRegression(
        C=best_C,
        max_iter=2000,
        solver="lbfgs",
        multi_class="multinomial"
    )
    final_model.fit(X_all_pca, y_all)

    test_pred = final_model.predict(X_test_pca)
    test_acc = accuracy_score(y_test_idx, test_pred)

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    results_dir = os.path.join(os.path.dirname(__file__), "../results_logreg")
    os.makedirs(results_dir, exist_ok=True)

    cm = confusion_matrix(y_test_idx, test_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Logistic Regression Confusion Matrix (Final Model)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "logreg_confusion_matrix.png"))

    report = classification_report(y_test_idx, test_pred, target_names=class_names)
    with open(os.path.join(results_dir, "logreg_classification_report.txt"), "w") as f:
        f.write(report)

    pca_best_acc = {}
    for pca_dim, C, acc in results:
        if pca_dim not in pca_best_acc or acc > pca_best_acc[pca_dim]:
            pca_best_acc[pca_dim] = acc

    plt.figure(figsize=(8, 6))
    plt.plot(list(pca_best_acc.keys()), list(pca_best_acc.values()), marker="o")
    plt.title("Validation Accuracy vs PCA Components (LogReg)")
    plt.xlabel("PCA Components")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "logreg_pca_vs_accuracy.png"))

    C_acc = {C: max([acc for p, Cc, acc in results if Cc == C]) for C in C_options}

    plt.figure(figsize=(8, 6))
    plt.plot(list(C_acc.keys()), list(C_acc.values()), marker="o")
    plt.title("Validation Accuracy vs C (LogReg)")
    plt.xlabel("C value")
    plt.ylabel("Validation Accuracy")
    plt.xscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "logreg_C_vs_accuracy.png"))

    print("\nSaved all results to:", results_dir)


if __name__ == "__main__":
    main()


