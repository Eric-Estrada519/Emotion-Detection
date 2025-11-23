import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

from model_code.data_loader import (
    get_train_loader,
    get_val_loader,
    get_test_loader,
)

# --------------------------------------------------------------
# Correct class names (index-aligned)
# --------------------------------------------------------------
CLASS_NAMES = [
    "anger",      # 0
    "contempt",   # 1
    "disgust",    # 2
    "fear",       # 3
    "happiness",  # 4
    "neutral",    # 5
    "sadness",    # 6
    "surprise"    # 7
]

# --------------------------------------------------------------
# Mapping string → integer label
# --------------------------------------------------------------
LABEL_TO_IDX = {
    "anger": 0,
    "contempt": 1,
    "disgust": 2,
    "fear": 3,
    "happiness": 4,
    "neutral": 5,
    "sadness": 6,
    "surprise": 7,
}


# --------------------------------------------------------------
# Convert DataLoader to NumPy arrays
# --------------------------------------------------------------
def get_numpy_data(loader):
    X_list, y_list = [], []

    for imgs, labels in loader:
        # imgs shape: (B, 3, 224, 224) → flatten each image
        X_list.append(imgs.numpy().reshape(len(imgs), -1))

        numeric_labels = [LABEL_TO_IDX[str(l).lower()] for l in labels]
        y_list.append(np.array(numeric_labels))

    return np.vstack(X_list), np.hstack(y_list)


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
def main():
    results_dir = os.path.join(os.path.dirname(__file__), "../results_xgboost")
    os.makedirs(results_dir, exist_ok=True)

    print("Loading data...")
    train_loader = get_train_loader(batch_size=64)
    val_loader = get_val_loader(batch_size=64)
    test_loader = get_test_loader(batch_size=64)

    X_train_raw, y_train = get_numpy_data(train_loader)
    X_val_raw, y_val = get_numpy_data(val_loader)
    X_test_raw, y_test = get_numpy_data(test_loader)

    # Combine train + val for CV
    X_full = np.vstack([X_train_raw, X_val_raw])
    y_full = np.hstack([y_train, y_val])

    # ----------------------------------------------------------
    # PCA REDUCTION
    # ----------------------------------------------------------
    print("Running PCA...")
    pca = PCA(n_components=256)
    X_full_pca = pca.fit_transform(X_full)
    X_test_pca = pca.transform(X_test_raw)

    # ----------------------------------------------------------
    # HYPERPARAMETERS
    # ----------------------------------------------------------
    param_grid = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.03},
    ]

    best_acc = 0.0
    best_params = None

    # For plots
    param_labels = []
    mean_train_accs = []
    mean_val_accs = []

    print("\n=== 5-FOLD CROSS VALIDATION ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for params in param_grid:
        print(f"\nTesting params: {params}")

        fold_train_acc = []
        fold_val_acc = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_full_pca, y_full)):
            print(f"  Fold {fold+1}/5")

            X_tr, X_va = X_full_pca[train_idx], X_full_pca[val_idx]
            y_tr, y_va = y_full[train_idx], y_full[val_idx]

            model = XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softmax",
                num_class=len(CLASS_NAMES)
            )

            model.fit(X_tr, y_tr)

            # Training accuracy
            preds_tr = model.predict(X_tr)
            tr_acc = accuracy_score(y_tr, preds_tr)
            fold_train_acc.append(tr_acc)

            # Validation accuracy
            preds_va = model.predict(X_va)
            va_acc = accuracy_score(y_va, preds_va)
            fold_val_acc.append(va_acc)

            print(f"    Train Acc: {tr_acc:.4f} | Val Acc: {va_acc:.4f}")

        mean_train = np.mean(fold_train_acc)
        mean_val = np.mean(fold_val_acc)

        print(f" Mean Train Acc: {mean_train:.4f}")
        print(f" Mean Val Acc:   {mean_val:.4f}")

        # save for plotting
        label = f"{params['n_estimators']} trees, depth={params['max_depth']}, lr={params['learning_rate']}"
        param_labels.append(label)
        mean_train_accs.append(mean_train)
        mean_val_accs.append(mean_val)

        # Track best hyperparameters
        if mean_val > best_acc:
            best_acc = mean_val
            best_params = params

    print("\n=== BEST XGBOOST PARAMETERS ===")
    print(best_params)
    print(f"Best CV Accuracy: {best_acc:.4f}")

    # ----------------------------------------------------------
    # PLOT TRAIN/VAL ERROR ACROSS PARAMETER SETS
    # ----------------------------------------------------------
    train_errors = [1 - acc for acc in mean_train_accs]
    val_errors = [1 - acc for acc in mean_val_accs]

    plt.figure(figsize=(10, 6))
    plt.plot(param_labels, train_errors, marker="o", label="Train Error")
    plt.plot(param_labels, val_errors, marker="o", label="Validation Error")
    plt.xticks(rotation=45)
    plt.ylabel("Error Rate (1 - Accuracy)")
    plt.title("Train vs Validation Error Across XGBoost Hyperparameters")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "xgboost_train_val_error.png"))
    plt.close()

    # ----------------------------------------------------------
    # FINAL TRAINING ON FULL TRAIN+VAL
    # ----------------------------------------------------------
    print("\nRetraining final model...")
    best_model = XGBClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(CLASS_NAMES)
    )
    best_model.fit(X_full_pca, y_full)

    # ----------------------------------------------------------
    # TEST EVALUATION
    # ----------------------------------------------------------
    preds_test = best_model.predict(X_test_pca)
    test_acc = accuracy_score(y_test, preds_test)

    print(f"\n=== FINAL TEST ACCURACY: {test_acc:.4f} ===")

    # ----------------------------------------------------------
    # CONFUSION MATRIX
    # ----------------------------------------------------------
    cm = confusion_matrix(y_test, preds_test)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title("XGBoost Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "xgboost_confusion_matrix.png"))
    plt.close()

    # ----------------------------------------------------------
    # PER-CLASS ERROR BAR CHART
    # ----------------------------------------------------------
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    per_class_error = 1 - per_class_accuracy

    plt.figure(figsize=(10, 6))
    plt.bar(CLASS_NAMES, per_class_error, color="crimson")
    plt.title("Per-Class Test Error (XGBoost)")
    plt.ylabel("Error Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "xgboost_per_class_error.png"))
    plt.close()

    # ----------------------------------------------------------
    # CLASSIFICATION REPORT
    # ----------------------------------------------------------
    report = classification_report(
        y_test,
        preds_test,
        target_names=CLASS_NAMES,
        digits=4
    )
    with open(os.path.join(results_dir, "xgboost_classification_report.txt"), "w") as f:
        f.write(report)

    print("\nSaved ALL XGBoost results to:", results_dir)


if __name__ == "__main__":
    main()
