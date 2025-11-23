import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model_code.data_loader import (
    get_train_loader,
    get_val_loader,
    get_test_loader,
)

# ===========================
# CLASS NAMES
# ===========================
CLASS_NAMES = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise"
]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

def convert_labels(labels, device):
    if isinstance(labels, torch.Tensor):
        return labels.long().to(device)
    return torch.tensor([LABEL_TO_IDX[str(l).lower()] for l in labels],
                        dtype=torch.long, device=device)


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=8, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56

            # Block 3 (NEW)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = convert_labels(labels, device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = convert_labels(labels, device)
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader), correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    results_dir = os.path.join(os.path.dirname(__file__), "../results_cnn_v2")
    os.makedirs(results_dir, exist_ok=True)

    # Loaders
    train_loader = get_train_loader(batch_size=32)
    val_loader = get_val_loader(batch_size=32)
    test_loader = get_test_loader(batch_size=32)

    # Model
    model = ImprovedCNN(dropout=0.5).to(device)

    # Weight decay for regularization
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0005,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Class weights for imbalance
    weights = torch.tensor(
        [6778, 5453, 362, 244, 195, 71, 16, 9],
        dtype=torch.float32
    )
    weights = (weights.sum() / weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ============ TRAIN FOR 6 EPOCHS ============
    epochs = 6
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1}/{epochs} =====")

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(f"Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f}")

        torch.save(model.state_dict(), os.path.join(results_dir, "best_cnn_v2.pt"))

    model.load_state_dict(torch.load(os.path.join(results_dir, "best_cnn_v2.pt")))
    model.eval()

    preds_all, labels_all = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            label_ids = convert_labels(labels, device).cpu()
            preds = model(imgs).argmax(1).cpu()
            preds_all.append(preds)
            labels_all.append(label_ids)

    preds = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Improved CNN v2 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix_v2.png"))

    # Classification Report
    report = classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=4
    )
    with open(os.path.join(results_dir, "classification_report_v2.txt"), "w") as f:
        f.write(report)

    print("\nSaved Improved CNN results to:", results_dir)


if __name__ == "__main__":
    main()

