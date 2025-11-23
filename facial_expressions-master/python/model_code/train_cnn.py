import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from model_code.data_loader import (
    get_train_loader,
    get_val_loader,
    get_test_loader,
)

CLASS_NAMES = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

def convert_labels(labels, device):
    """Convert labels (strings/tuples) → LongTensor of class IDs."""
    if isinstance(labels, torch.Tensor):
        return labels.long().to(device)

    if isinstance(labels, (list, tuple)):
        processed = [LABEL_TO_IDX[str(l).lower()] for l in labels]
    else:
        processed = [LABEL_TO_IDX[str(labels).lower()]]

    return torch.tensor(processed, dtype=torch.long, device=device)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8, dropout=0.5):
        super(SimpleCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 256),  # 224x224 input → (224/4)=56
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = convert_labels(labels, device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = convert_labels(labels, device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def run_training(dropout, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_train_loader(batch_size=32)
    val_loader = get_val_loader(batch_size=32)

    model = SimpleCNN(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(6):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/6 | Train Acc={tr_acc:.4f}, Val Acc={val_acc:.4f}")

    # Save the last model of this configuration
    torch.save(model.state_dict(), "temp_cnn.pt")

    return val_acc, dropout, lr, train_losses, val_losses, train_accs, val_accs

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "../results_cnn")
    os.makedirs(results_dir, exist_ok=True)

    dropout_vals = [0.3, 0.5]
    lr_vals = [1e-3, 5e-4]

    best_model = None
    best_acc = -1

    for d in dropout_vals:
        for lr in lr_vals:
            print(f"\n=== Training CNN (dropout={d}, lr={lr}) for 6 epochs ===")
            acc, dropout, lr_used, tr_losses, val_losses, tr_accs, val_accs = run_training(d, lr)

            if acc > best_acc:
                best_acc = acc
                best_model = (dropout, lr_used, tr_losses, val_losses, tr_accs, val_accs)
                torch.save(torch.load("temp_cnn.pt"), "best_cnn.pt")

    best_dropout, best_lr, tr_losses, val_losses, tr_accs, val_accs = best_model

    print("\n=== BEST CNN CONFIGURATION ===")
    print("Dropout:", best_dropout)
    print("Learning Rate:", best_lr)
    print("Validation Accuracy:", best_acc)

    # Load best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(dropout=best_dropout).to(device)
    model.load_state_dict(torch.load("best_cnn.pt"))

    test_loader = get_test_loader(batch_size=32)

    model.eval()
    preds_list, labels_list = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels_tensor = convert_labels(labels, device).cpu()

            preds = model(images).argmax(1).cpu()

            preds_list.append(preds)
            labels_list.append(labels_tensor)

    preds = torch.cat(preds_list).numpy()
    labels = torch.cat(labels_list).numpy()

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("CNN Confusion Matrix (6 Epoch Final Model)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cnn_confusion_matrix.png"))

    # Accuracy curve
    epochs = range(1, 7)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, tr_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Accuracy Curve (6 Epochs)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cnn_accuracy_curve.png"))

    # Loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, tr_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Loss Curve (6 Epochs)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cnn_loss_curve.png"))

    # Classification report
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
    with open(os.path.join(results_dir, "cnn_classification_report.txt"), "w") as f:
        f.write(report)

    print("\nSaved CNN 6-epoch results to:", results_dir)


if __name__ == "__main__":
    main()
