# =============================================================================
# train.py — Train the MLP on Fashion MNIST
# -----------------------------------------------------------------------------
# - Optimizer: SGD, learning rate 0.1
# - Loss: cross-entropy
# - 20 epochs; per epoch records mean training loss, training accuracy, and
#   validation accuracy (torchmetrics multiclass accuracy) in `history`
# - Runs on a GPU if available, otherwise CPU
# - After training: evaluates test accuracy, saves model weights and history
#
# Notebook-friendly: paste into a Jupyter cell after the data.py and model.py
# cells; it reuses `train_loader`, `val_loader`, `test_loader`, `model`, and
# `device` from those cells. Run standalone, it imports them from the files.
# =============================================================================

import json
import time

import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

# Reuse objects from earlier notebook cells, or import them when run as a script.
try:
    train_loader, val_loader, test_loader, model, device
except NameError:
    from data import train_loader, val_loader, test_loader, device
    from model import MLPClassifier
    torch.manual_seed(42)
    model = MLPClassifier()

# ----------------------------- Configuration ---------------------------------
NUM_EPOCHS = 20
LEARNING_RATE = 0.1
NUM_CLASSES = 10
WEIGHTS_PATH = "mlp_fashion_mnist.pt"
HISTORY_PATH = "history.json"

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# average="micro" gives standard accuracy (correct / total). torchmetrics'
# default for MulticlassAccuracy is "macro" (mean of per-class accuracies).
train_metric = MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro").to(device)
eval_metric = MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro").to(device)


# ----------------------------- Helpers ---------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, metric, device):
    """One pass over `loader`. Returns (mean loss per sample, accuracy)."""
    model.train()
    metric.reset()
    total_loss, total_samples = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Weight by batch size so a smaller final batch doesn't skew the mean.
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        metric.update(logits.detach(), labels)
    return total_loss / total_samples, metric.compute().item()


@torch.no_grad()
def evaluate(model, loader, metric, device):
    """Accuracy of `model` on `loader` (no gradient tracking)."""
    model.eval()
    metric.reset()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        metric.update(model(images), labels)
    return metric.compute().item()


# ----------------------------- Training loop ---------------------------------
history = {"train_loss": [], "train_acc": [], "val_acc": []}

print(f"Training on {device} for {NUM_EPOCHS} epochs (SGD, lr={LEARNING_RATE})")
start = time.time()
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, train_metric, device)
    val_acc = evaluate(model, val_loader, eval_metric, device)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
          f"train loss {train_loss:.4f} | train acc {train_acc:.4f} | val acc {val_acc:.4f}")
print(f"Training time: {time.time() - start:.1f}s")


# ----------------------------- Test + save -----------------------------------
test_acc = evaluate(model, test_loader, eval_metric, device)
history["test_acc"] = test_acc
print(f"Test accuracy: {test_acc:.4f}")

torch.save(model.state_dict(), WEIGHTS_PATH)
with open(HISTORY_PATH, "w") as f:
    json.dump(history, f, indent=2)
print(f"Saved weights to {WEIGHTS_PATH} and history to {HISTORY_PATH}")
