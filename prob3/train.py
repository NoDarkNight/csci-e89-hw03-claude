"""Step 3 — train.py: train the MLP with SGD and evaluate it.

- SGD, learning rate 0.1, cross-entropy loss, 20 epochs.
- torchmetrics MulticlassAccuracy with average="micro" = plain accuracy
  (correct / total). Note: torchmetrics defaults to average="macro", which is
  the mean of per-class accuracies, so "micro" has to be set explicitly.
- Per epoch: mean training loss, training accuracy, validation accuracy.
- Afterwards: test accuracy, model weights (.pt), history (.json), and the
  accuracy plot from plot.py.
"""
import json
import time

import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

# --- local imports: when pasted into Jupyter without the .py files, these
# names already exist from the earlier cells, so the ImportError is ignored.
try:
    from data import BASE_DIR, SEED, get_dataloaders, set_seed
    from model import MLP
except ImportError:
    pass
# --- end local imports

EPOCHS = 20
LEARNING_RATE = 0.1
WEIGHTS_PATH = BASE_DIR / "mlp_fashion_mnist.pt"
HISTORY_PATH = BASE_DIR / "history.json"

# Use a GPU if one is available, otherwise the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_accuracy(device):
    """Plain multiclass accuracy (correct / total) as a torchmetrics metric."""
    return MulticlassAccuracy(num_classes=10, average="micro").to(device)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """One pass over `loader`; returns (mean training loss, training accuracy).

    The training accuracy is accumulated over the batches while the weights
    are being updated (like Keras reports it).
    """
    model.train()
    accuracy = make_accuracy(device)
    total_loss, n_samples = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)  # sum of per-sample losses
        n_samples += labels.size(0)
        accuracy.update(logits.detach(), labels)
    return total_loss / n_samples, accuracy.compute().item()


@torch.no_grad()
def evaluate(model, loader, device):
    """Accuracy of `model` on every batch of `loader`."""
    model.eval()
    accuracy = make_accuracy(device)
    for images, labels in loader:
        accuracy.update(model(images.to(device)), labels.to(device))
    return accuracy.compute().item()


def fit(model, train_loader, valid_loader, epochs=EPOCHS, lr=LEARNING_RATE,
        momentum=0.0, device=device, verbose=True):
    """Train with SGD + cross-entropy; returns the per-epoch history dict."""
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    history = {"epoch": [], "train_loss": [], "train_accuracy": [], "val_accuracy": []}
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_acc = evaluate(model, valid_loader, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        if verbose:
            print(f"Epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
                  f"({time.time() - start:.1f}s)")
    return history


if __name__ == "__main__":
    print(f"Using device: {device}")
    set_seed(SEED)
    train_loader, valid_loader, test_loader = get_dataloaders()
    model = MLP(hidden_sizes=(300, 100))

    history = fit(model, train_loader, valid_loader, epochs=EPOCHS, lr=LEARNING_RATE)

    test_acc = evaluate(model, test_loader, device)
    history["test_accuracy"] = test_acc
    history["config"] = {"hidden_sizes": [300, 100], "lr": LEARNING_RATE, "momentum": 0.0,
                         "epochs": EPOCHS, "batch_size": train_loader.batch_size,
                         "seed": SEED, "device": str(device)}
    print(f"\nFinal: train_acc={history['train_accuracy'][-1]:.4f}  "
          f"val_acc={history['val_accuracy'][-1]:.4f}  test_acc={test_acc:.4f}")

    torch.save(model.state_dict(), WEIGHTS_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved weights to {WEIGHTS_PATH.name} and history to {HISTORY_PATH.name}")

    # Plot the accuracy curves with plot.py. When the cells are pasted into
    # Jupyter in order, plot.py's cell comes after this one, so the plot is
    # drawn there instead.
    try:
        from plot import plot_history
    except ImportError:
        plot_history = globals().get("plot_history")
    if plot_history is not None:
        plot_history(history)
    else:
        print("plot_history is not defined yet; the plot.py cell draws the chart.")
