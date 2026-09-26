"""Step 4 — plot.py: plot training vs. validation accuracy per epoch.

Saves the chart as a PNG and shows it inline when running inside Jupyter.
train.py calls `plot_history` after training; run on its own, this script
reads the saved history.json.
"""
import json
import sys
from pathlib import Path

import matplotlib

# Outside Jupyter, use the non-interactive Agg backend so the script also
# works on machines without a display. Inside Jupyter keep the inline backend.
IN_JUPYTER = "ipykernel" in sys.modules
if not IN_JUPYTER:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PLOT_PATH = BASE_DIR / "accuracy.png"


def plot_history(history, path=PLOT_PATH, title="MLP on Fashion MNIST: accuracy per epoch"):
    """Plot train and validation accuracy from a history dict, save it as a PNG."""
    epochs = history.get("epoch") or list(range(1, len(history["train_accuracy"]) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_accuracy"], marker="o", label="Training accuracy")
    ax.plot(epochs, history["val_accuracy"], marker="s", label="Validation accuracy")
    if "test_accuracy" in history:
        ax.axhline(history["test_accuracy"], color="gray", linestyle="--", linewidth=1,
                   label=f"Test accuracy ({history['test_accuracy']:.4f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved plot to {Path(path).name}")
    if IN_JUPYTER:
        plt.show()  # displayed inline in the notebook
    else:
        plt.close(fig)
    return path


if __name__ == "__main__":
    # In Jupyter, reuse the `history` dict from the train.py cell if present;
    # otherwise load the history saved by train.py.
    if "history" not in globals():
        with open(BASE_DIR / "history.json") as f:
            history = json.load(f)
    plot_history(history)
