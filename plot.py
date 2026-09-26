# =============================================================================
# plot.py — Plot training vs. validation accuracy per epoch
# -----------------------------------------------------------------------------
# - One chart: training accuracy and validation accuracy per epoch, with the
#   final test accuracy as a dashed reference line (if present in the history)
# - Saves the chart as a PNG
# - Displays it inline when running in Jupyter; in a plain script it only saves
#
# Usage:
#   python plot.py                -> regenerates plots from history.json
#   plot_history(history)         -> called by train.py after training
#
# Notebook-friendly: place this cell BEFORE the train.py cell so that train.py
# can call `plot_history`. When pasted into a cell, the block at the bottom
# plots the in-memory `history` if it exists, otherwise history.json.
# =============================================================================

import json
import os

import matplotlib.pyplot as plt

ACCURACY_PLOT_PATH = "accuracy.png"

# Colors: categorical slots 1 and 2 of a colorblind-validated palette;
# text/grid use neutral inks so identity is carried by the lines + labels.
TRAIN_COLOR = "#2a78d6"   # blue
VAL_COLOR = "#eb6834"     # orange
TEST_COLOR = "#8a8984"    # neutral gray (reference line)
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


def _in_notebook():
    """True when running inside a Jupyter/IPython kernel."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def _show_or_close(fig):
    """Display inline in Jupyter; otherwise just free the figure."""
    if _in_notebook():
        plt.show()
    else:
        plt.close(fig)


def plot_accuracy(history, save_path=ACCURACY_PLOT_PATH):
    """Training and validation accuracy per epoch on one chart."""
    train_acc = [a * 100 for a in history["train_acc"]]
    val_acc = [a * 100 for a in history["val_acc"]]
    epochs = range(1, len(train_acc) + 1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(epochs, train_acc, color=TRAIN_COLOR, linewidth=2,
            marker="o", markersize=5, label="Training")
    ax.plot(epochs, val_acc, color=VAL_COLOR, linewidth=2,
            marker="o", markersize=5, label="Validation")

    if "test_acc" in history:
        test_acc = history["test_acc"] * 100
        ax.axhline(test_acc, color=TEST_COLOR, linewidth=1.2, linestyle="--",
                   label=f"Test (final): {test_acc:.2f}%")

    # Direct labels at the end of each line (value in text ink, not series color).
    for series in (train_acc, val_acc):
        ax.annotate(f"{series[-1]:.2f}%", xy=(epochs[-1], series[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TEXT_PRIMARY)

    ax.set_title("Training vs. validation accuracy", color=TEXT_PRIMARY,
                 fontsize=13, loc="left", pad=12)
    ax.set_xlabel("Epoch", color=TEXT_SECONDARY)
    ax.set_ylabel("Accuracy (%)", color=TEXT_SECONDARY)
    ax.set_xticks(list(epochs))
    ax.set_xlim(0.5, epochs[-1] + 1.5)  # room for the end labels

    # Recessive axes and grid.
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)

    ax.legend(frameon=False, loc="lower right", labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved accuracy plot to {save_path}")
    _show_or_close(fig)


def plot_history(history):
    """Generate all training plots from a history dict."""
    plot_accuracy(history)


# ----------------------------- Regenerate ------------------------------------
if __name__ == "__main__":
    try:
        history  # from the train.py cell, if it has already run
    except NameError:
        history = None
        if os.path.exists("history.json"):
            with open("history.json") as f:
                history = json.load(f)

    if history is not None:
        plot_history(history)
    else:
        print("No history found yet; train.py will call plot_history() after training.")
