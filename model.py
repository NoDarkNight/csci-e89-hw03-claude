# =============================================================================
# model.py — MLP classifier for Fashion MNIST
# -----------------------------------------------------------------------------
# Architecture:
#   Input (1x28x28) -> Flatten (784)
#   -> Linear(784, 300) -> ReLU
#   -> Linear(300, 100) -> ReLU
#   -> Linear(100, 10)  -> logits (one per class)
#
# The output is raw logits (no softmax); nn.CrossEntropyLoss applies
# log-softmax internally during training.
#
# Notebook-friendly: paste into a Jupyter cell after the data.py cell. After it
# runs, `MLPClassifier` and `model` (already on `device`) are available.
# =============================================================================

import torch
from torch import nn

# `device` comes from data.py when run in the notebook; fall back if run alone.
try:
    device
except NameError:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

INPUT_DIM = 28 * 28
HIDDEN_DIMS = (300, 100)
NUM_CLASSES = 10


class MLPClassifier(nn.Module):
    """Multilayer perceptron: 784 -> 300 -> 100 -> 10 with ReLU activations."""

    def __init__(self, input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, num_classes=NUM_CLASSES):
        super().__init__()
        h1, h2 = hidden_dims
        self.flatten = nn.Flatten()  # (N, 1, 28, 28) -> (N, 784)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x):
        return self.layers(self.flatten(x))  # logits, shape (N, num_classes)


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Re-seed right before construction so weight initialization is reproducible
# regardless of how many random draws happened in earlier cells.
torch.manual_seed(42)
model = MLPClassifier().to(device)


# ----------------------------- Sanity check ----------------------------------
if __name__ == "__main__":
    print(model)
    print(f"Trainable parameters: {count_parameters(model):,}")

    dummy = torch.randn(32, 1, 28, 28, device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input shape: {tuple(dummy.shape)} -> Output shape: {tuple(out.shape)}")
