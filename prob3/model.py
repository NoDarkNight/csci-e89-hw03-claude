"""Step 2 — model.py: the MLP classifier.

Flatten (28x28 -> 784) -> Linear(300) -> ReLU -> Linear(100) -> ReLU -> Linear(10).
The output layer returns raw logits (one per class); CrossEntropyLoss applies
the softmax internally. Hidden sizes are configurable via `hidden_sizes`.
"""
import torch
from torch import nn


class MLP(nn.Module):
    """Multilayer perceptron for 28x28 grayscale images and 10 classes."""

    def __init__(self, hidden_sizes=(300, 100), input_size: int = 28 * 28,
                 num_classes: int = 10):
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        layers = [nn.Flatten()]
        in_features = input_size
        for size in self.hidden_sizes:
            layers += [nn.Linear(in_features, size), nn.ReLU()]
            in_features = size
        layers.append(nn.Linear(in_features, num_classes))  # logits, no softmax
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MLP()
    print(model)
    print(f"Trainable parameters: {count_parameters(model):,}")
    # Sanity check: a dummy batch of 32 images gives 32 x 10 logits.
    print("Output shape for a batch of 32:", tuple(model(torch.zeros(32, 1, 28, 28)).shape))
