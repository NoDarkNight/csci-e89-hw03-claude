"""Step 1 — data.py: load Fashion MNIST and build the DataLoaders.

- Downloads Fashion MNIST with torchvision into prob3/data/.
- Converts the uint8 images to float32 tensors in [0, 1] and normalizes them
  with the training-set mean and standard deviation.
- Seeds everything with 42 and splits the 60,000 training images into
  55,000 training / 5,000 validation images.
- Builds DataLoaders with batch size 32 (only the training loader shuffles).

Works as a script (`python data.py`) or pasted into a Jupyter cell.
"""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets

# ---------------------------------------------------------------- constants
SEED = 42
BATCH_SIZE = 32
N_TRAIN, N_VALID = 55_000, 5_000

# Output folder: the folder holding this script, or the notebook's working
# directory when pasted into Jupyter (where __file__ is not defined).
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR / "data"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch (CPU and GPU) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_normalized_tensors(dataset, mean=None, std=None):
    """Turn a torchvision dataset into (images, labels) tensors.

    Images become float32 of shape (N, 1, 28, 28), scaled to [0, 1] and then
    standardized with `mean`/`std` (computed from these images if not given).
    Converting everything once up front is much faster than applying a
    per-image transform every time a batch is loaded.
    """
    x = dataset.data.unsqueeze(1).to(torch.float32) / 255.0
    if mean is None:
        mean, std = x.mean().item(), x.std().item()
    x = (x - mean) / std
    y = dataset.targets.clone().long()
    return x, y, mean, std


def get_datasets(data_dir=DATA_DIR, seed: int = SEED):
    """Download Fashion MNIST and return (train_set, valid_set, test_set, stats)."""
    train_full = datasets.FashionMNIST(data_dir, train=True, download=True)
    test_raw = datasets.FashionMNIST(data_dir, train=False, download=True)

    # Split indices first so the normalization statistics come from the
    # 55,000 training images only (no information leaks from validation).
    generator = torch.Generator().manual_seed(seed)
    train_idx, valid_idx = random_split(range(len(train_full)), [N_TRAIN, N_VALID],
                                        generator=generator)
    train_idx, valid_idx = list(train_idx), list(valid_idx)

    x_all = train_full.data.unsqueeze(1).to(torch.float32) / 255.0
    mean = x_all[train_idx].mean().item()
    std = x_all[train_idx].std().item()

    x_full, y_full, _, _ = to_normalized_tensors(train_full, mean, std)
    x_test, y_test, _, _ = to_normalized_tensors(test_raw, mean, std)

    train_set = TensorDataset(x_full[train_idx], y_full[train_idx])
    valid_set = TensorDataset(x_full[valid_idx], y_full[valid_idx])
    test_set = TensorDataset(x_test, y_test)
    return train_set, valid_set, test_set, {"mean": mean, "std": std}


def get_dataloaders(batch_size: int = BATCH_SIZE, seed: int = SEED, data_dir=DATA_DIR):
    """Return (train_loader, valid_loader, test_loader); only training shuffles."""
    set_seed(seed)
    train_set, valid_set, test_set, _ = get_datasets(data_dir, seed)
    # A dedicated seeded generator makes the shuffling order reproducible.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    set_seed(SEED)
    train_set, valid_set, test_set, stats = get_datasets()
    train_loader, valid_loader, test_loader = get_dataloaders()

    print(f"Normalization (from training split): mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    print(f"Train / valid / test sizes: {len(train_set)} / {len(valid_set)} / {len(test_set)}")
    print(f"Batches per epoch: train={len(train_loader)}, valid={len(valid_loader)}, "
          f"test={len(test_loader)}")
    images, labels = next(iter(train_loader))
    print(f"One batch: images {tuple(images.shape)} {images.dtype}, labels {tuple(labels.shape)}")
    print("First labels:", [CLASS_NAMES[i] for i in labels[:5].tolist()])
