# =============================================================================
# data.py — Fashion MNIST data loading and preprocessing
# -----------------------------------------------------------------------------
# - Seeds all random number generators (seed = 42) for reproducibility
# - Downloads Fashion MNIST via torchvision
# - Splits the 60,000 training images into 55,000 train / 5,000 validation
# - Converts images to float32 tensors and normalizes them using the mean/std
#   of the 55,000-image training split (so no validation/test info leaks in)
# - Builds DataLoaders with batch size 32
#
# Notebook-friendly: no command-line arguments and no relative imports, so this
# file can be pasted into a Jupyter cell as-is. After it runs, `train_loader`,
# `val_loader`, `test_loader`, `CLASS_NAMES`, and `device` are available to
# later cells/scripts.
# =============================================================================

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ----------------------------- Configuration ---------------------------------
SEED = 42
DATA_DIR = "./data"
BATCH_SIZE = 32
NUM_TRAIN = 55_000
NUM_VAL = 5_000
NUM_WORKERS = 0  # 0 avoids multiprocessing issues inside Jupyter on macOS/Windows

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ----------------------------- Reproducibility -------------------------------
def set_seed(seed=SEED):
    """Seed Python, NumPy, and PyTorch (CPU and GPU) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ----------------------------- Train/val split -------------------------------
# Download once (without transforms) to get the dataset size and raw pixels.
_raw_train = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True)
assert len(_raw_train) == NUM_TRAIN + NUM_VAL, "Unexpected training set size"

# Seeded permutation -> identical split every run.
_split_gen = torch.Generator().manual_seed(SEED)
_perm = torch.randperm(len(_raw_train), generator=_split_gen)
train_indices = _perm[:NUM_TRAIN].tolist()
val_indices = _perm[NUM_TRAIN:].tolist()


# ----------------------------- Normalization ---------------------------------
# Compute mean/std over the training split only. `.data` is a uint8 tensor of
# shape (60000, 28, 28); scale to [0, 1] float32 to match ToTensor().
_train_pixels = _raw_train.data[train_indices].to(torch.float32) / 255.0
MEAN = _train_pixels.mean().item()
STD = _train_pixels.std().item()
del _train_pixels

# ToTensor(): PIL image (uint8, HxW) -> float32 tensor (1x28x28) in [0, 1]
# Normalize(): (x - MEAN) / STD
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,)),
])


# ----------------------------- Datasets --------------------------------------
_full_train = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform)

train_dataset = Subset(_full_train, train_indices)
val_dataset = Subset(_full_train, val_indices)


# ----------------------------- DataLoaders -----------------------------------
def _seed_worker(worker_id):
    """Keep per-worker RNGs reproducible if NUM_WORKERS > 0."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


_loader_gen = torch.Generator().manual_seed(SEED)
_pin_memory = device.type == "cuda"

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=_pin_memory,
    worker_init_fn=_seed_worker, generator=_loader_gen,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=_pin_memory,
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=_pin_memory,
)


# ----------------------------- Sanity check ----------------------------------
# Runs when executed as a script (`python data.py`) and in a Jupyter cell.
if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"Normalization: mean={MEAN:.4f}, std={STD:.4f}")
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
    print(f"Batches -> train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")

    # Drawing a batch advances the loader's seeded shuffling generator; restore
    # its state so a notebook cell running this check doesn't change training.
    _gen_state = _loader_gen.get_state()
    images, labels = next(iter(train_loader))
    _loader_gen.set_state(_gen_state)
    print(f"Batch images: shape={tuple(images.shape)}, dtype={images.dtype}")
    print(f"Batch labels: shape={tuple(labels.shape)}, dtype={labels.dtype}")
    print(f"Batch pixel mean={images.mean():.3f}, std={images.std():.3f}")
    print(f"First labels: {[CLASS_NAMES[i] for i in labels[:5].tolist()]}")
