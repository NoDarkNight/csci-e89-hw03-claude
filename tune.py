# =============================================================================
# tune.py — Hyperparameter tuning with Optuna
# -----------------------------------------------------------------------------
# Search space (SGD + cross-entropy, batch size 32, as in train.py):
#   - learning rate          : 1e-5 to 1e-1, log scale
#   - hidden layer 1 neurons : 50 to 500
#   - hidden layer 2 neurons : 25 to 300
#   - SGD momentum           : 0.0 to 0.99
# 30 trials x 5 epochs, maximizing validation accuracy. A median pruner stops
# clearly bad trials early. Then the best configuration is retrained for 20
# epochs and its test accuracy is compared with the train.py baseline.
#
# Speed: the datasets are converted to tensors once and kept on `device`, and
# batches are sliced from them directly. This skips the per-image PIL->tensor
# transform in the DataLoader, which dominates runtime for a model this small.
# The data (and its normalization) is identical to what the DataLoaders yield.
#
# Notebook-friendly: paste into a Jupyter cell after the data.py and model.py
# cells; it reuses `train_dataset`, `val_dataset`, `test_dataset`, `device`,
# `BATCH_SIZE`, and `MLPClassifier`. Run standalone, it imports them.
# =============================================================================

import json
import os
import time

import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

# Reuse objects from earlier notebook cells, or import them when run as a script.
try:
    train_dataset, val_dataset, test_dataset, device, BATCH_SIZE, MLPClassifier
except NameError:
    from data import train_dataset, val_dataset, test_dataset, device, BATCH_SIZE
    from model import MLPClassifier

# ----------------------------- Configuration ---------------------------------
SEED = 42
N_TRIALS = 30
TUNE_EPOCHS = 5
FINAL_EPOCHS = 20
NUM_CLASSES = 10
BASELINE_HISTORY_PATH = "history.json"
TRIALS_CSV_PATH = "optuna_trials.csv"
RESULTS_PATH = "tuning_results.json"
TUNED_WEIGHTS_PATH = "mlp_fashion_mnist_tuned.pt"


# ----------------------------- Fast in-memory data ---------------------------
def to_tensors(dataset):
    """Materialize a dataset as (images, labels) tensors on `device`."""
    images, labels = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    return images.to(device), labels.to(device)


def batches(images, labels, batch_size, shuffle, generator=None):
    """Yield (images, labels) mini-batches, like a DataLoader would."""
    n = labels.size(0)
    order = (torch.randperm(n, generator=generator).to(device) if shuffle
             else torch.arange(n, device=device))
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield images[idx], labels[idx]


print("Loading datasets into memory...")
X_train, y_train = to_tensors(train_dataset)
X_val, y_val = to_tensors(val_dataset)
X_test, y_test = to_tensors(test_dataset)


# ----------------------------- Train / evaluate ------------------------------
def fit_one_epoch(model, optimizer, criterion, generator):
    model.train()
    for images, labels in batches(X_train, y_train, BATCH_SIZE, shuffle=True, generator=generator):
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def accuracy(model, images, labels):
    """Plain (micro-averaged) multiclass accuracy via torchmetrics."""
    model.eval()
    metric = MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro").to(device)
    for xb, yb in batches(images, labels, 1024, shuffle=False):
        metric.update(model(xb), yb)
    return metric.compute().item()


def build(params):
    """Seeded model + SGD optimizer for a set of hyperparameters."""
    torch.manual_seed(SEED)
    model = MLPClassifier(hidden_dims=(params["hidden1"], params["hidden2"])).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=params["momentum"])
    return model, optimizer


# ----------------------------- Optuna objective ------------------------------
def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        "hidden1": trial.suggest_int("hidden1", 50, 500, step=25),
        "hidden2": trial.suggest_int("hidden2", 25, 300, step=25),
        "momentum": trial.suggest_float("momentum", 0.0, 0.99),
    }
    model, optimizer = build(params)
    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(SEED)

    val_acc = 0.0
    for epoch in range(1, TUNE_EPOCHS + 1):
        fit_one_epoch(model, optimizer, criterion, gen)
        val_acc = accuracy(model, X_val, y_val)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return val_acc


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
)

print(f"Running {N_TRIALS} trials x up to {TUNE_EPOCHS} epochs on {device}...")
start = time.time()
study.optimize(objective, n_trials=N_TRIALS)
print(f"Search time: {time.time() - start:.1f}s\n")


# ----------------------------- Trials table ----------------------------------
trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
trials_df = trials_df.rename(columns=lambda c: c.replace("params_", ""))
trials_df = trials_df.rename(columns={"value": "val_acc"})
trials_df = trials_df[["number", "state", "val_acc", "lr", "momentum", "hidden1", "hidden2"]]
trials_df.to_csv(TRIALS_CSV_PATH, index=False)

print("Trials:")
print(trials_df.to_string(index=False, formatters={
    "val_acc": lambda v: f"{v:.4f}",
    "lr": lambda v: f"{v:.2e}",
    "momentum": lambda v: f"{v:.3f}",
}))

best_params = study.best_params
print(f"\nBest trial: #{study.best_trial.number} (val acc {study.best_value:.4f})")
print("Best parameters:")
for name, value in best_params.items():
    print(f"  {name:<9} {value:.3e}" if name == "lr" else f"  {name:<9} {value}")


# ----------------------------- Retrain best ----------------------------------
print(f"\nRetraining best configuration for {FINAL_EPOCHS} epochs...")
model_tuned, optimizer = build(best_params)
criterion = nn.CrossEntropyLoss()
gen = torch.Generator().manual_seed(SEED)
tuned_history = {"val_acc": []}
for epoch in range(1, FINAL_EPOCHS + 1):
    fit_one_epoch(model_tuned, optimizer, criterion, gen)
    val_acc = accuracy(model_tuned, X_val, y_val)
    tuned_history["val_acc"].append(val_acc)
    print(f"Epoch {epoch:2d}/{FINAL_EPOCHS} | val acc {val_acc:.4f}")

tuned_test_acc = accuracy(model_tuned, X_test, y_test)
torch.save(model_tuned.state_dict(), TUNED_WEIGHTS_PATH)


# ----------------------------- Compare with baseline -------------------------
try:
    baseline_test_acc = history["test_acc"]  # from the train.py cell
except NameError:
    baseline_test_acc = None
    if os.path.exists(BASELINE_HISTORY_PATH):
        with open(BASELINE_HISTORY_PATH) as f:
            baseline_test_acc = json.load(f)["test_acc"]

print("\n=== Test accuracy ===")
if baseline_test_acc is not None:
    print(f"{'Baseline (train.py: lr=0.1, momentum=0, 300/100)':<50} {baseline_test_acc:.4f}")
print(f"{'Tuned (Optuna best, retrained ' + str(FINAL_EPOCHS) + ' epochs)':<50} {tuned_test_acc:.4f}")
if baseline_test_acc is not None:
    diff = tuned_test_acc - baseline_test_acc
    print(f"Difference: {diff * 100:+.2f} percentage points")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "n_trials": N_TRIALS,
        "tune_epochs": TUNE_EPOCHS,
        "best_trial": study.best_trial.number,
        "best_val_acc_during_search": study.best_value,
        "best_params": best_params,
        "final_epochs": FINAL_EPOCHS,
        "tuned_val_acc": tuned_history["val_acc"],
        "tuned_test_acc": tuned_test_acc,
        "baseline_test_acc": baseline_test_acc,
    }, f, indent=2)
print(f"\nSaved trials to {TRIALS_CSV_PATH}, results to {RESULTS_PATH}, "
      f"weights to {TUNED_WEIGHTS_PATH}")
