"""Step 6 — tune.py: hyperparameter tuning with Optuna.

Search space (seeded TPE sampler, 10 trials x 5 epochs, maximize validation
accuracy):
- learning rate: log scale, 1e-5 .. 1e-1
- neurons in hidden layer 1: 50 .. 500
- neurons in hidden layer 2: 10 .. 300
- SGD momentum: 0.0 .. 0.99

Then retrains the best configuration for 20 epochs and compares its test
accuracy with the baseline from train.py.
"""
import json

import optuna
import pandas as pd
import torch

# --- local imports: when pasted into Jupyter without the .py files, these
# names already exist from the earlier cells, so the ImportError is ignored.
try:
    from data import BASE_DIR, SEED, get_dataloaders, set_seed
    from model import MLP
    from train import EPOCHS, device, evaluate, fit
except ImportError:
    pass
# --- end local imports

N_TRIALS = 10
TRIAL_EPOCHS = 5
TRIALS_CSV = BASE_DIR / "optuna_trials.csv"
RESULTS_JSON = BASE_DIR / "tuning_results.json"
TUNED_WEIGHTS = BASE_DIR / "mlp_fashion_mnist_tuned.pt"
TUNED_PLOT = BASE_DIR / "accuracy_tuned.png"


def objective(trial):
    """Train a sampled configuration for TRIAL_EPOCHS; return validation accuracy."""
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_hidden1 = trial.suggest_int("n_hidden1", 50, 500)
    n_hidden2 = trial.suggest_int("n_hidden2", 10, 300)
    momentum = trial.suggest_float("momentum", 0.0, 0.99)

    set_seed(SEED)  # same initialization/shuffling for every trial
    train_loader, valid_loader, _ = get_dataloaders()
    model = MLP(hidden_sizes=(n_hidden1, n_hidden2))
    hist = fit(model, train_loader, valid_loader, epochs=TRIAL_EPOCHS, lr=lr,
               momentum=momentum, device=device, verbose=False)
    val_acc = hist["val_accuracy"][-1]
    print(f"Trial {trial.number}: lr={lr:.2e} hidden=({n_hidden1}, {n_hidden2}) "
          f"momentum={momentum:.3f} -> val_acc={val_acc:.4f}")
    return val_acc


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # we print our own lines
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)

    # Table of all trials.
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "duration"))
    trials_df = trials_df.rename(columns={"value": "val_accuracy"})
    trials_df.columns = [c.replace("params_", "") for c in trials_df.columns]
    trials_df["duration"] = trials_df["duration"].dt.total_seconds().round(1)
    trials_df.to_csv(TRIALS_CSV, index=False)
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print("\nOptuna trials:")
        print(trials_df.round({"val_accuracy": 4, "lr": 6, "momentum": 3}).to_string(index=False))

    best = study.best_params
    print(f"\nBest trial: #{study.best_trial.number}, val_acc={study.best_value:.4f}")
    print("Best parameters:", best)

    # Retrain the best configuration for the full 20 epochs.
    print(f"\nRetraining the best configuration for {EPOCHS} epochs:")
    set_seed(SEED)
    train_loader, valid_loader, test_loader = get_dataloaders()
    tuned_model = MLP(hidden_sizes=(best["n_hidden1"], best["n_hidden2"]))
    tuned_history = fit(tuned_model, train_loader, valid_loader, epochs=EPOCHS,
                        lr=best["lr"], momentum=best["momentum"], device=device)
    tuned_test_acc = evaluate(tuned_model, test_loader, device)
    tuned_history["test_accuracy"] = tuned_test_acc
    torch.save(tuned_model.state_dict(), TUNED_WEIGHTS)

    # Baseline from train.py: the `history` dict from the train.py cell in
    # Jupyter, otherwise the history.json that train.py saved.
    if "history" in globals() and "test_accuracy" in globals()["history"]:
        baseline = globals()["history"]
    else:
        with open(BASE_DIR / "history.json") as f:
            baseline = json.load(f)

    print("\nBaseline vs. tuned (after 20 epochs):")
    print(f"{'':10}{'train_acc':>11}{'val_acc':>10}{'test_acc':>10}")
    for name, h in [("baseline", baseline), ("tuned", tuned_history)]:
        print(f"{name:10}{h['train_accuracy'][-1]:11.4f}{h['val_accuracy'][-1]:10.4f}"
              f"{h['test_accuracy']:10.4f}")
    diff = tuned_test_acc - baseline["test_accuracy"]
    print(f"Test accuracy change: {diff:+.4f} ({diff * 100:+.2f} percentage points)")

    results = {
        "n_trials": N_TRIALS, "trial_epochs": TRIAL_EPOCHS, "sampler": f"TPESampler(seed={SEED})",
        "best_trial": study.best_trial.number, "best_val_accuracy_5_epochs": study.best_value,
        "best_params": best,
        "trials": trials_df.to_dict(orient="records"),
        "tuned_history": tuned_history,
        "baseline_test_accuracy": baseline["test_accuracy"],
        "tuned_test_accuracy": tuned_test_acc,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {TRIALS_CSV.name}, {RESULTS_JSON.name}, {TUNED_WEIGHTS.name}")

    # Accuracy curves of the tuned model (plot_history comes from plot.py).
    try:
        from plot import plot_history
    except ImportError:
        pass
    plot_history(tuned_history, path=TUNED_PLOT,
                 title="Tuned MLP on Fashion MNIST: accuracy per epoch")
