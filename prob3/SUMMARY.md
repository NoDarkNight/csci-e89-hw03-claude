# Problem 3 — Fashion MNIST MLP classifier in PyTorch: session summary

## What was asked

Ethan asked for a complete Fashion MNIST image classifier in PyTorch, built
fresh inside `prob3/`. Each step was to be its own Python script with its own
commit: data, model, training, plotting, prediction and Optuna tuning. Every
script also had to work when pasted into consecutive Jupyter cells. After
that: run everything, commit the outputs, write this summary, and wrap the
code into one notebook. The exact prompt is in [PROMPTS.md](PROMPTS.md).

## Scripts: commits and run order

Run them in this order from inside `prob3/` (`python data.py`, and so on), or
paste them into consecutive Jupyter cells in the same order.

| Order | Script | What it does | Created in commit |
|---|---|---|---|
| 1 | `data.py` | Downloads Fashion MNIST, converts to normalized float32 tensors, seed 42, 55k/5k train/validation split, DataLoaders (batch 32) | `a187a32` |
| 2 | `model.py` | `MLP` nn.Module: Flatten → 300 → ReLU → 100 → ReLU → 10 logits (`hidden_sizes` configurable) | `3921d9e` |
| 3 | `train.py` | SGD (lr 0.1) + cross-entropy for 20 epochs; torchmetrics accuracy; test accuracy; saves weights, `history.json`; calls the plot | `4067e4a` |
| 4 | `plot.py` | Training vs. validation accuracy per epoch → `accuracy.png`, shown inline in Jupyter | `1134a72` |
| 5 | `predict.py` | First 3 validation images: predicted/true class, all 10 softmax probabilities (3 decimals), top 4 | `6206f17` |
| 6 | `tune.py` | Seeded Optuna search (10 trials × 5 epochs), trial table, best params, 20-epoch retrain, baseline comparison | `529fa92` (warning fix in `64f3758`) |

Other commits: `79fbdaa` added `.gitignore` and `PROMPTS.md`. `64f3758` added
the run outputs and the one fix.

## What was built in each step

1. **`data.py`** converts the whole dataset to tensors once, instead of using
   a per-image `transforms.ToTensor()`. The images are scaled to [0, 1] and
   standardized with the mean and std of the 55,000 training images only
   (mean 0.2856, std 0.3527), so no information leaks from validation. The
   55k/5k split uses `random_split` with a seeded generator, and the shuffling
   training loader has its own seeded generator. The result is fully
   reproducible: the pasted-cell test and the notebook repeated the script
   results to the last digit. The script also provides `CLASS_NAMES`,
   `set_seed()`, `get_datasets()` and `get_dataloaders()`. Batches per epoch:
   1,719 train, 157 validation, 313 test.
2. **`model.py`**: `MLP(hidden_sizes=(300, 100))`, with 266,610 trainable
   parameters. The output is raw logits because `CrossEntropyLoss` applies the
   softmax itself.
3. **`train.py`** has `train_one_epoch`, `evaluate` and `fit` functions, which
   `tune.py` reuses. Accuracy uses `MulticlassAccuracy(num_classes=10,
   average="micro")`. Training accuracy is accumulated over the epoch's
   batches while the weights change, the way Keras reports it. Validation and
   test accuracy are measured in `eval()` mode. The GPU is used when
   available; this run was on the CPU at about 3.4 s per epoch.
4. **`plot.py`**: `plot_history(history)` draws both curves plus a dashed line
   for test accuracy. It uses the Agg backend outside Jupyter and `plt.show()`
   inline inside Jupyter.
5. **`predict.py`** loads `mlp_fashion_mnist.pt` and uses `torch.softmax` and
   `torch.topk`.
6. **`tune.py`** uses a `TPESampler(seed=42)` over lr (log scale, 1e-5 to
   1e-1), hidden layer 1 (50–500), hidden layer 2 (10–300) and momentum
   (0–0.99). Each trial reseeds, trains for 5 epochs and returns the final
   validation accuracy. The script saves `optuna_trials.csv`,
   `tuning_results.json`, `accuracy_tuned.png` and the tuned weights.

### How the scripts work as Jupyter cells

- There is no argparse and no `sys.argv`, so Jupyter's `-f kernel.json`
  argument doesn't matter.
- Each script's main code sits under `if __name__ == "__main__":`, which is
  also true inside a notebook cell, so pasted cells run.
- Local imports (`from data import ...`) are wrapped in `try/except
  ImportError: pass`. Without the .py files, later cells reuse the names that
  earlier cells defined.
- `BASE_DIR` is the script's folder, or `Path.cwd()` in Jupyter where
  `__file__` is undefined.
- `train.py` calls `plot_history` when it can be imported. In pasted-cell
  order the plot cell comes after the train cell, so the train cell prints a
  note and the plot cell draws the chart from the `history` it left behind.

## Problems encountered and fixes

1. **PyTorch was not installed, and `download.pytorch.org` is blocked** by
   this environment's network policy. I installed `torch`/`torchvision` from
   PyPI instead (2.14.0 / 0.29.0; the CUDA build runs fine on the CPU), along
   with torchmetrics 1.9.0 and Optuna 5.0.0. This changed nothing in the code.
2. **torchmetrics' default averaging is macro.** `MulticlassAccuracy` defaults
   to `average="macro"`, the mean of per-class accuracies, and the task asked
   for plain correct/total. I set `average="micro"` from the start. It is
   documented in `train.py` because it's an easy mistake to miss.
3. **Speed:** a per-image `ToTensor`/`Normalize` transform makes the Python
   DataLoader the bottleneck. Converting all images to one normalized tensor
   up front (`TensorDataset`) brought epochs down to about 3.5 s on 4 CPU
   cores, so the whole pipeline takes about 7 minutes.
4. **Jupyter warning from Optuna.** When I ran the six scripts pasted verbatim
   into a scratch notebook (no .py files present), everything worked. But
   `import optuna` printed `TqdmWarning: IProgress not found` because
   ipywidgets isn't installed. `tune.py` now filters that warning (commit
   `64f3758`).
5. **Tuning did not beat the baseline** (see below). This is a result, not a
   bug, so I did not tweak anything to make the numbers look better.

All runs were done one at a time: never two trainings at once.

## Final results

Baseline (784-300-100-10, SGD lr 0.1, no momentum, 20 epochs, seed 42, CPU):

| | Train acc | Validation acc | Test acc |
|---|---|---|---|
| Epoch 1 | 0.8182 | 0.8558 | |
| Epoch 20 (final) | **0.9505** | **0.8908** | **0.8894** |

Validation accuracy levels off near 0.885–0.89 after about epoch 5 while
training accuracy keeps rising to 0.95, which is mild overfitting (see
`accuracy.png`).

Predictions (`predict.py`): all 3 validation images are correct. They are a
Sneaker (p = 0.999), a Coat (0.995, runner-up Pullover 0.005) and a Pullover
(0.987, runner-up Shirt 0.013).

Optuna (10 trials × 5 epochs): the best trial was #2 with lr = 2.54e-3,
hidden = (369, 15), momentum = 0.960, reaching validation accuracy 0.8760
after 5 epochs. Every trial with lr below about 7e-4 scored under 0.82.

| After 20 epochs | Train acc | Validation acc | Test acc |
|---|---|---|---|
| Baseline (lr 0.1, 300/100, momentum 0) | 0.9505 | 0.8908 | **0.8894** |
| Tuned (lr 2.54e-3, 369/15, momentum 0.960) | 0.9465 | 0.8888 | **0.8875** |

The tuned model ends **0.19 percentage points below** the baseline on the
test set, which is within run-to-run noise. The reasons:

- Ten trials of 5 epochs is a very small budget.
- The log-uniform range spends most trials on learning rates that are far
  too small for 5 epochs.
- Trials are ranked on 5-epoch accuracy, which favours fast early progress
  rather than the best 20-epoch result.

The baseline's lr 0.1 is already a strong setting for this model. More
trials, a narrower lr range (about 1e-3 to 3e-1), or ranking on longer runs
would be the next steps.

## Files in `prob3/`

- Scripts: `data.py`, `model.py`, `train.py`, `plot.py`, `predict.py`, `tune.py`
- Outputs:
  - `history.json`: per-epoch baseline history plus test accuracy and config
  - `accuracy.png`, `accuracy_tuned.png`
  - `optuna_trials.csv`
  - `tuning_results.json`: trials, best parameters, tuned history, baseline vs. tuned
  - `train_log.txt`, `tuning_log.txt`: console output
  - `mlp_fashion_mnist.pt`, `mlp_fashion_mnist_tuned.pt`: model weights
- `PROMPTS.md` (prompt log), this `SUMMARY.md`, `.gitignore` (`data/`,
  `__pycache__/`, `.ipynb_checkpoints/`)
