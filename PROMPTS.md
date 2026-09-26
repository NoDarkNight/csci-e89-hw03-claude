# Prompt Log

A running log of each request made during this project.

---

## 1. Project setup + `data.py`

> Hello! I'd like your help in building an image classifier for the Fashion MNIST dataset with PyTorch. For each major step, write and commit a separate Python file.
>
> Some things additional things worth noting:
>
> 1. Part of the final goal includes creating a Jupyter Notebook containing all the scripts. Thus, make the scripts compatible with being pasted into Jupyter Notebook cells
> 2. Keep a file, PROMPTS.md, that logs each one of my requests. At the end of the project, I'll have you write a summary of the entire chat. The PROMPTS.md file may be useful for that :)
>
> Okay. The first step is to write "data.py". This should handle downloading Fashion MNIST with torchvision, converting and normalizing vectors to float32 tensors, setting the seed (42), splitting the 60,000 training images into 55,000 for training and 5,000 for validation, creating dataloaders with batch size 32, and whatever else is necessary.

**Result:** Created `data.py` (seeding, download, seeded 55k/5k split, float32 conversion + normalization using train-split mean/std, train/val/test DataLoaders at batch size 32, sanity-check printout), `.gitignore` (excludes `data/`), and this `PROMPTS.md`.

---

## 2. `model.py`

> Nice. Next, write model.py with a MLP classifier as an nn.Module. Flatten the 28x28 image, then use two hidden layers of 300 and 100 neurons with ReLU. The output layer should be 10 logits (one per class).

**Result:** Created `model.py` with `MLPClassifier` (Flatten → Linear 784→300 → ReLU → Linear 300→100 → ReLU → Linear 100→10, raw logits), a `count_parameters` helper (266,610 trainable parameters), a seeded `model` instance moved to `device`, and a shape sanity check.

---

## 3. `train.py`

> You got it! Write train.py -- train the model with SGD (learning rate 0.1) and cross-entropy loss for 20 epochs, using torchmetrics to compute multiclass accuracy. For each epoch, record the mean training loss, training accuracy, and validation accuracy in a variable and print them. Use a GPU if one is available, otherwise CPU is fine. After training, evaluate accuracy on the test set, save the model weights, and save the history (including test accuracy). Just as a reminder, make sure you also update PROMPTS.md.

**Result:** Created `train.py`: SGD (lr 0.1) + `CrossEntropyLoss` for 20 epochs, torchmetrics `MulticlassAccuracy` (`average="micro"`, i.e. plain accuracy), per-epoch `history` dict (train loss / train acc / val acc) printed each epoch, then test evaluation. Saves weights to `mlp_fashion_mnist.pt` and history (with `test_acc`) to `history.json`. Verified with a full CPU run (~3.5 min): final train acc 0.9505, val acc 0.8908, **test acc 0.8894**.

---

## 4. `plot.py`

> You've guessed it again :)
>
> Write plot.py to plot the training accuracy per epoch. Put validation accuracy on the same chart for comparison. Save it as png, but make it display them inline when running in Jupyter. Have train.py call the plotting after training. Regenerate the plots from history.json, then commit with the images and update PROMPTS.md.

**Result:** Created `plot.py` with `plot_accuracy` / `plot_history`: one chart of training and validation accuracy per epoch (plus final test accuracy as a dashed reference line), colorblind-safe colors, end-of-line value labels, saved to `accuracy.png`. Shown inline via `plt.show()` only when running in a Jupyter kernel; closed otherwise. `train.py` now calls `plot_history(history)` after saving (in the notebook, the plot.py cell goes before the train.py cell). Regenerated `accuracy.png` from the existing `history.json` without retraining.

---

## 5. `predict.py`

> Lets do some predictions now. Write predict.py to use the trained model: take the first 3 images from the validation set, print the predicted and true class names, the softmax probabilities for all 10 classes (rounded to 3 decimals), and the top 4 most likely classes for each image. Again, it should for Jupyter Notebook. Commit and update PROMPTS.md.

**Result:** Created `predict.py`: loads `mlp_fashion_mnist.pt`, batches the first 3 validation images, applies softmax, and prints predicted vs. true class, all 10 class probabilities (3 decimals), and the top-4 classes via `topk`. Verified run: all 3 correct — Sneaker (0.999), Coat (0.995, runner-up Pullover 0.005), Pullover (0.987, runner-up Shirt 0.013).

---

## 6. `tune.py` (Optuna hyperparameter tuning)

> Actually, let's now try some hyperparameter tuning. Use Optuna to search the learning rate (log scale from about 1e-5 to 1e-1), the number of neurons in each hidden layer, and the optimizer momentum. Try your best to keep it fast; maybe do about 10 trials of 5 epochs each. Print a table of the trials and the best parameters, then retrain the best configuration for 20 epochs and report its test accuracy compared with the baseline from train.py. Commit the results and update PROMPTS.md again.

**Result:** Created `tune.py`. Search space: lr 1e-5–1e-1 (log), hidden1 50–500, hidden2 25–300 (step 25), SGD momentum 0–0.99; 10 trials × 5 epochs, TPE sampler (seed 42) + median pruner (5 of 10 trials pruned). For speed, datasets are preloaded as tensors on `device` and batched by slicing (same data, batch size 32) — the whole search took ~58 s on CPU.

- Best trial #3: lr 2.14e-2, hidden 150/75, momentum 0.182 (val acc 0.8684 after 5 epochs)
- Retrained 20 epochs → **test acc 0.8850** vs. baseline **0.8894** (−0.44 pp): the tuned config did *not* beat the baseline. With only 10 trials, the search mostly sampled learning rates well below the baseline's effective step size.

Outputs committed: `optuna_trials.csv`, `tuning_results.json`, `tuning_log.txt`, `mlp_fashion_mnist_tuned.pt`.
