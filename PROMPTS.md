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
