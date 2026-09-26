# =============================================================================
# predict.py — Predictions with the trained MLP
# -----------------------------------------------------------------------------
# For the first 3 images of the validation set, prints:
#   - the predicted and true class names
#   - softmax probabilities for all 10 classes (rounded to 3 decimals)
#   - the top 4 most likely classes
#
# Uses the weights saved by train.py (mlp_fashion_mnist.pt).
#
# Notebook-friendly: paste into a Jupyter cell after the data.py, model.py
# (and optionally train.py) cells; it reuses `val_dataset`, `CLASS_NAMES`,
# `device`, and `model` from those cells. Run standalone, it imports them.
# =============================================================================

import torch

# Reuse objects from earlier notebook cells, or import them when run as a script.
try:
    val_dataset, CLASS_NAMES, device
except NameError:
    from data import val_dataset, CLASS_NAMES, device
try:
    model
except NameError:
    from model import MLPClassifier
    model = MLPClassifier()

# ----------------------------- Configuration ---------------------------------
WEIGHTS_PATH = "mlp_fashion_mnist.pt"
NUM_IMAGES = 3
TOP_K = 4

# Load the trained weights (map_location lets GPU-trained weights load on CPU).
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model = model.to(device)
model.eval()

# ----------------------------- Predict ---------------------------------------
# Stack the first NUM_IMAGES validation samples into one batch: (3, 1, 28, 28).
images = torch.stack([val_dataset[i][0] for i in range(NUM_IMAGES)]).to(device)
true_labels = [val_dataset[i][1] for i in range(NUM_IMAGES)]

with torch.no_grad():
    logits = model(images)
    probs = torch.softmax(logits, dim=1).cpu()  # (3, 10), each row sums to 1

pred_labels = probs.argmax(dim=1).tolist()
top_probs, top_idx = probs.topk(TOP_K, dim=1)

# ----------------------------- Report ----------------------------------------
name_width = max(len(name) for name in CLASS_NAMES)

for i in range(NUM_IMAGES):
    pred, true = pred_labels[i], true_labels[i]
    status = "correct" if pred == true else "WRONG"
    print(f"=== Validation image {i} ===")
    print(f"Predicted: {CLASS_NAMES[pred]}  |  True: {CLASS_NAMES[true]}  ({status})")

    print("Softmax probabilities:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {c}  {name:<{name_width}}  {probs[i, c].item():.3f}")

    print(f"Top {TOP_K} classes:")
    for rank, (p, c) in enumerate(zip(top_probs[i].tolist(), top_idx[i].tolist()), start=1):
        print(f"  {rank}. {CLASS_NAMES[c]:<{name_width}}  {p:.3f}")
    print()
