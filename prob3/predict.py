"""Step 5 — predict.py: predictions for the first 3 validation images.

For each image prints the predicted and true class names, the softmax
probabilities for all 10 classes (rounded to 3 decimals), and the top 4
most likely classes. Uses the weights saved by train.py.
"""
import torch

# --- local imports: when pasted into Jupyter without the .py files, these
# names already exist from the earlier cells, so the ImportError is ignored.
try:
    from data import BASE_DIR, CLASS_NAMES, SEED, get_datasets, set_seed
    from model import MLP
except ImportError:
    pass
# --- end local imports

N_IMAGES = 3
TOP_K = 4


@torch.no_grad()
def predict_proba(model, images, device="cpu"):
    """Softmax class probabilities for a batch of images."""
    model.eval()
    logits = model(images.to(device))
    return torch.softmax(logits, dim=1).cpu()


def show_predictions(probs, labels, class_names=CLASS_NAMES, top_k=TOP_K):
    """Print prediction, truth, all probabilities and the top-k classes per image."""
    for i, (p, true) in enumerate(zip(probs, labels)):
        pred = int(p.argmax())
        status = "correct" if pred == int(true) else "WRONG"
        print(f"Validation image {i}: predicted = {class_names[pred]!r}, "
              f"true = {class_names[int(true)]!r} ({status})")
        print("  Probabilities (3 decimals):")
        for name, prob in zip(class_names, p.tolist()):
            print(f"    {name:<12} {prob:.3f}")
        top_p, top_i = torch.topk(p, top_k)
        top = ", ".join(f"{class_names[j]} ({q:.3f})" for q, j in zip(top_p.tolist(), top_i.tolist()))
        print(f"  Top {top_k}: {top}\n")


if __name__ == "__main__":
    set_seed(SEED)
    # The validation split is seeded, so these are the same 3 images every run.
    _, valid_set, _, _ = get_datasets()
    images, labels = valid_set[:N_IMAGES]

    # Load the trained baseline model saved by train.py.
    trained = MLP(hidden_sizes=(300, 100))
    trained.load_state_dict(torch.load(BASE_DIR / "mlp_fashion_mnist.pt", map_location="cpu"))

    probs = predict_proba(trained, images)
    show_predictions(probs, labels)
