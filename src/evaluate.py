"""
evaluate.py
-----------
Evaluates the trained U-Net on the held-out test set.

Metrics computed
----------------
* Pixel-level accuracy  (sklearn accuracy_score)
* Confusion matrix      (visualised as a seaborn heatmap)
* Precision / Recall    (sklearn classification_report)

Usage
-----
    python src/evaluate.py \
        --image_dir  data/images \
        --mask_dir   data/masks  \
        --model_path models/unet_building.pth
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report
)

from dataset import CustomDataset, get_transform
from model   import build_model


# ────────────────────────────────────────────────────────────────────────────
# Inference over the test set
# ────────────────────────────────────────────────────────────────────────────
def test_model(model, test_loader, device: str = "cpu", threshold: float = 0.5):
    """
    Runs the model on every batch in `test_loader` and returns flat arrays
    of predictions and ground-truth labels.

    Args:
        model       : Trained PyTorch model.
        test_loader : DataLoader for the test split.
        device      : "cuda" or "cpu".
        threshold   : Probability cutoff for binary classification.

    Returns:
        all_preds (np.ndarray): Predicted binary labels, flattened.
        all_masks (np.ndarray): Ground-truth binary labels, flattened.
    """
    model.eval()
    all_preds, all_masks = [], []

    with torch.no_grad():
        for inputs, masks in test_loader:
            inputs = inputs.to(device)
            masks  = masks.to(device)

            outputs = model(inputs)

            # Align mask size to model output
            masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')

            preds = (outputs > threshold).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_masks.extend(masks.cpu().numpy().flatten())

    return np.array(all_preds), np.array(all_masks)


# ────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ────────────────────────────────────────────────────────────────────────────
def save_confusion_matrix(all_masks, all_preds,
                          save_path: str = "results/confusion_matrix.png"):
    """Plots and saves the confusion matrix heatmap."""
    # Binarise (threshold=0.5 already applied, but masks may be float)
    preds_int = (all_preds > 0.5).astype(int)
    masks_int = (all_masks > 0.5).astype(int)

    cm = confusion_matrix(masks_int, preds_int)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Background', 'Building'],
                yticklabels=['Background', 'Building'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained U-Net")
    parser.add_argument("--image_dir",  default="data/images")
    parser.add_argument("--mask_dir",   default="data/masks")
    parser.add_argument("--model_path", default="models/unet_building.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size",   type=int, default=128)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Data ──────────────────────────────────────────────────────────── #
    transform = get_transform(args.img_size)
    dataset   = CustomDataset(args.image_dir, args.mask_dir, transform=transform)
    _, test_set = train_test_split(dataset, test_size=0.1, random_state=42)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # ── Load model ────────────────────────────────────────────────────── #
    model = build_model(num_classes=1, device=device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"[INFO] Loaded weights from {args.model_path}")

    # ── Evaluate ──────────────────────────────────────────────────────── #
    preds, masks = test_model(model, test_loader, device=device)

    preds_bin = (preds > 0.5).astype(int)
    masks_bin = (masks > 0.5).astype(int)

    acc = accuracy_score(masks_bin, preds_bin)
    print(f"\n[RESULT] Pixel Accuracy : {acc:.4f}")
    print("\n[RESULT] Classification Report:")
    print(classification_report(masks_bin, preds_bin,
                                target_names=['Background', 'Building']))

    save_confusion_matrix(masks, preds)


if __name__ == "__main__":
    main()
