"""
train.py
--------
End-to-end training script for the U-Net building detection model.

Steps
-----
1.  Load dataset (images + masks) via CustomDataset.
2.  Split 90 / 10 into train / test sets.
3.  Build U-Net with ResNet-18 encoder.
4.  Train with Binary Cross-Entropy loss + Adam optimiser for N epochs.
5.  Save the trained weights to models/unet_building.pth.
6.  Plot and save the training-loss curve.

Usage
-----
    python src/train.py \
        --image_dir  data/images \
        --mask_dir   data/masks  \
        --epochs     25          \
        --batch_size 16          \
        --lr         1e-4        \
        --img_size   128
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import CustomDataset, get_transform
from model   import build_model


# ────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────
def train_model(model, criterion, optimizer, train_loader,
                num_epochs: int = 25, device: str = "cpu"):
    """
    Trains the model for `num_epochs` and returns per-epoch loss values.

    Args:
        model        : PyTorch model to train.
        criterion    : Loss function.
        optimizer    : Optimiser.
        train_loader : DataLoader for training data.
        num_epochs   : Number of full passes over the training set.
        device       : "cuda" or "cpu".

    Returns:
        train_losses (list[float]): Average loss per epoch.
    """
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, masks in train_loader:
            inputs = inputs.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            # Resize masks to match model output spatial size
            masks = torch.nn.functional.interpolate(
                masks, size=outputs.shape[2:], mode='nearest'
            )

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}")

    return train_losses


# ────────────────────────────────────────────────────────────────────────────
# Plotting helper
# ────────────────────────────────────────────────────────────────────────────
def plot_loss(train_losses, save_path: str = "results/training_loss_curve.png"):
    """Saves the training-loss curve to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Loss curve saved to {save_path}")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for building detection")
    parser.add_argument("--image_dir",  default="data/images",  help="Path to satellite images")
    parser.add_argument("--mask_dir",   default="data/masks",   help="Path to segmentation masks")
    parser.add_argument("--epochs",     type=int, default=25,   help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size",   type=int, default=128,  help="Resize images to this size (square)")
    parser.add_argument("--model_out",  default="models/unet_building.pth", help="Where to save trained weights")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────── #
    transform = get_transform(args.img_size)
    dataset   = CustomDataset(args.image_dir, args.mask_dir, transform=transform)

    print(f"[INFO] Total samples in dataset: {len(dataset)}")

    train_set, _ = train_test_split(dataset, test_size=0.1, random_state=42)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # ── Model ─────────────────────────────────────────────────────────── #
    model     = build_model(num_classes=1, device=device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ── Train ─────────────────────────────────────────────────────────── #
    train_losses = train_model(
        model, criterion, optimizer, train_loader,
        num_epochs=args.epochs, device=device
    )

    # ── Save weights ──────────────────────────────────────────────────── #
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    print(f"[INFO] Model weights saved to {args.model_out}")

    # ── Plot loss ─────────────────────────────────────────────────────── #
    plot_loss(train_losses)


if __name__ == "__main__":
    main()
