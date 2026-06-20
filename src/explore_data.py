"""
explore_data.py
---------------
Objective 2 — Dataset Exploration (OpenCV / NumPy pipeline).

This script replicates the exploratory workflow from the internship report:
  • Loads all .JPG images and .PNG masks into numpy arrays using OpenCV.
  • Prints dataset shape information.
  • Displays random image–mask pairs for visual verification.

Usage
-----
    python src/explore_data.py \
        --image_dir  data/images \
        --mask_dir   data/masks  \
        --num_pairs  4
"""

import os
import glob
import random
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────────────────────
# Loaders
# ────────────────────────────────────────────────────────────────────────────
def load_images(image_dir: str) -> np.ndarray:
    """
    Recursively collects all .JPG files under `image_dir` and loads them
    as colour images (BGR → RGB for matplotlib).

    Returns:
        np.ndarray of shape (N, H, W, 3)
    """
    paths = sorted(glob.glob(os.path.join(image_dir, "**", "*.JPG"),
                             recursive=True))
    if not paths:
        # Also try lowercase extension
        paths = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"),
                                 recursive=True))

    print(f"[INFO] Found {len(paths)} image(s) in '{image_dir}'")
    for p in paths[:3]:
        print(f"       {p}")

    images = [cv2.cvtColor(cv2.imread(p, 1), cv2.COLOR_BGR2RGB) for p in paths]
    return np.array(images), paths


def load_masks(mask_dir: str) -> np.ndarray:
    """
    Recursively collects all .PNG mask files under `mask_dir` and loads
    them in grayscale mode.

    Returns:
        np.ndarray of shape (N, H, W)
    """
    paths = sorted(glob.glob(os.path.join(mask_dir, "**", "*.PNG"),
                             recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(mask_dir, "**", "*.png"),
                                 recursive=True))

    print(f"[INFO] Found {len(paths)} mask(s)  in '{mask_dir}'")
    for p in paths[:3]:
        print(f"       {p}")

    masks = [cv2.imread(p, 0) for p in paths]
    return np.array(masks), paths


# ────────────────────────────────────────────────────────────────────────────
# Visualisation
# ────────────────────────────────────────────────────────────────────────────
def display_random_pairs(images: np.ndarray, masks: np.ndarray,
                         num_pairs: int = 4,
                         save_path: str = "results/sample_pairs.png"):
    """
    Randomly selects `num_pairs` image-mask pairs and plots them
    side-by-side in a grid.
    """
    n         = min(num_pairs, len(images))
    indices   = random.sample(range(len(images)), n)

    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]          # Ensure axes is always a list of rows

    for row, idx in enumerate(indices):
        axes[row][0].imshow(images[idx])
        axes[row][0].set_title(f"Image #{idx}", fontsize=11)
        axes[row][0].axis("off")

        axes[row][1].imshow(masks[idx], cmap="viridis")
        axes[row][1].set_title(f"Mask #{idx}", fontsize=11)
        axes[row][1].axis("off")

    plt.suptitle("Sample Satellite Images and Their Segmentation Masks",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Sample pairs figure saved to {save_path}")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Explore the building detection dataset")
    parser.add_argument("--image_dir", default="data/images", help="Directory with .JPG images")
    parser.add_argument("--mask_dir",  default="data/masks",  help="Directory with .PNG masks")
    parser.add_argument("--num_pairs", type=int, default=4,   help="Number of random pairs to display")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load ──────────────────────────────────────────────────────────── #
    train_images, image_names = load_images(args.image_dir)
    train_masks,  mask_names  = load_masks(args.mask_dir)

    # ── Dataset statistics ────────────────────────────────────────────── #
    print(f"\n{'='*40}")
    print(f"  Number of images found : {len(train_images)}")
    print(f"  Number of masks found  : {len(train_masks)}")
    print(f"  Total samples          : {min(len(train_images), len(train_masks))}")
    print(f"{'='*40}")
    print(f"  train_images.shape     : {train_images.shape}")
    print(f"  train_masks.shape      : {train_masks.shape}")
    print(f"{'='*40}\n")

    # ── Visualise ─────────────────────────────────────────────────────── #
    display_random_pairs(train_images, train_masks, num_pairs=args.num_pairs)


if __name__ == "__main__":
    main()
