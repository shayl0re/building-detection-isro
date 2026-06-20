"""
inference.py
------------
Run the trained U-Net on a single new satellite image and display
the original image alongside its predicted building mask.

Usage
-----
    python src/inference.py \
        --image_path  path/to/your_image.JPG \
        --model_path  models/unet_building.pth \
        --output_path results/prediction.png
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms as transforms

from model import build_model


# ────────────────────────────────────────────────────────────────────────────
def inference(model, image_path: str, transform, device: str = "cpu",
              threshold: float = 0.5):
    """
    Runs the model on a single image and returns the original PIL image
    and the binary predicted mask.

    Args:
        model      : Trained PyTorch model.
        image_path : Path to the .JPG / .PNG input image.
        transform  : torchvision transform to pre-process the image.
        device     : "cuda" or "cpu".
        threshold  : Probability cutoff for binary mask.

    Returns:
        original_image (PIL.Image): The original input image.
        pred_mask      (np.ndarray): Binary predicted mask [H, W].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    model.eval()

    image             = Image.open(image_path).convert("RGB")
    image_transformed = transform(image).unsqueeze(0).to(device)  # [1, C, H, W]

    with torch.no_grad():
        output = model(image_transformed)

    pred_mask = (output > threshold).float().cpu().squeeze().numpy()  # [H, W]
    return image, pred_mask


# ────────────────────────────────────────────────────────────────────────────
def save_prediction(original_image, pred_mask, save_path: str):
    """Saves a side-by-side visualisation of the image and its mask."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image)
    axes[0].set_title("Input Satellite Image")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="gray")
    axes[1].set_title("Predicted Building Mask")
    axes[1].axis("off")

    plt.suptitle("Automatic Building Detection — U-Net", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Prediction saved to {save_path}")


# ────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Run U-Net inference on a single image")
    parser.add_argument("--image_path",  required=True,  help="Path to input satellite image")
    parser.add_argument("--model_path",  default="models/unet_building.pth")
    parser.add_argument("--output_path", default="results/prediction.png")
    parser.add_argument("--img_size",    type=int, default=128)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    model = build_model(num_classes=1, device=device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"[INFO] Model loaded from {args.model_path}")

    original_image, pred_mask = inference(
        model, args.image_path, transform, device
    )

    save_prediction(original_image, pred_mask, args.output_path)


if __name__ == "__main__":
    main()
