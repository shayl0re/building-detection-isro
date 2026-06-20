"""
dataset.py
----------
Custom PyTorch Dataset for loading satellite images and their segmentation masks.
Used in Objective 1 of the building detection pipeline.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    """
    Loads paired satellite images (.JPG) and binary masks (.PNG)
    from two separate directories.

    Args:
        image_dir  (str): Path to the folder containing .JPG images.
        mask_dir   (str): Path to the folder containing .PNG masks.
        transform  (callable, optional): torchvision transform applied to
                   both images and masks.
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir

        # Collect & sort files so images[i] matches masks[i]
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith('.JPG') or f.endswith('.PNG')
        ])
        self.mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.endswith('.JPG') or f.endswith('.PNG')
        ])

        self.transform = transform

    # ------------------------------------------------------------------ #
    def __len__(self):
        """Returns the total number of image-mask pairs."""
        return len(self.image_files)

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Transformed satellite image   [C, H, W]
            mask  (Tensor): Transformed segmentation mask [1, H, W]
        """
        img_path  = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir,  self.mask_files[idx])

        # Open image as RGB, mask as grayscale (L)
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask  = self.transform(mask)

        return image, mask


# ------------------------------------------------------------------ #
# Default transform: resize to 128×128 and convert to tensor
# ------------------------------------------------------------------ #
def get_transform(img_size: int = 128) -> transforms.Compose:
    """Returns the standard transform pipeline used in training."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
