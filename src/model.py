"""
model.py
--------
Modified U-Net architecture using a pre-trained ResNet-18 encoder.

Architecture overview
---------------------
Encoder : ResNet-18 (pretrained on ImageNet, final FC + AvgPool removed)
          → learns hierarchical spatial features from satellite imagery.
Decoder : 4× ConvTranspose2d blocks that progressively upsample the feature
          maps back to the original spatial resolution.
Head    : Conv2d(32 → num_classes) + Sigmoid for binary segmentation.

Reference
---------
Ronneberger, O., Fischer, P., & Brox, T. (2015).
U-Net: Convolutional Networks for Biomedical Image Segmentation.
arXiv:1505.04597
"""

import torch
import torch.nn as nn
import torchvision.models as models


class UNet(nn.Module):
    """
    Lightweight U-Net variant with a ResNet-18 backbone encoder.

    Args:
        num_classes (int): Number of output segmentation classes.
                           Use 1 for binary (building / background).
    """

    def __init__(self, num_classes: int = 1):
        super(UNet, self).__init__()

        # ── Encoder ─────────────────────────────────────────────────── #
        # Strip the final classification layers (AvgPool + FC) from
        # ResNet-18 so we retain the spatial feature maps.
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # ── Decoder ─────────────────────────────────────────────────── #
        # Each ConvTranspose2d doubles the spatial resolution (stride=2).
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d( 64,  32, kernel_size=2, stride=2)

        # ── Segmentation Head ────────────────────────────────────────── #
        self.conv_final = nn.Conv2d(32, num_classes, kernel_size=1)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input batch of satellite images [B, 3, H, W]

        Returns:
            Tensor: Predicted probability maps [B, num_classes, H, W]
                    Values in (0, 1) after sigmoid.
        """
        x = self.encoder(x)   # [B, 512, H/32, W/32]
        x = self.upconv1(x)   # [B, 256, H/16, W/16]
        x = self.upconv2(x)   # [B, 128, H/8,  W/8 ]
        x = self.upconv3(x)   # [B, 64,  H/4,  W/4 ]
        x = self.upconv4(x)   # [B, 32,  H/2,  W/2 ]
        x = self.conv_final(x)  # [B, num_classes, H/2, W/2]
        return torch.sigmoid(x)


# ------------------------------------------------------------------ #
# Convenience factory
# ------------------------------------------------------------------ #
def build_model(num_classes: int = 1, device: str = "cpu") -> UNet:
    """Instantiate and move the model to the requested device."""
    model = UNet(num_classes=num_classes).to(device)
    return model
