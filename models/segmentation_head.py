import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class LightweightSegmentationUNet(nn.Module):
    """
    Lightweight 4-level UNet for Building and Road Binary Segmentation in Satellite Imagery.
    
    Tensor Flow:
    Input Image [B, 3, H, W] -> Logit Mask Prediction [B, 1, H, W]
    """
    def __init__(self, in_channels=3, num_classes=1, num_features=16):
        super().__init__()
        nf = num_features

        # Encoder
        self.enc1 = ConvBlock(in_channels, nf)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(nf, nf * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(nf * 2, nf * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(nf * 4, nf * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(nf * 4, nf * 2)
        self.up1 = nn.ConvTranspose2d(nf * 2, nf, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(nf * 2, nf)

        # Head
        self.head = nn.Conv2d(nf, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input satellite image tensor [B, 3, H, W]
        Returns:
            logits: Segmentation prediction logits [B, 1, H, W]
        """
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        logits = self.head(d1)
        return logits
