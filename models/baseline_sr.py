import torch
import torch.nn as nn
import torch.nn.functional as F


class BicubicBaseline(nn.Module):
    """
    Pure Bicubic Interpolation Baseline module.
    """
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, C, H, W]
        Returns:
            Upscaled Tensor of shape [B, C, H*scale, W*scale]
        """
        out = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return torch.clamp(out, 0.0, 1.0)


class SimpleSpatialSR(nn.Module):
    """
    Lightweight Spatial CNN/Residual Super-Resolution Baseline Model.
    
    Architecture:
    Input LR -> Head Conv -> 2x Residual Blocks -> Pixel Shuffle Upsampler -> Tail Conv + Global Bicubic Residual -> SR Output
    """
    def __init__(self, scale=4, in_channels=3, out_channels=3, num_features=32):
        super().__init__()
        self.scale = scale
        self.bicubic = BicubicBaseline(scale=scale)

        # 1. Feature Extraction Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 2. Residual Blocks
        self.res1 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

        # 3. Upsampling (PixelShuffle)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True)
        )

        # 4. Reconstruction Tail
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        bicubic_base = self.bicubic(x)

        feat = self.head(x)
        feat = feat + self.res1(feat)
        feat = feat + self.res2(feat)

        up = self.upsample(feat)
        residual = self.tail(up)

        sr = bicubic_base + residual
        return torch.clamp(sr, 0.0, 1.0)
