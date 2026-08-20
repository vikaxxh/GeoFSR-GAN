import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from models.spatial_encoder import SpatialEncoder
from models.frequency_encoder import FrequencyEncoder
from models.fusion import ConcatFusion


class NearestBaseline(nn.Module):
    """
    Pure Nearest-Neighbor Interpolation Baseline module.
    """
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return torch.clamp(out, 0.0, 1.0)


class BicubicBaseline(nn.Module):
    """
    Pure Bicubic Interpolation Baseline module.
    """
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return torch.clamp(out, 0.0, 1.0)


class LanczosBaseline(nn.Module):
    """
    Pure Lanczos (Resampling Kernel) Interpolation Baseline module.
    """
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        b, c, h, w = x.shape
        target_size = (w * self.scale, h * self.scale)
        out_tensors = []
        for i in range(b):
            img_pil = TF.to_pil_image(x[i].cpu())
            img_lanczos = img_pil.resize(target_size, resample=Image.LANCZOS)
            out_tensors.append(TF.to_tensor(img_lanczos))
        out = torch.stack(out_tensors, dim=0).to(x.device)
        return torch.clamp(out, 0.0, 1.0)


class SimpleSpatialSR(nn.Module):
    """
    Lightweight Spatial CNN/Residual Super-Resolution Baseline Model.
    """
    def __init__(self, scale=4, in_channels=3, out_channels=3, num_features=32):
        super().__init__()
        self.scale = scale
        self.bicubic = BicubicBaseline(scale=scale)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

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

        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True)
        )

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


class ImprovedSpatialSR(nn.Module):
    """
    High-Performance Improved Spatial SR Baseline using RRDB, Residual Scaling (0.2), 
    LeakyReLU (0.2), 64 Features, and Multi-Stage PixelShuffle.
    """
    def __init__(self, scale=4, in_channels=3, out_channels=3, num_features=64, num_blocks=3, growth_rate=32, res_scale=0.2):
        super().__init__()
        self.scale = scale
        self.bicubic = BicubicBaseline(scale=scale)

        self.encoder = SpatialEncoder(
            in_channels=in_channels,
            num_features=num_features,
            num_blocks=num_blocks,
            growth_rate=growth_rate,
            res_scale=res_scale
        )

        self.upsample1 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features // 2, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        bicubic_base = self.bicubic(x)

        feat = self.encoder(x)
        up = self.upsample1(feat)
        up = self.upsample2(up)

        residual = self.tail(up)
        sr = bicubic_base + residual

        return torch.clamp(sr, 0.0, 1.0)


class SpatialFrequencySR(nn.Module):
    """
    Dual-Domain (Spatial + DWT Frequency) Super-Resolution Model.
    Combines SpatialEncoder and DWT FrequencyEncoder via ConcatFusion.
    """
    def __init__(self, scale=4, in_channels=3, out_channels=3, num_features=64, num_blocks=3, growth_rate=32, res_scale=0.2):
        super().__init__()
        self.scale = scale
        self.bicubic = BicubicBaseline(scale=scale)

        self.spatial_encoder = SpatialEncoder(
            in_channels=in_channels,
            num_features=num_features,
            num_blocks=num_blocks,
            growth_rate=growth_rate,
            res_scale=res_scale
        )
        self.frequency_encoder = FrequencyEncoder(
            in_channels=in_channels,
            num_features=num_features
        )
        self.fusion = ConcatFusion(num_features=num_features)

        self.upsample1 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features // 2, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        bicubic_base = self.bicubic(x)

        feat_spat = self.spatial_encoder(x)
        feat_freq = self.frequency_encoder(x)
        feat_fused = self.fusion(feat_spat, feat_freq)

        up = self.upsample1(feat_fused)
        up = self.upsample2(up)

        residual = self.tail(up)
        sr = bicubic_base + residual
        return torch.clamp(sr, 0.0, 1.0)
