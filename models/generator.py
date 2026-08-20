import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline_sr import BicubicBaseline
from .spatial_encoder import SpatialEncoder
from .frequency_encoder import FrequencyEncoder
from .fusion import SpatialFrequencyFusion


class GeoFSRGenerator(nn.Module):
    """
    GeoFSR-GAN Integrated Generator Architecture (Milestone 5 Progressive Build).
    
    Tensor Flow:
    Input LR: [B, 3, H, W]
    -> Spatial Encoder:   F_spatial  [B, num_features, H, W]
    -> Frequency Encoder: F_freq     [B, num_features, H, W]
    -> Fusion (Concat/Learned/Attention): F_fused [B, num_features, H, W]
    -> Reconstruction Blocks: F_rec  [B, num_features, H, W]
    -> PixelShuffle Upsampler: [B, num_features, H*scale, W*scale]
    -> Tail Conv + Bicubic Global Residual: SR Output [B, 3, H*scale, W*scale]
    """
    def __init__(
        self,
        scale=4,
        in_channels=3,
        out_channels=3,
        num_features=32,
        num_spatial_blocks=2,
        fusion_type="concat"
    ):
        super().__init__()
        self.scale = scale
        self.bicubic = BicubicBaseline(scale=scale)

        # 1. Dual Feature Encoders
        self.spatial_encoder = SpatialEncoder(
            in_channels=in_channels,
            num_features=num_features,
            num_blocks=num_spatial_blocks,
            growth_rate=16
        )
        self.frequency_encoder = FrequencyEncoder(
            in_channels=in_channels,
            num_features=num_features
        )

        # 2. Spatial-Frequency Fusion Module
        self.fusion = SpatialFrequencyFusion(
            num_features=num_features,
            fusion_type=fusion_type
        )

        # 3. Feature Reconstruction Trunk
        self.reconstruction = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 4. Upsampling Sub-Pixel Convolution (PixelShuffle)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 5. Output Reconstruction Tail
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, lr):
        bicubic_base = self.bicubic(lr)

        f_spatial = self.spatial_encoder(lr)
        f_freq = self.frequency_encoder(lr)

        f_fused = self.fusion(f_spatial, f_freq)
        f_rec = f_fused + self.reconstruction(f_fused)

        up = self.upsample(f_rec)
        residual = self.tail(up)

        sr = bicubic_base + residual
        return torch.clamp(sr, 0.0, 1.0)
