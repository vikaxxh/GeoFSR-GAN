import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class SpatialPatchGANDiscriminator(nn.Module):
    """
    Spatial PatchGAN Discriminator with Spectral Normalization for Satellite Imagery.
    
    Discriminates local image patches (e.g. 30x30 receptive field) rather than full image,
    enforcing sharp texture synthesis and fine boundary features.
    
    Tensor Flow:
    Input HR/SR: [B, 3, H, W]
    -> Conv1 (stride 2): [B, num_features, H/2, W/2]
    -> Conv2 (stride 2): [B, num_features*2, H/4, W/4]
    -> Conv3 (stride 2): [B, num_features*4, H/8, W/8]
    -> Conv4 (stride 1): [B, num_features*8, H/8 - 1, W/8 - 1]
    -> ConvOut (stride 1): Patch Validity Map [B, 1, H_patch, W_patch]
    """
    def __init__(self, in_channels=3, num_features=64, num_layers=3, use_spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features

        def norm_conv(conv):
            return spectral_norm(conv) if use_spectral_norm else conv

        layers = []
        # Layer 1: No normalization on input
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate PatchGAN blocks
        nf_mult = 1
        for i in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8)
            stride = 2 if i < num_layers - 1 else 1
            layers.append(norm_conv(nn.Conv2d(
                num_features * nf_mult_prev,
                num_features * nf_mult,
                kernel_size=4,
                stride=stride,
                padding=1
            )))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Output Layer: 1-channel validity patch prediction map
        layers.append(norm_conv(nn.Conv2d(
            num_features * nf_mult,
            1,
            kernel_size=4,
            stride=1,
            padding=1
        )))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Real HR or Generated SR image tensor [B, 3, H, W]
        Returns:
            validity: Patch validity score map [B, 1, H_patch, W_patch]
        """
        return self.model(x)
