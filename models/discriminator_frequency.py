import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from .frequency_encoder import DWT2D


class FrequencyPatchGANDiscriminator(nn.Module):
    """
    Frequency Domain PatchGAN Discriminator for Satellite Imagery Super-Resolution.
    
    Discriminates real vs fake wavelet sub-band representations (LL, LH, HL, HH concatenated: 12 channels for RGB),
    forcing the generator to synthesize natural spectral power distributions across high frequencies.
    
    Tensor Flow:
    Input Image [B, 3, H, W] -> DWT2D -> [B, 12, H/2, W/2]
    -> Conv1 (12 -> 64, stride 2):  [B, 64, H/4, W/4]
    -> Conv2 (64 -> 128, stride 2): [B, 128, H/8, W/8]
    -> Conv3 (128 -> 256, stride 1):[B, 256, H/8 - 1, W/8 - 1]
    -> ConvOut (256 -> 1, stride 1): Patch Validity Map [B, 1, H_p, W_p]
    """
    def __init__(self, in_channels=3, num_features=64, num_layers=3, use_spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.dwt = DWT2D(in_channels=in_channels)
        dwt_channels = in_channels * 4  # 3 * 4 = 12 channels for Haar wavelet sub-bands

        def norm_conv(conv):
            return spectral_norm(conv) if use_spectral_norm else conv

        layers = []
        # Layer 1: No normalization on input DWT features
        layers.append(nn.Conv2d(dwt_channels, num_features, kernel_size=4, stride=2, padding=1))
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

        # Output Layer
        layers.append(norm_conv(nn.Conv2d(
            num_features * nf_mult,
            1,
            kernel_size=4,
            stride=1,
            padding=1
        )))

        self.model = nn.Sequential(*layers)

    def extract_dwt_subbands(self, x):
        """
        Converts image tensor [B, 3, H, W] into 12-channel DWT subband tensor [B, 12, H/2, W/2]
        """
        ll, lh, hl, hh = self.dwt(x)
        return torch.cat([ll, lh, hl, hh], dim=1)

    def forward(self, x):
        """
        Args:
            x: Image tensor [B, 3, H, W] or concatenated DWT tensor [B, 12, H/2, W/2]
        Returns:
            validity: Frequency patch validity map [B, 1, H_patch, W_patch]
        """
        if x.shape[1] == self.in_channels:
            x_dwt = self.extract_dwt_subbands(x)
        else:
            x_dwt = x

        return self.model(x_dwt)
