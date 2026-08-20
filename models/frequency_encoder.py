import torch
import torch.nn as nn
import torch.nn.functional as F


class DWT2D(nn.Module):
    """
    Differentiable 2D Discrete Wavelet Transform (DWT) using Haar Wavelets.
    
    Decomposes input tensor [B, C, H, W] into 4 sub-bands:
    - LL: Low-frequency spatial approximation [B, C, H/2, W/2]
    - LH: Horizontal high-frequency details [B, C, H/2, W/2]
    - HL: Vertical high-frequency details [B, C, H/2, W/2]
    - HH: Diagonal high-frequency details [B, C, H/2, W/2]
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        # Haar Wavelet Filters (Normalized)
        dwt_kernel = torch.tensor([
            [[[0.5, 0.5], [0.5, 0.5]]],    # LL
            [[[0.5, 0.5], [-0.5, -0.5]]],  # LH
            [[[0.5, -0.5], [0.5, -0.5]]],  # HL
            [[[0.5, -0.5], [-0.5, 0.5]]]   # HH
        ], dtype=torch.float32)

        # Repeat weights across input channels for depthwise grouped convolution
        self.register_buffer("weight", dwt_kernel.repeat(in_channels, 1, 1, 1))

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, f"Image dimensions ({h}x{w}) must be even for DWT."

        # Depthwise grouped convolution
        out = F.conv2d(x, self.weight, stride=2, groups=channels)
        out = out.view(batch_size, channels, 4, h // 2, w // 2)

        ll = out[:, :, 0, :, :]
        lh = out[:, :, 1, :, :]
        hl = out[:, :, 2, :, :]
        hh = out[:, :, 3, :, :]

        return ll, lh, hl, hh


class IDWT2D(nn.Module):
    """
    Differentiable 2D Inverse Discrete Wavelet Transform (IDWT) using Haar Wavelets.
    
    Reconstructs original image [B, C, H, W] from (LL, LH, HL, HH).
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        # Inverse Haar Wavelet Filters
        idwt_kernel = torch.tensor([
            [[[0.5, 0.5], [0.5, 0.5]]],    # LL
            [[[0.5, 0.5], [-0.5, -0.5]]],  # LH
            [[[0.5, -0.5], [0.5, -0.5]]],  # HL
            [[[0.5, -0.5], [-0.5, 0.5]]]   # HH
        ], dtype=torch.float32)

        self.register_buffer("weight", idwt_kernel.repeat(in_channels, 1, 1, 1))

    def forward(self, ll, lh, hl, hh):
        batch_size, channels, h_sub, w_sub = ll.shape
        stacked = torch.stack([ll, lh, hl, hh], dim=2)
        stacked = stacked.view(batch_size, channels * 4, h_sub, w_sub)

        # Transposed convolution for exact reconstruction
        rec = F.conv_transpose2d(stacked, self.weight, stride=2, groups=channels)
        return rec


class FrequencyEncoder(nn.Module):
    """
    Multi-Band Frequency Encoder for Satellite Imagery Super-Resolution.
    
    Explicitly processes LL, LH, HL, HH sub-bands and generates frequency feature representation.
    Output shape: [B, num_features, H, W] matching SpatialEncoder shape for fusion.
    """
    def __init__(self, in_channels=3, num_features=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        sub_feat = max(num_features // 4, 8)

        self.dwt = DWT2D(in_channels=in_channels)

        # Dedicated Sub-Band Convolution Processors
        self.conv_ll = nn.Sequential(
            nn.Conv2d(in_channels, sub_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_lh = nn.Sequential(
            nn.Conv2d(in_channels, sub_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_hl = nn.Sequential(
            nn.Conv2d(in_channels, sub_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_hh = nn.Sequential(
            nn.Conv2d(in_channels, sub_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Sub-band Fusion & Spatial Restoration
        total_sub_feat = sub_feat * 4
        self.fuse = nn.Sequential(
            nn.Conv2d(total_sub_feat, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        Args:
            x: Input LR image tensor [B, in_channels, H, W]
        Returns:
            freq_features: Frequency feature map [B, num_features, H, W]
        """
        # 1. 2D Discrete Wavelet Decomposition
        ll, lh, hl, hh = self.dwt(x)

        # 2. Process Sub-Bands Individually
        feat_ll = self.conv_ll(ll)
        feat_lh = self.conv_lh(lh)
        feat_hl = self.conv_hl(hl)
        feat_hh = self.conv_hh(hh)

        # 3. Concatenate Sub-Band Feature Maps [B, sub_feat * 4, H/2, W/2]
        cat_sub = torch.cat([feat_ll, feat_lh, feat_hl, feat_hh], dim=1)
        feat_fused = self.fuse(cat_sub)

        # 4. Upsample back to spatial LR resolution [B, num_features, H, W]
        freq_features = F.interpolate(feat_fused, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
        return freq_features
