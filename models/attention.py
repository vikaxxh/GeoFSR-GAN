import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightCrossAttention(nn.Module):
    """
    Lightweight CPU-Friendly Cross-Attention Module.
    
    Attends Query features (e.g. Spatial) to Key/Value features (e.g. Frequency)
    with spatial dimension reduction (r=2) for O(N * N/4) CPU computational efficiency.
    
    Tensor Flow:
    q_feat  [B, C, H, W] -> Q [B, H*W, C]
    kv_feat [B, C, H, W] -> Spatial Reduction (r=2) -> K [B, C, H_r*W_r], V [B, H_r*W_r, C]
    Attention Map = Softmax(Q * K / sqrt(C)) -> [B, H*W, H_r*W_r]
    Out = Attention Map * V -> Reshape [B, C, H, W] -> Conv Out + Residual Skip
    """
    def __init__(self, num_features=32, reduction_factor=2):
        super().__init__()
        self.num_features = num_features
        self.reduction_factor = reduction_factor

        self.query_conv = nn.Conv2d(num_features, num_features, kernel_size=1)
        self.key_conv = nn.Conv2d(num_features, num_features, kernel_size=1)
        self.value_conv = nn.Conv2d(num_features, num_features, kernel_size=1)

        if reduction_factor > 1:
            self.pool = nn.AdaptiveAvgPool2d((None, None))  # Dynamic pooling in forward

        self.out_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.scale = 1.0 / math.sqrt(num_features)

    def forward(self, q_feat, kv_feat):
        b, c, h, w = q_feat.shape

        # 1. Project Query
        q = self.query_conv(q_feat).view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

        # 2. Project & Spatially Reduce Key / Value for CPU Efficiency
        if self.reduction_factor > 1:
            h_r, w_r = max(h // self.reduction_factor, 4), max(w // self.reduction_factor, 4)
            kv_reduced = F.adaptive_avg_pool2d(kv_feat, (h_r, w_r))
        else:
            kv_reduced = kv_feat

        k = self.key_conv(kv_reduced).view(b, c, -1)                   # [B, C, H_r*W_r]
        v = self.value_conv(kv_reduced).view(b, c, -1).permute(0, 2, 1) # [B, H_r*W_r, C]

        # 3. Scaled Dot-Product Attention
        attn = torch.bmm(q, k) * self.scale                             # [B, H*W, H_r*W_r]
        attn_weights = F.softmax(attn, dim=-1)

        # 4. Aggregate Value Features
        out = torch.bmm(attn_weights, v)                                # [B, H*W, C]
        out = out.permute(0, 2, 1).view(b, c, h, w)                    # [B, C, H, W]

        # 5. Output Conv + Residual Connection
        return q_feat + self.out_conv(out)


class DualCrossAttention(nn.Module):
    """
    Bidirectional Spatial-Frequency Cross-Attention Module.
    
    Direction 1: Spatial Queries attending to Frequency Keys/Values
    Direction 2: Frequency Queries attending to Spatial Keys/Values
    """
    def __init__(self, num_features=32, reduction_factor=2):
        super().__init__()
        self.cross_sf = LightweightCrossAttention(num_features=num_features, reduction_factor=reduction_factor)
        self.cross_fs = LightweightCrossAttention(num_features=num_features, reduction_factor=reduction_factor)

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

    def forward(self, f_spatial, f_freq):
        out_sf = self.cross_sf(q_feat=f_spatial, kv_feat=f_freq)
        out_fs = self.cross_fs(q_feat=f_freq, kv_feat=f_spatial)

        cat_feat = torch.cat([out_sf, out_fs], dim=1)
        return self.fuse_conv(cat_feat)
