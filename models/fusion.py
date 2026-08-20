import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import LightweightCrossAttention, DualCrossAttention


class ConcatFusion(nn.Module):
    """
    Concatenation + Convolution Spatial-Frequency Fusion Module.
    """
    def __init__(self, num_features=32):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

    def forward(self, f_spatial, f_freq):
        cat_feat = torch.cat([f_spatial, f_freq], dim=1)
        return self.fuse(cat_feat)


class LearnedWeightFusion(nn.Module):
    """
    Learned Channel Gating Weight Fusion Module.
    """
    def __init__(self, num_features=32):
        super().__init__()
        self.gate_s = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=1),
            nn.Sigmoid()
        )
        self.gate_f = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=1),
            nn.Sigmoid()
        )
        self.refine = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, f_spatial, f_freq):
        w_s = self.gate_s(f_spatial)
        w_f = self.gate_f(f_freq)

        gated_feat = w_s * f_spatial + w_f * f_freq
        return self.refine(gated_feat)


class AttentionFusion(nn.Module):
    """
    Channel-Attention (Squeeze & Excitation) Adaptive Fusion Module.
    """
    def __init__(self, num_features=32):
        super().__init__()
        self.num_features = num_features
        reduced_dim = max(num_features // 2, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(num_features * 2, reduced_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, num_features * 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.refine = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, f_spatial, f_freq):
        cat_feat = torch.cat([f_spatial, f_freq], dim=1)
        attn = self.mlp(self.pool(cat_feat))

        a_s, a_f = torch.split(attn, self.num_features, dim=1)
        fused = a_s * f_spatial + a_f * f_freq
        return self.refine(fused)


class SpatialFrequencyFusion(nn.Module):
    """
    Unified, configurable Spatial-Frequency Fusion wrapper supporting:
    - 'concat': Concatenation + Conv
    - 'learned': Learned channel weights
    - 'attention': Adaptive channel attention
    - 'cross_attention': Spatial-to-Frequency Cross-Attention
    - 'dual_cross_attention': Bidirectional Spatial-Frequency Cross-Attention
    """
    def __init__(self, num_features=32, fusion_type="concat"):
        super().__init__()
        self.fusion_type = fusion_type.lower()

        if self.fusion_type == "concat":
            self.fusion = ConcatFusion(num_features=num_features)
        elif self.fusion_type == "learned":
            self.fusion = LearnedWeightFusion(num_features=num_features)
        elif self.fusion_type == "attention":
            self.fusion = AttentionFusion(num_features=num_features)
        elif self.fusion_type == "cross_attention":
            self.fusion = LightweightCrossAttention(num_features=num_features, reduction_factor=2)
        elif self.fusion_type == "dual_cross_attention":
            self.fusion = DualCrossAttention(num_features=num_features, reduction_factor=2)
        else:
            raise ValueError(
                f"Unsupported fusion_type: {fusion_type}. Choose from "
                f"'concat', 'learned', 'attention', 'cross_attention', 'dual_cross_attention'."
            )

    def forward(self, f_spatial, f_freq):
        return self.fusion(f_spatial, f_freq)
