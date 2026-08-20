import torch
import torch.nn as nn


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB) with local dense feature reuse and residual scaling.
    
    Tensor Flow:
    Input x: [B, C, H, W]
    -> Conv1: [B, GC, H, W], concat -> [B, C+GC, H, W]
    -> Conv2: [B, GC, H, W], concat -> [B, C+2*GC, H, W]
    -> Conv3: [B, GC, H, W], concat -> [B, C+3*GC, H, W]
    -> Conv4: [B, GC, H, W], concat -> [B, C+4*GC, H, W]
    -> Conv5 (Compression): [B, C, H, W]
    -> Output: x + res_scale * out [B, C, H, W]
    """
    def __init__(self, num_features=64, growth_rate=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale

        self.conv1 = nn.Conv2d(num_features, growth_rate, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features + growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_features + 2 * growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_features + 3 * growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(num_features + 4 * growth_rate, num_features, kernel_size=3, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))

        return x + self.res_scale * x5


class ResidualInResidualDenseBlock(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB) combining 3 RDBs with outer residual connection.
    
    Tensor Flow:
    Input x: [B, C, H, W]
    -> RDB1 -> RDB2 -> RDB3: [B, C, H, W]
    -> Output: x + res_scale * out [B, C, H, W]
    """
    def __init__(self, num_features=64, growth_rate=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale

        self.rdb1 = ResidualDenseBlock(num_features, growth_rate, res_scale)
        self.rdb2 = ResidualDenseBlock(num_features, growth_rate, res_scale)
        self.rdb3 = ResidualDenseBlock(num_features, growth_rate, res_scale)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + self.res_scale * out


class SpatialEncoder(nn.Module):
    """
    High-capacity, configurable Spatial Feature Encoder for Satellite Imagery Super-Resolution.
    
    Tensor Flow:
    Input LR Image: [B, in_channels, H, W]
    -> Head Conv:   [B, num_features, H, W]
    -> RRDB Body:   [B, num_features, H, W] (across num_blocks RRDBs)
    -> Body Conv:   [B, num_features, H, W] + Head Skip
    -> Output:      [B, num_features, H, W] (Spatial Feature Maps)
    """
    def __init__(self, in_channels=3, num_features=32, num_blocks=2, growth_rate=16, res_scale=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.num_blocks = num_blocks

        # 1. Feature Extraction Head
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # 2. RRDB Trunk
        rrdb_blocks = []
        for _ in range(num_blocks):
            rrdb_blocks.append(ResidualInResidualDenseBlock(num_features=num_features, growth_rate=growth_rate, res_scale=res_scale))
        self.body = nn.Sequential(*rrdb_blocks)

        # 3. Trunk Output Conv
        self.body_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        feat_head = self.head(x)
        feat_body = self.body(feat_head)
        feat_body = self.body_conv(feat_body)

        # Global trunk residual skip
        spatial_features = feat_head + feat_body
        return spatial_features
