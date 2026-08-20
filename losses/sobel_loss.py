import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeFilter(nn.Module):
    """
    Differentiable 2D Sobel Edge Extraction Filter.
    
    Computes horizontal (Gx) and vertical (Gy) spatial gradients:
    G_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    G_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    Gradient magnitude: G = sqrt(G_x^2 + G_y^2 + eps)
    """
    def __init__(self, in_channels=3, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps

        kx = torch.tensor([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]], dtype=torch.float32)

        ky = torch.tensor([[-1.0, -2.0, -1.0],
                           [ 0.0,  0.0,  0.0],
                           [ 1.0,  2.0,  1.0]], dtype=torch.float32)

        # Depthwise grouped convolution weights
        self.register_buffer("weight_x", kx.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.register_buffer("weight_y", ky.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))

    def forward(self, x):
        """
        Args:
            x: Input image tensor [B, C, H, W]
        Returns:
            magnitude: Gradient magnitude tensor [B, C, H, W]
            gx: Horizontal gradient tensor [B, C, H, W]
            gy: Vertical gradient tensor [B, C, H, W]
        """
        gx = F.conv2d(x, self.weight_x, padding=1, groups=self.in_channels)
        gy = F.conv2d(x, self.weight_y, padding=1, groups=self.in_channels)

        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + self.eps)
        return magnitude, gx, gy


class SobelEdgeLoss(nn.Module):
    """
    Differentiable Sobel Edge Loss Module for Satellite Imagery Super-Resolution.
    
    Formula:
    L_edge = || Sobel(SR) - Sobel(HR) ||_1
    """
    def __init__(self, in_channels=3, eps=1e-6):
        super().__init__()
        self.filter = SobelEdgeFilter(in_channels=in_channels, eps=eps)
        self.l1 = nn.L1Loss(reduction="mean")

    def forward(self, sr, hr):
        """
        Args:
            sr: Super-Resolution prediction [B, C, H, W]
            hr: Ground-Truth target [B, C, H, W]
        Returns:
            loss: L1 edge magnitude loss
        """
        g_sr, _, _ = self.filter(sr)
        g_hr, _, _ = self.filter(hr)

        return self.l1(g_sr, g_hr)
