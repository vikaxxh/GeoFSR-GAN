import logging
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ImageNetNormalize(nn.Module):
    """
    Normalizes RGB image tensors [B, 3, H, W] in [0, 1] using standard ImageNet mean and std.
    """
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std


class LightweightPerceptualExtractor(nn.Module):
    """
    Lightweight, CPU-friendly multi-scale convolutional feature extractor.
    
    Used for CPU debug mode or offline environments without requiring external model weight downloads.
    Extracts features at 3 resolution scales:
    - Scale 1 (Full Res): 16 channels
    - Scale 2 (1/2 Res):  32 channels
    - Scale 3 (1/4 Res):  64 channels
    """
    def __init__(self, in_channels=3):
        super().__init__()
        torch.manual_seed(42)
        
        self.slice1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.slice2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.slice3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        return [f1, f2, f3]


class VGG19PerceptualExtractor(nn.Module):
    """
    Pre-trained VGG19 multi-layer feature extractor for production perceptual loss.
    
    Extracts intermediate activations from:
    - conv1_2 (layer index 3)
    - conv2_2 (layer index 8)
    - conv3_4 (layer index 17)
    - conv4_4 (layer index 26)
    - conv5_4 (layer index 35)
    """
    def __init__(self, timeout_sec=2.0):
        super().__init__()
        import torchvision.models as models
        
        # Set socket timeout to prevent long hangs when offline
        orig_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout_sec)
        try:
            vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        except Exception:
            vgg19 = models.vgg19(pretrained=True).features
        finally:
            socket.setdefaulttimeout(orig_timeout)

        self.slice1 = vgg19[:4]    # up to conv1_2
        self.slice2 = vgg19[4:9]   # up to conv2_2
        self.slice3 = vgg19[9:18]  # up to conv3_4
        self.slice4 = vgg19[18:27] # up to conv4_4
        self.slice5 = vgg19[27:36] # up to conv5_4

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        f4 = self.slice4(f3)
        f5 = self.slice5(f4)
        return [f1, f2, f3, f4, f5]


class PerceptualLoss(nn.Module):
    """
    Configurable Perceptual Loss Module supporting:
    - 'lightweight': Fast CPU debug feature extractor (0 download dependency)
    - 'vgg19': Production VGG19 feature extractor
    - 'auto': Attempts VGG19 with timeout, falling back gracefully to lightweight
    """
    def __init__(self, mode="lightweight", layer_weights=None):
        super().__init__()
        self.mode = mode.lower()
        self.norm = ImageNetNormalize()

        if self.mode in ("vgg19", "auto"):
            try:
                self.extractor = VGG19PerceptualExtractor(timeout_sec=2.0)
                self.weights = layer_weights or [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
            except Exception as e:
                logger.info(f"[PerceptualLoss] VGG19 weights unavailable ({e}). Using 'lightweight' feature extractor.")
                self.extractor = LightweightPerceptualExtractor()
                self.weights = layer_weights or [1.0, 1.0, 1.0]
        else:  # 'lightweight'
            self.extractor = LightweightPerceptualExtractor()
            self.weights = layer_weights or [1.0, 1.0, 1.0]

        self.l1 = nn.L1Loss(reduction="mean")

    def forward(self, sr, hr):
        """
        Args:
            sr: Generated Super-Resolution image [B, 3, H, W] in [0, 1]
            hr: Ground-Truth target image [B, 3, H, W] in [0, 1]
        Returns:
            loss: Scaled multi-layer perceptual L1 loss
        """
        sr_norm = self.norm(sr)
        hr_norm = self.norm(hr)

        feat_sr = self.extractor(sr_norm)
        feat_hr = self.extractor(hr_norm)

        loss = sum(w * self.l1(f_s, f_h) for w, f_s, f_h in zip(self.weights, feat_sr, feat_hr))
        return loss
