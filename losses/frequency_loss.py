import torch
import torch.nn as nn
from models.frequency_encoder import DWT2D


class MultiBandFrequencyLoss(nn.Module):
    """
    Multi-Band Wavelet Frequency Loss Module for Satellite Imagery Super-Resolution.
    
    Decomposes generated Super-Resolution (SR) and Ground-Truth (HR) images into 
    DWT wavelet sub-bands (LL, LH, HL, HH) and applies weighted L1 losses with high-frequency emphasis.
    
    Formula:
    L_freq = w_LL * ||DWT(SR)_LL - DWT(HR)_LL||_1 +
             w_LH * ||DWT(SR)_LH - DWT(HR)_LH||_1 +
             w_HL * ||DWT(SR)_HL - DWT(HR)_HL||_1 +
             w_HH * ||DWT(SR)_HH - DWT(HR)_HH||_1
    """
    def __init__(self, in_channels=3, w_ll=1.0, w_lh=2.0, w_hl=2.0, w_hh=3.0, subband_weights=None):
        super().__init__()
        if subband_weights is not None:
            w_ll = subband_weights.get("ll", w_ll)
            w_lh = subband_weights.get("lh", w_lh)
            w_hl = subband_weights.get("hl", w_hl)
            w_hh = subband_weights.get("hh", w_hh)

        self.w_ll = w_ll
        self.w_lh = w_lh
        self.w_hl = w_hl
        self.w_hh = w_hh

        self.dwt = DWT2D(in_channels=in_channels)
        self.l1 = nn.L1Loss(reduction="mean")

    def forward(self, sr, hr):
        """
        Args:
            sr: Generated Super-Resolution Tensor [B, C, H, W]
            hr: Target High-Resolution Ground-Truth Tensor [B, C, H, W]
        Returns:
            total_loss: Scalar PyTorch tensor for backpropagation
            loss_dict: Dictionary containing sub-band loss breakdown values
        """
        sr_ll, sr_lh, sr_hl, sr_hh = self.dwt(sr)
        hr_ll, hr_lh, hr_hl, hr_hh = self.dwt(hr)

        loss_ll = self.l1(sr_ll, hr_ll)
        loss_lh = self.l1(sr_lh, hr_lh)
        loss_hl = self.l1(sr_hl, hr_hl)
        loss_hh = self.l1(sr_hh, hr_hh)

        total_loss = (
            self.w_ll * loss_ll +
            self.w_lh * loss_lh +
            self.w_hl * loss_hl +
            self.w_hh * loss_hh
        )

        loss_dict = {
            "loss_freq_total": total_loss.item(),
            "loss_freq_ll": loss_ll.item(),
            "loss_freq_lh": loss_lh.item(),
            "loss_freq_hl": loss_hl.item(),
            "loss_freq_hh": loss_hh.item()
        }

        return total_loss, loss_dict
