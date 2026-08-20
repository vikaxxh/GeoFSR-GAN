import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Differentiable Soft Dice Loss Module for Binary Segmentation.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_probs, target_probs):
        """
        Args:
            pred_probs: Predicted probability tensor [B, 1, H, W] in [0, 1]
            target_probs: Target binary mask tensor [B, 1, H, W] in [0, 1]
        Returns:
            loss: Mean 1.0 - Dice coefficient
        """
        intersection = (pred_probs * target_probs).sum(dim=(1, 2, 3))
        union = pred_probs.sum(dim=(1, 2, 3)) + target_probs.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DownstreamSegmentationLoss(nn.Module):
    """
    Downstream Task Guidance Loss Module for Satellite Imagery Super-Resolution.
    
    Enforces task consistency by comparing segmentation predictions generated from
    SR image vs HR ground-truth image:
    L_seg = L_BCE(Seg(SR), Seg(HR)) + L_Dice(Seg(SR), Seg(HR))
    """
    def __init__(self, seg_net, freeze_seg_net=True):
        super().__init__()
        self.seg_net = seg_net
        self.dice_loss = DiceLoss()

        if freeze_seg_net:
            for param in self.seg_net.parameters():
                param.requires_grad = False
            self.seg_net.eval()

    def forward(self, sr, hr, target_mask=None):
        """
        Args:
            sr: Generated Super-Resolution image tensor [B, 3, H, W]
            hr: Ground-Truth target image tensor [B, 3, H, W]
            target_mask: Optional Ground-Truth binary segmentation mask [B, 1, H, W]
        Returns:
            loss: Combined BCE + Dice segmentation loss
        """
        logits_sr = self.seg_net(sr)
        probs_sr = torch.sigmoid(logits_sr)

        if target_mask is not None:
            target_probs = target_mask.float()
        else:
            with torch.no_grad():
                logits_hr = self.seg_net(hr)
                target_probs = torch.sigmoid(logits_hr)

        loss_bce = F.binary_cross_entropy_with_logits(logits_sr, target_probs)
        loss_dice = self.dice_loss(probs_sr, target_probs)

        return loss_bce + loss_dice
