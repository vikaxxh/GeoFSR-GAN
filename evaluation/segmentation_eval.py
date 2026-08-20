import torch


def compute_miou(pred_mask, gt_mask, threshold=0.5, eps=1e-6):
    """
    Computes Mean Intersection over Union (mIoU) for Binary Segmentation.
    
    Args:
        pred_mask: Predicted probabilities tensor [B, 1, H, W] in [0, 1]
        gt_mask: Ground-Truth binary mask tensor [B, 1, H, W] in [0, 1]
        threshold: Binarization decision threshold (default 0.5)
        eps: Epsilon for numerical stability
    Returns:
        miou: Mean IoU score in [0, 1]
    """
    pred_bin = (pred_mask > threshold).float()
    gt_bin = (gt_mask > threshold).float()

    intersection = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
    union = (pred_bin + gt_bin - (pred_bin * gt_bin)).sum(dim=(1, 2, 3))

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def compute_dice_score(pred_mask, gt_mask, threshold=0.5, eps=1e-6):
    """
    Computes Dice Coefficient (F1 Score) for Binary Segmentation.
    
    Args:
        pred_mask: Predicted probabilities tensor [B, 1, H, W] in [0, 1]
        gt_mask: Ground-Truth binary mask tensor [B, 1, H, W] in [0, 1]
        threshold: Binarization decision threshold (default 0.5)
        eps: Epsilon for numerical stability
    Returns:
        dice: Mean Dice score in [0, 1]
    """
    pred_bin = (pred_mask > threshold).float()
    gt_bin = (gt_mask > threshold).float()

    intersection = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
    total_area = pred_bin.sum(dim=(1, 2, 3)) + gt_bin.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + eps) / (total_area + eps)
    return dice.mean().item()
