import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.segmentation_head import LightweightSegmentationUNet


def generate_ground_truth_mask(hr_tensor, threshold=0.35):
    """
    Generates a deterministic ground-truth building/structure binary mask from HR satellite image.
    Uses multi-channel luminance + Sobel gradient magnitude to isolate structural footprints.
    
    Args:
        hr_tensor: HR RGB image tensor [B, 3, H, W] in [0, 1]
        threshold: Decision threshold for structural mask binarization
    Returns:
        gt_mask: Binary mask tensor [B, 1, H, W] in {0.0, 1.0}
    """
    # 1. Luminance computation: Y = 0.299 R + 0.587 G + 0.114 B
    r, g, b = hr_tensor[:, 0:1, :, :], hr_tensor[:, 1:2, :, :], hr_tensor[:, 2:3, :, :]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # 2. Sobel edge magnitude
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=hr_tensor.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=hr_tensor.device).view(1, 1, 3, 3)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    edge_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    # 3. Combine normalized edge magnitude and high-contrast structural features
    edge_norm = (edge_mag - edge_mag.min()) / (edge_mag.max() - edge_mag.min() + 1e-6)
    struct_score = 0.6 * gray + 0.4 * edge_norm

    # 4. Adaptive binarization per image
    mean_val = struct_score.mean(dim=(2, 3), keepdim=True)
    gt_mask = (struct_score > (mean_val * 0.9 + threshold * 0.1)).float()
    return gt_mask


def compute_miou(pred_mask, gt_mask, threshold=0.5, eps=1e-6):
    """Computes Mean Intersection over Union (mIoU) for Binary Segmentation."""
    pred_bin = (pred_mask > threshold).float()
    gt_bin = (gt_mask > threshold).float()

    intersection = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
    union = (pred_bin + gt_bin - (pred_bin * gt_bin)).sum(dim=(1, 2, 3))

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def compute_dice_score(pred_mask, gt_mask, threshold=0.5, eps=1e-6):
    """Computes Dice Coefficient (F1 Score) for Binary Segmentation."""
    pred_bin = (pred_mask > threshold).float()
    gt_bin = (gt_mask > threshold).float()

    intersection = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
    total_area = pred_bin.sum(dim=(1, 2, 3)) + gt_bin.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + eps) / (total_area + eps)
    return dice.mean().item()


def compute_precision_recall(pred_mask, gt_mask, threshold=0.5, eps=1e-6):
    """Computes Pixel Precision and Recall for Binary Segmentation."""
    pred_bin = (pred_mask > threshold).float()
    gt_bin = (gt_mask > threshold).float()

    tp = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1.0 - gt_bin)).sum(dim=(1, 2, 3))
    fn = ((1.0 - pred_bin) * gt_bin).sum(dim=(1, 2, 3))

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return precision.mean().item(), recall.mean().item()


def extract_boundary(mask):
    """Extracts 1-pixel binary boundary from mask using max pool erosion difference."""
    kernel = torch.ones((1, 1, 3, 3), device=mask.device)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    boundary = (mask - eroded) > 0.5
    return boundary.float()


def compute_boundary_f1(pred_mask, gt_mask, threshold=0.5, eps=1e-6):
    """Computes Boundary F1 Score evaluating edge/contour fidelity."""
    pred_bin = (pred_mask > threshold).float()
    gt_bin = (gt_mask > threshold).float()

    pred_b = extract_boundary(pred_bin)
    gt_b = extract_boundary(gt_bin)

    tp_b = (pred_b * gt_b).sum(dim=(1, 2, 3))
    fp_b = (pred_b * (1.0 - gt_b)).sum(dim=(1, 2, 3))
    fn_b = ((1.0 - pred_b) * gt_b).sum(dim=(1, 2, 3))

    prec_b = (tp_b + eps) / (tp_b + fp_b + eps)
    rec_b = (tp_b + eps) / (tp_b + fn_b + eps)
    f1_b = (2.0 * prec_b * rec_b + eps) / (prec_b + rec_b + eps)

    return f1_b.mean().item()


def get_trained_segmentation_net(device="cpu", save_path="experiments/segmentation_head.pth", dataset=None):
    """
    Returns a pre-trained, calibrated LightweightSegmentationUNet model for downstream evaluation.
    If checkpoint doesn't exist, trains it on HR images and GT target masks for 10 epochs.
    """
    seg_net = LightweightSegmentationUNet(in_channels=3, num_classes=1, num_features=16).to(device)

    if os.path.exists(save_path):
        seg_net.load_state_dict(torch.load(save_path, map_location=device))
        seg_net.eval()
        return seg_net

    if dataset is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
        optimizer = torch.optim.Adam(seg_net.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        seg_net.train()
        for epoch in range(10):
            for batch in loader:
                hr = batch["hr"].to(device)
                gt_mask = generate_ground_truth_mask(hr)
                optimizer.zero_grad()
                logits = seg_net(hr)
                loss = criterion(logits, gt_mask)
                loss.backward()
                optimizer.step()

        torch.save(seg_net.state_dict(), save_path)

    seg_net.eval()
    return seg_net
