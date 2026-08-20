import pytest
import torch
from models import LightweightSegmentationUNet
from losses import DiceLoss, DownstreamSegmentationLoss
from evaluation import compute_miou, compute_dice_score


def test_segmentation_unet_shape():
    unet = LightweightSegmentationUNet(in_channels=3, num_classes=1, num_features=16)
    x = torch.rand(2, 3, 64, 64, dtype=torch.float32)

    logits = unet(x)
    assert logits.shape == (2, 1, 64, 64), f"Expected shape (2, 1, 64, 64), got {logits.shape}"
    assert not torch.isnan(logits).any(), "UNet logit tensor contains NaNs."


def test_dice_loss_invariants():
    dice = DiceLoss()
    mask = torch.zeros(2, 1, 32, 32, dtype=torch.float32)
    mask[:, :, 8:24, 8:24] = 1.0

    loss_identical = dice(mask, mask)
    assert pytest.approx(loss_identical.item(), abs=1e-5) == 0.0, "Expected zero loss for identical target & prediction masks."

    empty_pred = torch.zeros(2, 1, 32, 32, dtype=torch.float32)
    loss_empty = dice(empty_pred, mask)
    assert loss_empty.item() > 0.5, "Expected high loss for empty prediction against populated target mask."


def test_segmentation_evaluation_metrics():
    gt_mask = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    gt_mask[:, :, 10:20, 10:20] = 1.0  # 10x10 = 100 pixels

    # 1. Exact match
    pred_exact = gt_mask.clone()
    miou_exact = compute_miou(pred_exact, gt_mask)
    dice_exact = compute_dice_score(pred_exact, gt_mask)

    assert pytest.approx(miou_exact, abs=1e-5) == 1.0
    assert pytest.approx(dice_exact, abs=1e-5) == 1.0

    # 2. Half overlap (10x5 = 50 overlap pixels)
    pred_half = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    pred_half[:, :, 10:20, 10:15] = 1.0

    miou_half = compute_miou(pred_half, gt_mask)
    dice_half = compute_dice_score(pred_half, gt_mask)

    # IoU = 50 / 100 = 0.5
    # Dice = 2 * 50 / (50 + 100) = 100 / 150 = 0.6667
    assert pytest.approx(miou_half, abs=1e-3) == 0.5
    assert pytest.approx(dice_half, abs=1e-3) == 2.0 / 3.0


def test_downstream_segmentation_loss_forward():
    unet = LightweightSegmentationUNet(in_channels=3, num_classes=1, num_features=16)
    seg_loss = DownstreamSegmentationLoss(seg_net=unet, freeze_seg_net=True)

    hr = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    sr = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    target_mask = (torch.rand(2, 1, 64, 64) > 0.5).float()

    loss_with_gt = seg_loss(sr, hr, target_mask=target_mask)
    loss_pseudo_gt = seg_loss(sr, hr, target_mask=None)

    assert isinstance(loss_with_gt, torch.Tensor) and loss_with_gt.item() > 0.0
    assert isinstance(loss_pseudo_gt, torch.Tensor) and loss_pseudo_gt.item() > 0.0


def test_segmentation_loss_autograd():
    unet = LightweightSegmentationUNet(in_channels=3, num_classes=1, num_features=16)
    seg_loss = DownstreamSegmentationLoss(seg_net=unet, freeze_seg_net=True)

    hr = torch.rand(1, 3, 32, 32, dtype=torch.float32)
    sr = torch.rand(1, 3, 32, 32, requires_grad=True)

    loss = seg_loss(sr, hr)
    loss.backward()

    assert sr.grad is not None, "Gradients should backpropagate through UNet segmentation head back to SR tensor."
    assert not torch.isnan(sr.grad).any(), "Gradient contains NaN values."
