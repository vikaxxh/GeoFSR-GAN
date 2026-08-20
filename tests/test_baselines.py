import pytest
import torch
import numpy as np
from models import BicubicBaseline, SimpleSpatialSR
from evaluation import calculate_psnr, calculate_ssim


def test_bicubic_baseline_forward():
    model = BicubicBaseline(scale=4)
    lr = torch.rand(2, 3, 24, 24, dtype=torch.float32)
    sr = model(lr)

    assert sr.shape == (2, 3, 96, 96), f"Expected shape (2, 3, 96, 96), got {sr.shape}"
    assert sr.min().item() >= 0.0 and sr.max().item() <= 1.0, "Values must be bounded in [0.0, 1.0]"


def test_simple_spatial_sr_forward():
    model = SimpleSpatialSR(scale=4, in_channels=3, out_channels=3, num_features=32)
    lr = torch.rand(2, 3, 24, 24, dtype=torch.float32)
    sr = model(lr)

    assert sr.shape == (2, 3, 96, 96), f"Expected shape (2, 3, 96, 96), got {sr.shape}"
    assert sr.min().item() >= 0.0 and sr.max().item() <= 1.0, "Values must be bounded in [0.0, 1.0]"


def test_image_metrics_invariants():
    img1 = torch.rand(3, 64, 64, dtype=torch.float32)
    img2 = img1.clone()

    psnr_same = calculate_psnr(img1, img2)
    ssim_same = calculate_ssim(img1, img2)

    assert psnr_same >= 90.0, f"Expected PSNR >= 90 for identical images, got {psnr_same}"
    assert pytest.approx(ssim_same, abs=1e-4) == 1.0, f"Expected SSIM == 1.0 for identical images, got {ssim_same}"

    # Distorted image
    img_distorted = img1 + 0.1 * torch.randn_like(img1)
    img_distorted = torch.clamp(img_distorted, 0.0, 1.0)

    psnr_diff = calculate_psnr(img1, img_distorted)
    ssim_diff = calculate_ssim(img1, img_distorted)

    assert psnr_diff < 90.0, "PSNR should decrease for distorted images."
    assert ssim_diff < 1.0, "SSIM should be < 1.0 for distorted images."
