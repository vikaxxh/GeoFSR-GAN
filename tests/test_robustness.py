import pytest
import torch
from models import BicubicBaseline
from datasets import SatelliteDataset
from torch.utils.data import DataLoader
from evaluation import (
    apply_gaussian_noise,
    apply_gaussian_blur,
    apply_jpeg_compression,
    evaluate_perturbation_resilience
)


def test_perturbation_operators_shape_and_bounds():
    x = torch.rand(2, 3, 48, 48, dtype=torch.float32)

    noisy = apply_gaussian_noise(x, std=0.05)
    assert noisy.shape == (2, 3, 48, 48)
    assert (noisy >= 0.0).all() and (noisy <= 1.0).all()

    blurred = apply_gaussian_blur(x, sigma=1.0)
    assert blurred.shape == (2, 3, 48, 48)
    assert (blurred >= 0.0).all() and (blurred <= 1.0).all()

    jpeg = apply_jpeg_compression(x, quality=50)
    assert jpeg.shape == (2, 3, 48, 48)
    assert (jpeg >= 0.0).all() and (jpeg <= 1.0).all()


def test_evaluate_perturbation_resilience():
    model = BicubicBaseline(scale=4)
    x_lr = torch.rand(1, 3, 16, 16)
    x_hr = torch.rand(1, 3, 64, 64)

    dummy_loader = [{"lr": x_lr, "hr": x_hr}]
    res = evaluate_perturbation_resilience(model, dummy_loader, perturbation_fn=None, device="cpu")

    assert "psnr" in res and "ssim" in res
    assert isinstance(res["psnr"], float) and isinstance(res["ssim"], float)
