import pytest
import torch
from PIL import Image
from datasets import RealisticDegradation


def test_degradation_tensor_scale_x4():
    degradation = RealisticDegradation()
    hr_tensor = torch.rand(3, 96, 96, dtype=torch.float32)
    lr_tensor = degradation.degrade(hr_tensor, scale=4, seed=42)

    assert torch.is_tensor(lr_tensor), "Output should be a PyTorch tensor."
    assert lr_tensor.dtype == torch.float32, f"Expected float32 dtype, got {lr_tensor.dtype}."
    assert lr_tensor.shape == (3, 24, 24), f"Expected shape (3, 24, 24), got {lr_tensor.shape}."
    assert lr_tensor.min().item() >= 0.0, "Values should be >= 0.0"
    assert lr_tensor.max().item() <= 1.0, "Values should be <= 1.0"


def test_degradation_tensor_scale_x2():
    degradation = RealisticDegradation()
    hr_tensor = torch.rand(3, 128, 128, dtype=torch.float32)
    lr_tensor = degradation.degrade(hr_tensor, scale=2, seed=123)

    assert lr_tensor.shape == (3, 64, 64), f"Expected shape (3, 64, 64), got {lr_tensor.shape}."


def test_degradation_seed_reproducibility():
    degradation = RealisticDegradation()
    hr_tensor = torch.rand(3, 64, 64, dtype=torch.float32)

    lr1 = degradation.degrade(hr_tensor, scale=4, seed=99)
    lr2 = degradation.degrade(hr_tensor, scale=4, seed=99)
    lr_different_seed = degradation.degrade(hr_tensor, scale=4, seed=100)

    assert torch.allclose(lr1, lr2, atol=1e-5), "Same seed must produce identical degradation outputs."
    assert not torch.allclose(lr1, lr_different_seed, atol=1e-5), "Different seeds should produce distinct degradation outputs."
