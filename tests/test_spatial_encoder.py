import pytest
import torch
from models import ResidualDenseBlock, ResidualInResidualDenseBlock, SpatialEncoder


def test_rdb_forward():
    rdb = ResidualDenseBlock(num_features=32, growth_rate=16, res_scale=0.2)
    x = torch.rand(2, 32, 24, 24, dtype=torch.float32)
    out = rdb(x)

    assert out.shape == (2, 32, 24, 24), f"Expected shape (2, 32, 24, 24), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN values."
    assert not torch.isinf(out).any(), "Output contains Inf values."


def test_rrdb_forward():
    rrdb = ResidualInResidualDenseBlock(num_features=32, growth_rate=16, res_scale=0.2)
    x = torch.rand(2, 32, 24, 24, dtype=torch.float32)
    out = rrdb(x)

    assert out.shape == (2, 32, 24, 24), f"Expected shape (2, 32, 24, 24), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN values."


def test_spatial_encoder_cpu_debug_shape():
    encoder = SpatialEncoder(in_channels=3, num_features=32, num_blocks=2, growth_rate=16, res_scale=0.2)
    lr = torch.rand(1, 3, 24, 24, dtype=torch.float32)
    feat = encoder(lr)

    assert feat.shape == (1, 32, 24, 24), f"Expected shape (1, 32, 24, 24), got {feat.shape}"

    param_count = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\n[SpatialEncoder CPU Debug] Parameter count: {param_count:,}")
    assert param_count < 500_000, f"CPU encoder parameter count ({param_count}) should be under 500k."


def test_spatial_encoder_gpu_full_shape():
    encoder = SpatialEncoder(in_channels=3, num_features=64, num_blocks=4, growth_rate=32, res_scale=0.2)
    lr = torch.rand(1, 3, 32, 32, dtype=torch.float32)
    feat = encoder(lr)

    assert feat.shape == (1, 64, 32, 32), f"Expected shape (1, 64, 32, 32), got {feat.shape}"


def test_spatial_encoder_autograd():
    encoder = SpatialEncoder(in_channels=3, num_features=32, num_blocks=2, growth_rate=16)
    lr = torch.rand(2, 3, 24, 24, requires_grad=True)
    feat = encoder(lr)
    loss = feat.sum()
    loss.backward()

    assert lr.grad is not None, "Gradients should backpropagate to input."
    assert not torch.isnan(lr.grad).any(), "Input gradient contains NaN values."
    assert not torch.isinf(lr.grad).any(), "Input gradient contains Inf values."
