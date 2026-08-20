import pytest
import torch
from models import LightweightCrossAttention, DualCrossAttention, GeoFSRGenerator


def test_lightweight_cross_attention_shape():
    attn = LightweightCrossAttention(num_features=32, reduction_factor=2)
    q = torch.rand(2, 32, 24, 24, dtype=torch.float32)
    kv = torch.rand(2, 32, 24, 24, dtype=torch.float32)

    out = attn(q, kv)
    assert out.shape == (2, 32, 24, 24), f"Expected shape (2, 32, 24, 24), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN values."


def test_dual_cross_attention_shape():
    attn = DualCrossAttention(num_features=32, reduction_factor=2)
    f_s = torch.rand(2, 32, 24, 24, dtype=torch.float32)
    f_f = torch.rand(2, 32, 24, 24, dtype=torch.float32)

    out = attn(f_s, f_f)
    assert out.shape == (2, 32, 24, 24), f"Expected shape (2, 32, 24, 24), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN values."


@pytest.mark.parametrize("fusion_type", ["cross_attention", "dual_cross_attention"])
def test_geofsr_generator_cross_attention_modes(fusion_type):
    generator = GeoFSRGenerator(
        scale=4,
        in_channels=3,
        out_channels=3,
        num_features=32,
        num_spatial_blocks=2,
        fusion_type=fusion_type
    )
    lr = torch.rand(1, 3, 24, 24, dtype=torch.float32)
    sr = generator(lr)

    assert sr.shape == (1, 3, 96, 96), f"Expected SR shape (1, 3, 96, 96), got {sr.shape}"
    assert sr.min().item() >= 0.0 and sr.max().item() <= 1.0, "SR values must be bounded in [0.0, 1.0]"


def test_cross_attention_autograd():
    attn = DualCrossAttention(num_features=32, reduction_factor=2)
    f_s = torch.rand(2, 32, 24, 24, requires_grad=True)
    f_f = torch.rand(2, 32, 24, 24, requires_grad=True)

    out = attn(f_s, f_f)
    loss = out.sum()
    loss.backward()

    assert f_s.grad is not None and f_f.grad is not None, "Gradients should backpropagate to both inputs."
    assert not torch.isnan(f_s.grad).any(), "Spatial gradient contains NaN values."
    assert not torch.isnan(f_f.grad).any(), "Frequency gradient contains NaN values."
