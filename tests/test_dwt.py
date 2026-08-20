import pytest
import torch
from models import DWT2D, IDWT2D, FrequencyEncoder


def test_dwt_subband_shapes():
    dwt = DWT2D(in_channels=3)
    x = torch.rand(2, 3, 32, 32, dtype=torch.float32)
    ll, lh, hl, hh = dwt(x)

    assert ll.shape == (2, 3, 16, 16), f"Expected LL shape (2, 3, 16, 16), got {ll.shape}"
    assert lh.shape == (2, 3, 16, 16), f"Expected LH shape (2, 3, 16, 16), got {lh.shape}"
    assert hl.shape == (2, 3, 16, 16), f"Expected HL shape (2, 3, 16, 16), got {hl.shape}"
    assert hh.shape == (2, 3, 16, 16), f"Expected HH shape (2, 3, 16, 16), got {hh.shape}"


def test_dwt_idwt_perfect_reconstruction():
    dwt = DWT2D(in_channels=3)
    idwt = IDWT2D(in_channels=3)

    x = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    ll, lh, hl, hh = dwt(x)
    rec = idwt(ll, lh, hl, hh)

    assert torch.allclose(x, rec, atol=1e-4), f"Max reconstruction error: {(x - rec).abs().max().item():.6f}"


def test_frequency_encoder_forward_shape():
    encoder = FrequencyEncoder(in_channels=3, num_features=32)
    lr = torch.rand(1, 3, 24, 24, dtype=torch.float32)
    freq_feat = encoder(lr)

    assert freq_feat.shape == (1, 32, 24, 24), f"Expected shape (1, 32, 24, 24), got {freq_feat.shape}"


def test_frequency_encoder_autograd():
    encoder = FrequencyEncoder(in_channels=3, num_features=32)
    lr = torch.rand(2, 3, 24, 24, requires_grad=True)
    freq_feat = encoder(lr)
    loss = freq_feat.sum()
    loss.backward()

    assert lr.grad is not None, "Gradients should backpropagate to input."
    assert not torch.isnan(lr.grad).any(), "Input gradient contains NaN values."
    assert not torch.isinf(lr.grad).any(), "Input gradient contains Inf values."
