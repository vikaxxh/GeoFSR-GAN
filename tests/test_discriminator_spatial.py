import pytest
import torch
from models import SpatialPatchGANDiscriminator
from losses import LSGANLoss, RelativisticGANLoss


def test_spatial_patchgan_discriminator_shape():
    disc = SpatialPatchGANDiscriminator(in_channels=3, num_features=64, num_layers=3, use_spectral_norm=True)
    hr = torch.rand(2, 3, 96, 96, dtype=torch.float32)

    validity = disc(hr)
    assert validity.ndim == 4, f"Expected 4D tensor, got {validity.ndim}D"
    assert validity.shape[0] == 2 and validity.shape[1] == 1, f"Expected [2, 1, H_p, W_p], got {validity.shape}"
    assert not torch.isnan(validity).any(), "PatchGAN output contains NaNs."


def test_lsgan_loss_computation():
    lsgan = LSGANLoss()
    d_real = torch.tensor([[0.9, 0.95], [0.85, 0.9]], dtype=torch.float32)
    d_fake = torch.tensor([[0.1, 0.05], [0.15, 0.2]], dtype=torch.float32)

    loss_d = lsgan.forward_d(d_real, d_fake)
    loss_g = lsgan.forward_g(d_fake)

    assert isinstance(loss_d, torch.Tensor) and loss_d.item() >= 0.0
    assert isinstance(loss_g, torch.Tensor) and loss_g.item() >= 0.0


def test_relativistic_gan_loss_computation():
    ragan = RelativisticGANLoss()
    d_real = torch.tensor([[0.9, 0.95], [0.85, 0.9]], dtype=torch.float32)
    d_fake = torch.tensor([[0.1, 0.05], [0.15, 0.2]], dtype=torch.float32)

    loss_d = ragan.forward_d(d_real, d_fake)
    loss_g = ragan.forward_g(d_real, d_fake)

    assert isinstance(loss_d, torch.Tensor) and loss_d.item() >= 0.0
    assert isinstance(loss_g, torch.Tensor) and loss_g.item() >= 0.0


def test_spatial_discriminator_autograd():
    disc = SpatialPatchGANDiscriminator(in_channels=3, num_features=32, num_layers=2)
    lsgan = LSGANLoss()

    hr = torch.rand(1, 3, 48, 48, dtype=torch.float32)
    sr = torch.rand(1, 3, 48, 48, requires_grad=True)

    d_real = disc(hr)
    d_fake = disc(sr)

    loss_g = lsgan.forward_g(d_fake)
    loss_g.backward()

    assert sr.grad is not None, "Gradients should backpropagate through Discriminator to Generator output SR."
    assert not torch.isnan(sr.grad).any(), "Gradient contains NaN values."
