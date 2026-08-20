import pytest
import torch
from models import SpatialPatchGANDiscriminator, FrequencyPatchGANDiscriminator
from losses import DualDomainAdversarialLoss


def test_frequency_patchgan_discriminator_image_input_shape():
    disc_freq = FrequencyPatchGANDiscriminator(in_channels=3, num_features=64, num_layers=3)
    hr = torch.rand(2, 3, 96, 96, dtype=torch.float32)

    validity = disc_freq(hr)
    assert validity.ndim == 4, f"Expected 4D validity map tensor, got {validity.ndim}D"
    assert validity.shape[0] == 2 and validity.shape[1] == 1, f"Expected [2, 1, H_p, W_p], got {validity.shape}"
    assert not torch.isnan(validity).any(), "Frequency PatchGAN output contains NaNs."


def test_frequency_patchgan_discriminator_dwt_tensor_input_shape():
    disc_freq = FrequencyPatchGANDiscriminator(in_channels=3, num_features=64, num_layers=3)
    dwt_tensor = torch.rand(2, 12, 48, 48, dtype=torch.float32)

    validity = disc_freq(dwt_tensor)
    assert validity.shape[0] == 2 and validity.shape[1] == 1, f"Expected [2, 1, H_p, W_p], got {validity.shape}"


def test_dual_domain_adversarial_loss_lsgan():
    dual_loss = DualDomainAdversarialLoss(gan_type="lsgan", lambda_spatial=0.005, lambda_freq=0.005)

    d_spat_real = torch.ones(2, 1, 9, 9, dtype=torch.float32) * 0.9
    d_spat_fake = torch.ones(2, 1, 9, 9, dtype=torch.float32) * 0.1

    d_freq_real = torch.ones(2, 1, 9, 9, dtype=torch.float32) * 0.85
    d_freq_fake = torch.ones(2, 1, 9, 9, dtype=torch.float32) * 0.15

    loss_g, dict_g = dual_loss.forward_g(d_spat_real, d_spat_fake, d_freq_real, d_freq_fake)
    loss_d, dict_d = dual_loss.forward_d(d_spat_real, d_spat_fake, d_freq_real, d_freq_fake)

    assert loss_g.item() > 0.0 and "loss_g_adv_total" in dict_g
    assert loss_d.item() > 0.0 and "loss_d_adv_total" in dict_d


def test_dual_domain_adversarial_loss_ragan():
    dual_loss = DualDomainAdversarialLoss(gan_type="ragan", lambda_spatial=0.005, lambda_freq=0.005)

    d_spat_real = torch.rand(2, 1, 9, 9, dtype=torch.float32)
    d_spat_fake = torch.rand(2, 1, 9, 9, dtype=torch.float32)

    d_freq_real = torch.rand(2, 1, 9, 9, dtype=torch.float32)
    d_freq_fake = torch.rand(2, 1, 9, 9, dtype=torch.float32)

    loss_g, dict_g = dual_loss.forward_g(d_spat_real, d_spat_fake, d_freq_real, d_freq_fake)
    loss_d, dict_d = dual_loss.forward_d(d_spat_real, d_spat_fake, d_freq_real, d_freq_fake)

    assert loss_g.item() >= 0.0
    assert loss_d.item() >= 0.0


def test_frequency_discriminator_autograd():
    disc_spat = SpatialPatchGANDiscriminator(in_channels=3, num_features=32, num_layers=2)
    disc_freq = FrequencyPatchGANDiscriminator(in_channels=3, num_features=32, num_layers=2)
    dual_loss = DualDomainAdversarialLoss(gan_type="lsgan")

    hr = torch.rand(1, 3, 48, 48, dtype=torch.float32)
    sr = torch.rand(1, 3, 48, 48, requires_grad=True)

    d_spat_real = disc_spat(hr)
    d_spat_fake = disc_spat(sr)

    d_freq_real = disc_freq(hr)
    d_freq_fake = disc_freq(sr)

    loss_g, _ = dual_loss.forward_g(d_spat_real, d_spat_fake, d_freq_real, d_freq_fake)
    loss_g.backward()

    assert sr.grad is not None, "Gradients should backpropagate through dual discriminators to SR input."
    assert not torch.isnan(sr.grad).any(), "Gradient contains NaN values."
