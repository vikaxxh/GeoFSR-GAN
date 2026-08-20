import pytest
import torch
import torch.nn.functional as F
from losses import MultiBandFrequencyLoss


def test_frequency_loss_zero_for_identical():
    criterion = MultiBandFrequencyLoss(in_channels=3)
    hr = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    sr = hr.clone()

    loss, loss_dict = criterion(sr, hr)

    assert pytest.approx(loss.item(), abs=1e-6) == 0.0, f"Expected 0 loss for identical images, got {loss.item()}"
    assert loss_dict["loss_freq_total"] == 0.0
    assert loss_dict["loss_freq_hh"] == 0.0


def test_frequency_loss_subband_responsiveness():
    criterion = MultiBandFrequencyLoss(in_channels=3, w_ll=1.0, w_lh=2.0, w_hl=2.0, w_hh=3.0)
    base_img = torch.full((2, 3, 64, 64), 0.5, dtype=torch.float32)

    # 1. Uniform low-frequency bias (+0.2 to all pixels) -> triggers LL sub-band
    sr_bias = base_img + 0.2
    _, dict_bias = criterion(sr_bias, base_img)

    # 2. Diagonal high-frequency noise (+0.2 on main diagonal, -0.2 on anti-diagonal of 2x2 blocks) -> triggers HH sub-band
    noise_hh = torch.zeros_like(base_img)
    noise_hh[:, :, 0::2, 0::2] += 0.2  # a
    noise_hh[:, :, 0::2, 1::2] -= 0.2  # b
    noise_hh[:, :, 1::2, 0::2] -= 0.2  # c
    noise_hh[:, :, 1::2, 1::2] += 0.2  # d
    sr_hh_noise = base_img + noise_hh
    _, dict_hh = criterion(sr_hh_noise, base_img)

    # Uniform bias should impact LL loss, whereas diagonal noise impacts HH loss
    assert dict_bias["loss_freq_ll"] > 0.1 and dict_bias["loss_freq_hh"] == 0.0, "Uniform bias should impact LL loss."
    assert dict_hh["loss_freq_hh"] > 0.3 and pytest.approx(dict_hh["loss_freq_ll"], abs=1e-5) == 0.0, "Diagonal noise should impact HH loss."


def test_subband_weight_scaling():
    hr = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    sr = torch.clamp(hr + 0.1 * torch.randn_like(hr), 0.0, 1.0)

    criterion1 = MultiBandFrequencyLoss(in_channels=3, w_ll=1.0, w_lh=1.0, w_hl=1.0, w_hh=1.0)
    criterion2 = MultiBandFrequencyLoss(in_channels=3, w_ll=1.0, w_lh=1.0, w_hl=1.0, w_hh=10.0)

    loss1, _ = criterion1(sr, hr)
    loss2, _ = criterion2(sr, hr)

    assert loss2.item() > loss1.item(), "Higher HH weight should increase total frequency loss."


def test_frequency_loss_autograd():
    criterion = MultiBandFrequencyLoss(in_channels=3)
    hr = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    sr = torch.rand(2, 3, 64, 64, requires_grad=True)

    loss, _ = criterion(sr, hr)
    loss.backward()

    assert sr.grad is not None, "Gradients should backpropagate to SR tensor."
    assert not torch.isnan(sr.grad).any(), "Gradient contains NaN values."
    assert not torch.isinf(sr.grad).any(), "Gradient contains Inf values."
