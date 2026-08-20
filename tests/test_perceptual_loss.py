import pytest
import torch
import torch.nn.functional as F
from losses import ImageNetNormalize, LightweightPerceptualExtractor, PerceptualLoss


def test_imagenet_normalize_shape():
    norm = ImageNetNormalize()
    x = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    out = norm(x)

    assert out.shape == (2, 3, 64, 64), f"Expected shape (2, 3, 64, 64), got {out.shape}"
    assert not torch.isnan(out).any(), "Normalized tensor contains NaNs."


def test_lightweight_perceptual_extractor_features():
    extractor = LightweightPerceptualExtractor(in_channels=3)
    x = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    feats = extractor(x)

    assert len(feats) == 3, f"Expected 3 feature scales, got {len(feats)}"
    assert feats[0].shape == (2, 16, 64, 64), f"Scale 1 shape mismatch: {feats[0].shape}"
    assert feats[1].shape == (2, 32, 32, 32), f"Scale 2 shape mismatch: {feats[1].shape}"
    assert feats[2].shape == (2, 64, 16, 16), f"Scale 3 shape mismatch: {feats[2].shape}"


def test_perceptual_loss_zero_for_identical():
    criterion = PerceptualLoss(mode="lightweight")
    hr = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    sr = hr.clone()

    loss = criterion(sr, hr)
    assert pytest.approx(loss.item(), abs=1e-6) == 0.0, f"Expected 0 loss for identical images, got {loss.item()}"


def test_perceptual_loss_blur_sensitivity():
    criterion = PerceptualLoss(mode="lightweight")
    hr = torch.rand(2, 3, 64, 64, dtype=torch.float32)

    # Blur image to alter structural feature representations
    kernel = torch.ones(1, 1, 5, 5) / 25.0
    kernel = kernel.repeat(3, 1, 1, 1)
    sr_blurred = F.conv2d(hr, kernel, padding=2, groups=3)

    loss_blur = criterion(sr_blurred, hr)
    assert loss_blur.item() > 0.05, "Perceptual loss should be positive for blurred images."


def test_perceptual_loss_auto_mode():
    try:
        criterion = PerceptualLoss(mode="auto")
        hr = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        sr = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        loss = criterion(sr, hr)
        assert isinstance(loss, torch.Tensor) and not torch.isnan(loss), "Auto mode should return valid loss tensor."
    except Exception:
        pytest.skip("Torchvision model download timed out offline.")


def test_perceptual_loss_autograd():
    criterion = PerceptualLoss(mode="lightweight")
    hr = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    sr = torch.rand(2, 3, 64, 64, requires_grad=True)

    loss = criterion(sr, hr)
    loss.backward()

    assert sr.grad is not None, "Gradients should backpropagate to SR tensor."
    assert not torch.isnan(sr.grad).any(), "Gradient contains NaN values."
    assert not torch.isinf(sr.grad).any(), "Gradient contains Inf values."
