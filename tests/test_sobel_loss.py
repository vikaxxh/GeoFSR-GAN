import pytest
import torch
import torch.nn.functional as F
from losses import SobelEdgeFilter, SobelEdgeLoss


def test_sobel_synthetic_shapes():
    filter_mod = SobelEdgeFilter(in_channels=1, eps=1e-6)

    # 1. Vertical Line Image (x = 16)
    img_v = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    img_v[:, :, :, 16] = 1.0
    _, gx_v, gy_v = filter_mod(img_v)

    assert gx_v.abs().sum() > gy_v.abs().sum(), "Vertical line should produce stronger Gx gradient."

    # 2. Horizontal Line Image (y = 16)
    img_h = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    img_h[:, :, 16, :] = 1.0
    _, gx_h, gy_h = filter_mod(img_h)

    assert gy_h.abs().sum() > gx_h.abs().sum(), "Horizontal line should produce stronger Gy gradient."

    # 3. Square Box Image (Center 16x16)
    img_box = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    img_box[:, :, 8:24, 8:24] = 1.0
    mag_box, _, _ = filter_mod(img_box)

    # Interior of box has zero gradient magnitude (~sqrt(eps) = 1e-3)
    interior_val = mag_box[:, :, 16, 16].item()
    edge_val = mag_box[:, :, 8, 8].item()

    assert pytest.approx(interior_val, abs=2e-3) == 0.001, "Interior of box should match sqrt(eps)."
    assert edge_val > 0.5, "Corner/edge of box should have strong edge magnitude."


def test_sobel_edge_loss_zero_for_identical():
    criterion = SobelEdgeLoss(in_channels=3)
    hr = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    sr = hr.clone()

    loss = criterion(sr, hr)
    assert pytest.approx(loss.item(), abs=1e-6) == 0.0, f"Expected 0 loss for identical images, got {loss.item()}"


def test_sobel_edge_loss_blur_sensitivity():
    criterion = SobelEdgeLoss(in_channels=3)
    hr = torch.rand(2, 3, 64, 64, dtype=torch.float32)

    # Blur image to smear edges
    kernel = torch.ones(1, 1, 3, 3) / 9.0
    kernel = kernel.repeat(3, 1, 1, 1)
    sr_blurred = F.conv2d(hr, kernel, padding=1, groups=3)

    loss_blur = criterion(sr_blurred, hr)
    assert loss_blur.item() > 0.05, "Edge loss should be significant for blurred images."


def test_sobel_edge_loss_autograd():
    criterion = SobelEdgeLoss(in_channels=3)
    hr = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    sr = torch.rand(2, 3, 64, 64, requires_grad=True)

    loss = criterion(sr, hr)
    loss.backward()

    assert sr.grad is not None, "Gradients should backpropagate to SR tensor."
    assert not torch.isnan(sr.grad).any(), "Gradient contains NaN values."
    assert not torch.isinf(sr.grad).any(), "Gradient contains Inf values."
