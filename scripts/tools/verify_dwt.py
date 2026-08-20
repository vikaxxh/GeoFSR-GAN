import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.frequency_encoder import DWT2D, IDWT2D
from losses.frequency_loss import MultiBandFrequencyLoss


def test_dwt_reconstruction():
    torch.manual_seed(42)
    dwt = DWT2D(in_channels=3)
    idwt = IDWT2D(in_channels=3)

    x = torch.randn(4, 3, 96, 96)
    ll, lh, hl, hh = dwt(x)

    print(f"Input Shape: {x.shape}")
    print(f"Sub-band Shapes: LL={ll.shape}, LH={lh.shape}, HL={hl.shape}, HH={hh.shape}")

    assert ll.shape == (4, 3, 48, 48), f"Incorrect LL shape: {ll.shape}"
    assert lh.shape == (4, 3, 48, 48), f"Incorrect LH shape: {lh.shape}"
    assert hl.shape == (4, 3, 48, 48), f"Incorrect HL shape: {hl.shape}"
    assert hh.shape == (4, 3, 48, 48), f"Incorrect HH shape: {hh.shape}"

    x_rec = idwt(ll, lh, hl, hh)
    max_diff = torch.max(torch.abs(x_rec - x)).item()
    mean_diff = torch.mean(torch.abs(x_rec - x)).item()

    print(f"Reconstruction Max Abs Error: {max_diff:.8e}")
    print(f"Reconstruction Mean Abs Error: {mean_diff:.8e}")

    assert max_diff < 1e-6, f"IDWT reconstruction error too high: {max_diff}"
    print("[DWT Verification] 100% Exact Sub-band Reconstruction Verified SUCCESSFUL!")

    freq_loss_module = MultiBandFrequencyLoss(in_channels=3)
    loss, loss_dict = freq_loss_module(x_rec, x)
    print(f"[MultiBandFrequencyLoss Test] Loss value on exact reconstruction: {loss.item():.8e}")


if __name__ == "__main__":
    test_dwt_reconstruction()
