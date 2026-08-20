import pytest
import torch
from models import ConcatFusion, LearnedWeightFusion, AttentionFusion, SpatialFrequencyFusion, GeoFSRGenerator


def test_fusion_modules_shapes():
    f_s = torch.rand(2, 32, 24, 24, dtype=torch.float32)
    f_f = torch.rand(2, 32, 24, 24, dtype=torch.float32)

    concat_fusion = ConcatFusion(num_features=32)
    learned_fusion = LearnedWeightFusion(num_features=32)
    attn_fusion = AttentionFusion(num_features=32)

    out_c = concat_fusion(f_s, f_f)
    out_l = learned_fusion(f_s, f_f)
    out_a = attn_fusion(f_s, f_f)

    assert out_c.shape == (2, 32, 24, 24), f"Expected shape (2, 32, 24, 24), got {out_c.shape}"
    assert out_l.shape == (2, 32, 24, 24), f"Expected shape (2, 32, 24, 24), got {out_l.shape}"
    assert out_a.shape == (2, 32, 24, 24), f"Expected shape (2, 32, 24, 24), got {out_a.shape}"


@pytest.mark.parametrize("fusion_type", ["concat", "learned", "attention"])
def test_geofsr_generator_fusion_modes(fusion_type):
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
    assert sr.min().item() >= 0.0 and sr.max().item() <= 1.0, "SR values must be in [0.0, 1.0]"


def test_geofsr_generator_autograd():
    generator = GeoFSRGenerator(scale=4, num_features=32, fusion_type="attention")
    lr = torch.rand(1, 3, 24, 24, requires_grad=True)
    sr = generator(lr)
    loss = sr.sum()
    loss.backward()

    assert lr.grad is not None, "Gradients should backpropagate to input LR."
    assert not torch.isnan(lr.grad).any(), "Input gradient contains NaN values."
