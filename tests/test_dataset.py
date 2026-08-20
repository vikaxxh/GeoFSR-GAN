import os
import pytest
import torch
import yaml
from torch.utils.data import DataLoader
from datasets import SatelliteDataset
from scripts.tools.prepare_dataset import generate_synthetic_satellite_image


@pytest.fixture(scope="module")
def temp_dataset_dir(tmp_path_factory):
    dataset_dir = tmp_path_factory.mktemp("synthetic_sat_data")
    for i in range(5):
        img = generate_synthetic_satellite_image(size=(256, 256), seed=10 + i)
        img.save(os.path.join(dataset_dir, f"test_img_{i+1:02d}.png"))
    return str(dataset_dir)


def test_satellite_dataset_train(temp_dataset_dir):
    config = {
        "dataset": {"scale": 4, "hr_patch_size": 96, "subset_size": None},
        "degradation": {"blur_kernel_size": 5, "noise_std_max": 0.02}
    }
    dataset = SatelliteDataset(
        data_dir=temp_dataset_dir,
        scale=4,
        hr_patch_size=96,
        is_train=True,
        config=config,
        seed=42
    )

    assert len(dataset) == 5, f"Expected 5 samples, got {len(dataset)}"
    sample = dataset[0]

    assert "lr" in sample and "hr" in sample and "filename" in sample
    assert sample["hr"].shape == (3, 96, 96), f"Expected HR shape (3, 96, 96), got {sample['hr'].shape}"
    assert sample["lr"].shape == (3, 24, 24), f"Expected LR shape (3, 24, 24), got {sample['lr'].shape}"
    assert sample["hr"].dtype == torch.float32 and sample["lr"].dtype == torch.float32
    assert 0.0 <= sample["hr"].min().item() and sample["hr"].max().item() <= 1.0
    assert 0.0 <= sample["lr"].min().item() and sample["lr"].max().item() <= 1.0


def test_satellite_dataset_val_determinism(temp_dataset_dir):
    config = {
        "dataset": {"scale": 4, "hr_patch_size": 96},
        "degradation": {"blur_kernel_size": 5, "noise_std_max": 0.02}
    }
    dataset_val1 = SatelliteDataset(data_dir=temp_dataset_dir, scale=4, hr_patch_size=96, is_train=False, config=config, seed=42)
    dataset_val2 = SatelliteDataset(data_dir=temp_dataset_dir, scale=4, hr_patch_size=96, is_train=False, config=config, seed=42)

    sample1 = dataset_val1[0]
    sample2 = dataset_val2[0]

    assert torch.allclose(sample1["hr"], sample2["hr"], atol=1e-5), "Validation HR patches must be deterministic."
    assert torch.allclose(sample1["lr"], sample2["lr"], atol=1e-5), "Validation LR patches must be deterministic."


def test_satellite_dataloader(temp_dataset_dir):
    config = {
        "dataset": {"scale": 4, "hr_patch_size": 96},
        "degradation": {"blur_kernel_size": 5}
    }
    dataset = SatelliteDataset(data_dir=temp_dataset_dir, scale=4, hr_patch_size=96, is_train=True, config=config)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(dataloader))
    assert batch["hr"].shape == (2, 3, 96, 96)
    assert batch["lr"].shape == (2, 3, 24, 24)
