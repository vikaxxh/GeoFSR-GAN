import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets import SatelliteDataset
from models import SimpleSpatialSR
from training.trainer import BaselineTrainer


def train_spatial_baseline(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")
    print(f"[Spatial SR Training] Device: {device}")

    dataset_cfg = config["dataset"]
    train_dataset = SatelliteDataset(
        data_dir=dataset_cfg["data_dir"],
        scale=dataset_cfg["scale"],
        hr_patch_size=dataset_cfg["hr_patch_size"],
        is_train=True,
        config=config
    )
    val_dataset = SatelliteDataset(
        data_dir=dataset_cfg["data_dir"],
        scale=dataset_cfg["scale"],
        hr_patch_size=dataset_cfg["hr_patch_size"],
        is_train=False,
        config=config
    )

    model = SimpleSpatialSR(
        scale=dataset_cfg["scale"],
        num_features=config["model"]["spatial_encoder"]["num_features"]
    )

    trainer = BaselineTrainer(model, train_dataset, val_dataset, config)
    trainer.train()

    ckpt_dir = config["training"]["checkpoint_dir"]
    spatial_ckpt = os.path.join(ckpt_dir, "spatial_sr_latest.pth")
    torch.save(model.state_dict(), spatial_ckpt)
    print(f"[Spatial SR] Saved trained spatial baseline to '{spatial_ckpt}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()
    train_spatial_baseline(args.config)
