import os
import sys
import argparse
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import SatelliteDataset
from models import SimpleSpatialSR
from training import BaselineTrainer


def main():
    parser = argparse.ArgumentParser(description="Train SimpleSpatialSR Baseline Model.")
    parser.add_argument("--config", type=str, default="configs/cpu_debug.yaml", help="Path to config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    data_dir = config["dataset"]["data_dir"]
    scale = config["dataset"]["scale"]
    hr_patch_size = config["dataset"]["hr_patch_size"]

    print(f"[Train Baseline] Initializing dataset from '{data_dir}'...")
    train_dataset = SatelliteDataset(data_dir=data_dir, scale=scale, hr_patch_size=hr_patch_size, is_train=True, config=config)
    val_dataset = SatelliteDataset(data_dir=data_dir, scale=scale, hr_patch_size=hr_patch_size, is_train=False, config=config)

    model = SimpleSpatialSR(
        scale=scale,
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        num_features=config["model"]["spatial_encoder"]["num_features"]
    )

    trainer = BaselineTrainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, config=config)
    trainer.train()


if __name__ == "__main__":
    main()
