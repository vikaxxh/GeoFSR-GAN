import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets import SatelliteDataset
from models import SpatialFrequencySR
from losses import MultiBandFrequencyLoss


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_spatial_frequency(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["project"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cpu")
    print(f"[Spatial+Frequency SR Training] Seed: {seed} | Device: {device}")

    dataset_cfg = config["dataset"]
    train_dataset = SatelliteDataset(
        data_dir=dataset_cfg["data_dir"],
        scale=dataset_cfg["scale"],
        hr_patch_size=dataset_cfg["hr_patch_size"],
        is_train=True,
        config=config
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    model = SpatialFrequencySR(
        scale=dataset_cfg["scale"],
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_blocks=3,
        growth_rate=32,
        res_scale=0.2
    ).to(device)

    l1_loss = nn.L1Loss()
    freq_loss_fn = MultiBandFrequencyLoss(in_channels=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    epochs = 10
    print(f"\n=======================================================================")
    print(f"  Starting Spatial + Frequency SR Training ({epochs} Epochs)  ")
    print(f"=======================================================================\n")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            optimizer.zero_grad()
            sr = model(lr)

            loss_pixel = l1_loss(sr, hr)
            loss_freq, _ = freq_loss_fn(sr, hr)
            total_loss = loss_pixel + 0.5 * loss_freq

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    elapsed = time.time() - start_time
    print(f"\n[Training Complete] Finished {epochs} epochs in {elapsed:.2f}s.")

    ckpt_dir = config["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "spatial_frequency_sr_latest.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Spatial+Frequency SR] Saved trained model to '{ckpt_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()
    train_spatial_frequency(args.config)
