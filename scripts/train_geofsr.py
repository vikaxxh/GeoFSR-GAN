import os
import sys
import argparse
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import SatelliteDataset
from models import (
    GeoFSRGenerator,
    SpatialPatchGANDiscriminator,
    FrequencyPatchGANDiscriminator,
    LightweightSegmentationUNet
)
from losses import (
    MultiBandFrequencyLoss,
    SobelEdgeLoss,
    PerceptualLoss,
    DualDomainAdversarialLoss,
    DownstreamSegmentationLoss
)
from evaluation import calculate_psnr, calculate_ssim, compute_miou


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_geofsr(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["project"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and config["project"]["device"] == "cuda" else "cpu")
    print(f"[GeoFSR Training] Fixed Random Seed: {seed} | Device: {device}")

    # 1. Dataset & DataLoaders
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

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 2. Models
    model_cfg = config["model"]
    generator = GeoFSRGenerator(
        scale=model_cfg["scale"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        num_features=model_cfg["spatial_encoder"]["num_features"],
        num_spatial_blocks=model_cfg["spatial_encoder"]["num_blocks"],
        fusion_type=model_cfg["fusion"]["type"]
    ).to(device)

    disc_spatial = SpatialPatchGANDiscriminator(in_channels=3, num_features=32, num_layers=2).to(device)
    disc_freq = FrequencyPatchGANDiscriminator(in_channels=3, num_features=32, num_layers=2).to(device)
    seg_head = LightweightSegmentationUNet(in_channels=3, num_classes=1, num_features=16).to(device)

    # 3. Loss Modules
    loss_cfg = config["losses"]
    criterion_pixel = nn.L1Loss()
    criterion_freq = MultiBandFrequencyLoss(subband_weights=loss_cfg.get("frequency_subband_weights")).to(device)
    criterion_edge = SobelEdgeLoss().to(device)
    criterion_perceptual = PerceptualLoss(mode=loss_cfg.get("perceptual_mode", "lightweight")).to(device)
    criterion_adv = DualDomainAdversarialLoss(
        gan_type="lsgan",
        lambda_spatial=loss_cfg.get("lambda_adv_spatial", 0.005),
        lambda_freq=loss_cfg.get("lambda_adv_frequency", 0.005)
    ).to(device)
    criterion_seg = DownstreamSegmentationLoss(seg_net=seg_head, freeze_seg_net=True).to(device)

    # 4. Optimizers
    train_cfg = config["training"]
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=train_cfg["lr"], betas=tuple(train_cfg["betas"]))
    optimizer_D = torch.optim.Adam(
        list(disc_spatial.parameters()) + list(disc_freq.parameters()),
        lr=train_cfg["lr"],
        betas=tuple(train_cfg["betas"])
    )

    # 5. Output directories
    ckpt_dir = train_cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # 6. Training Loop
    epochs = train_cfg["epochs"]
    print(f"\n=======================================================")
    print(f"   Starting GeoFSR-GAN Training ({epochs} Epochs)    ")
    print(f"=======================================================\n")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        generator.train()
        disc_spatial.train()
        disc_freq.train()

        running_g_loss = 0.0
        running_d_loss = 0.0

        for i, batch in enumerate(train_loader):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            # ------------------------------------
            # Train Discriminators
            # ------------------------------------
            optimizer_D.zero_grad()
            with torch.no_grad():
                sr = generator(lr)

            d_spat_real = disc_spatial(hr)
            d_spat_fake = disc_spatial(sr)
            d_freq_real = disc_freq(hr)
            d_freq_fake = disc_freq(sr)

            loss_D, d_dict = criterion_adv.forward_d(d_spat_real, d_spat_fake, d_freq_real, d_freq_fake)
            loss_D.backward()
            optimizer_D.step()

            # ------------------------------------
            # Train Generator
            # ------------------------------------
            optimizer_G.zero_grad()
            sr = generator(lr)

            # Spatial & Frequency predictions for G step
            d_spat_real_g = disc_spatial(hr)
            d_spat_fake_g = disc_spatial(sr)
            d_freq_real_g = disc_freq(hr)
            d_freq_fake_g = disc_freq(sr)

            l_pixel = criterion_pixel(sr, hr)
            l_freq, _ = criterion_freq(sr, hr)
            l_edge = criterion_edge(sr, hr)
            l_perceptual = criterion_perceptual(sr, hr)
            l_adv_g, g_adv_dict = criterion_adv.forward_g(d_spat_real_g, d_spat_fake_g, d_freq_real_g, d_freq_fake_g)
            l_seg = criterion_seg(sr, hr)

            loss_G = (
                loss_cfg["lambda_pixel"] * l_pixel +
                loss_cfg["lambda_freq"] * l_freq +
                loss_cfg["lambda_edge"] * l_edge +
                loss_cfg["lambda_perceptual"] * l_perceptual +
                l_adv_g +
                loss_cfg.get("lambda_seg", 0.0) * l_seg
            )

            loss_G.backward()
            optimizer_G.step()

            running_g_loss += loss_G.item()
            running_d_loss += loss_D.item()

        avg_g_loss = running_g_loss / len(train_loader)
        avg_d_loss = running_d_loss / len(train_loader)

        # 7. Validation Evaluation
        generator.eval()
        val_psnr, val_ssim, val_miou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                v_lr = val_batch["lr"].to(device)
                v_hr = val_batch["hr"].to(device)
                v_sr = generator(v_lr)

                val_psnr += calculate_psnr(v_sr, v_hr)
                val_ssim += calculate_ssim(v_sr, v_hr)

                # Segmentation mIoU
                v_mask_sr = torch.sigmoid(seg_head(v_sr))
                v_mask_hr = torch.sigmoid(seg_head(v_hr))
                val_miou += compute_miou(v_mask_sr, v_mask_hr)

        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        val_miou /= len(val_loader)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Loss G: {avg_g_loss:.4f} | Loss D: {avg_d_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | Val mIoU: {val_miou:.4f}")

    elapsed = time.time() - start_time
    print(f"\n[Training Complete] Finished {epochs} epochs in {elapsed:.2f}s.")

    # Save final model
    ckpt_path = os.path.join(ckpt_dir, "geofsr_generator_latest.pth")
    torch.save(generator.state_dict(), ckpt_path)
    print(f"[Checkpoint Saved] Generator weights saved to '{ckpt_path}'.")

    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GeoFSR-GAN Model.")
    parser.add_argument("--config", type=str, default="configs/cpu_debug.yaml", help="Path to YAML config file.")
    args = parser.parse_args()
    train_geofsr(args.config)
