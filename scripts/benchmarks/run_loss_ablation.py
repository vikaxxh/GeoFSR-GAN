import os
import sys
import time
import argparse
import yaml
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets import SatelliteDataset
from models import (
    SpatialFrequencySR,
    SpatialPatchGANDiscriminator,
    FrequencyPatchGANDiscriminator
)
from models.frequency_encoder import DWT2D
from losses import (
    SobelEdgeLoss,
    MultiBandFrequencyLoss,
    PerceptualLoss,
    DualDomainAdversarialLoss
)
from evaluation import (
    calculate_psnr,
    calculate_ssim,
    calculate_lpips,
    compute_miou,
    compute_dice_score,
    compute_boundary_f1,
    generate_ground_truth_mask,
    get_trained_segmentation_net
)


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_eval_loss_config(config_id, loss_name, config, train_loader, val_loader, seg_head, device):
    print(f"\n---> Training Loss Config {config_id}: '{loss_name}' (10 Epochs)")

    # Generator model with Lightweight Cross-Attention
    gen = SpatialFrequencySR(
        scale=4,
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_blocks=3,
        growth_rate=32,
        res_scale=0.2
    ).to(device)

    l1_loss = nn.L1Loss()
    edge_loss_fn = SobelEdgeLoss().to(device)
    freq_loss_fn = MultiBandFrequencyLoss(in_channels=3).to(device)
    perceptual_loss_fn = PerceptualLoss(mode="lightweight").to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.9, 0.999))
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=10, eta_min=1e-5)

    if config_id == 4:
        disc_spat = SpatialPatchGANDiscriminator(in_channels=3).to(device)
        disc_freq = FrequencyPatchGANDiscriminator(in_channels=3).to(device)
        dwt_func = DWT2D(in_channels=3).to(device)
        adv_loss_fn = DualDomainAdversarialLoss(gan_type="lsgan", lambda_spatial=0.005, lambda_freq=0.005).to(device)

        opt_d = torch.optim.Adam(
            list(disc_spat.parameters()) + list(disc_freq.parameters()),
            lr=2e-4, betas=(0.9, 0.999)
        )
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=10, eta_min=1e-5)
    else:
        disc_spat, disc_freq, opt_d, sched_d, adv_loss_fn = None, None, None, None, None

    gen.train()
    for epoch in range(1, 11):
        for batch in train_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            opt_g.zero_grad()
            sr = gen(lr)

            # Compute loss components
            loss_g = l1_loss(sr, hr)

            if config_id >= 2:
                l_edge = edge_loss_fn(sr, hr)
                l_freq, _ = freq_loss_fn(sr, hr)
                loss_g = loss_g + 0.1 * l_edge + 0.5 * l_freq

            if config_id >= 3:
                l_perc = perceptual_loss_fn(sr, hr)
                loss_g = loss_g + 0.1 * l_perc

            if config_id == 4:
                # Train Discriminators
                opt_d.zero_grad()
                d_spat_real = disc_spat(hr)
                d_spat_fake = disc_spat(sr.detach())

                hr_ll, hr_lh, hr_hl, hr_hh = dwt_func(hr)
                sr_ll, sr_lh, sr_hl, sr_hh = dwt_func(sr.detach())

                hr_dwt_cat = torch.cat([hr_ll, hr_lh, hr_hl, hr_hh], dim=1)
                sr_dwt_cat = torch.cat([sr_ll, sr_lh, sr_hl, sr_hh], dim=1)

                d_freq_real = disc_freq(hr_dwt_cat)
                d_freq_fake = disc_freq(sr_dwt_cat)

                loss_d, _ = adv_loss_fn.forward_d(d_spat_real, d_spat_fake, d_freq_real, d_freq_fake)
                loss_d.backward()
                opt_d.step()

                # Generator Adversarial Loss
                d_spat_fake_g = disc_spat(sr)
                sr_ll_g, sr_lh_g, sr_hl_g, sr_hh_g = dwt_func(sr)
                sr_dwt_cat_g = torch.cat([sr_ll_g, sr_lh_g, sr_hl_g, sr_hh_g], dim=1)
                d_freq_fake_g = disc_freq(sr_dwt_cat_g)

                loss_g_adv, _ = adv_loss_fn.forward_g(
                    d_spatial_real=d_spat_real.detach(),
                    d_spatial_fake=d_spat_fake_g,
                    d_freq_real=d_freq_real.detach(),
                    d_freq_fake=d_freq_fake_g
                )
                loss_g = loss_g + loss_g_adv

            loss_g.backward()
            opt_g.step()

        sched_g.step()
        if sched_d is not None:
            sched_d.step()

    # Evaluation
    gen.eval()
    n_samples = len(val_loader)
    m_psnr, m_ssim, m_lpips, m_miou, m_dice, m_bf1, m_perc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total_time = 0.0

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            gt_mask = generate_ground_truth_mask(hr)

            t0 = time.time()
            sr = gen(lr)
            t1 = time.time()
            total_time += (t1 - t0) * 1000.0

            m_psnr += calculate_psnr(sr, hr)
            m_ssim += calculate_ssim(sr, hr)
            lp = calculate_lpips(sr, hr)
            m_lpips += (lp if lp is not None else 0.0)

            pred_mask = torch.sigmoid(seg_head(sr))
            m_miou += compute_miou(pred_mask, gt_mask)
            m_dice += compute_dice_score(pred_mask, gt_mask)
            m_bf1 += compute_boundary_f1(pred_mask, gt_mask)

            m_perc += perceptual_loss_fn(sr, hr).item()

    avg_latency = total_time / n_samples
    res = {
        "config_id": f"Config {config_id}",
        "loss_name": loss_name,
        "psnr": round(m_psnr / n_samples, 2),
        "ssim": round(m_ssim / n_samples, 4),
        "lpips": round(m_lpips / n_samples, 4),
        "miou": round(m_miou / n_samples, 4),
        "dice": round(m_dice / n_samples, 4),
        "boundary_f1": round(m_bf1 / n_samples, 4),
        "perceptual_score": round(m_perc / n_samples, 5),
        "latency_ms": round(avg_latency, 2)
    }

    ckpt_dir = config["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"loss_config_{config_id}_latest.pth")
    torch.save(gen.state_dict(), ckpt_path)

    return res


def run_milestone7(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["project"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cpu")

    dataset_cfg = config["dataset"]
    train_dataset = SatelliteDataset(data_dir=dataset_cfg["data_dir"], scale=4, hr_patch_size=96, is_train=True, config=config)
    val_dataset = SatelliteDataset(data_dir=dataset_cfg["data_dir"], scale=4, hr_patch_size=96, is_train=False, config=config)

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    seg_head = get_trained_segmentation_net(device=device, save_path="experiments/segmentation_head.pth", dataset=val_dataset)
    seg_head.eval()

    loss_configs = [
        (1, "Pure L1 Pixel Loss"),
        (2, "L1 + L_edge + L_freq"),
        (3, "L1 + L_edge + L_freq + L_perceptual"),
        (4, "L1 + L_edge + L_freq + L_perceptual + L_adv (Full GAN)")
    ]

    results = []

    print(f"\n=======================================================================")
    print(f"      MILESTONE 7 — Adversarial & Multi-Loss Function Ablation          ")
    print(f"=======================================================================\n")

    for cid, name in loss_configs:
        res = train_and_eval_loss_config(cid, name, config, train_loader, val_loader, seg_head, device)
        results.append(res)

    print("\n--- Milestone 7 Quantitative Loss Ablation Summary ---")
    print(f"{'Config ID':<10} | {'Loss Formulation':<45} | {'PSNR ↑':<8} | {'SSIM ↑':<8} | {'mIoU ↑':<8} | {'Dice ↑':<8} | {'Bound F1 ↑':<10} | {'Perc Score ↓':<12}")
    print("-" * 125)

    for r in results:
        print(f"{r['config_id']:<10} | {r['loss_name']:<45} | {r['psnr']:<8.2f} | {r['ssim']:<8.4f} | {r['miou']:<8.4f} | {r['dice']:<8.4f} | {r['boundary_f1']:<10.4f} | {r['perceptual_score']:<12.5f}")

    print("-" * 125)

    # Save to experiments/loss_ablation.csv
    csv_path = "experiments/loss_ablation.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["config_id", "loss_name", "psnr", "ssim", "lpips", "miou", "dice", "boundary_f1", "perceptual_score", "latency_ms"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[Milestone 7] Loss ablation report saved to '{csv_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()
    run_milestone7(args.config)
