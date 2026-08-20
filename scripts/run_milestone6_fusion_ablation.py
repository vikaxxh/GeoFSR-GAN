import os
import sys
import time
import argparse
import yaml
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import SatelliteDataset
from models import SpatialFrequencySR, SpatialFrequencyFusion
from losses import MultiBandFrequencyLoss
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GeoFSRFusionModel(nn.Module):
    """
    Wrapper for GeoFSR model enabling dynamic selection of fusion mechanism.
    """
    def __init__(self, scale=4, num_features=64, fusion_type="concat"):
        super().__init__()
        self.sr_net = SpatialFrequencySR(
            scale=scale,
            in_channels=3,
            out_channels=3,
            num_features=num_features,
            num_blocks=3,
            growth_rate=32,
            res_scale=0.2
        )
        self.sr_net.fusion = SpatialFrequencyFusion(num_features=num_features, fusion_type=fusion_type)

    def forward(self, x):
        return self.sr_net(x)


def train_and_eval_fusion(fusion_type, config, train_loader, val_loader, seg_head, device):
    print(f"\n---> Training Fusion Strategy: '{fusion_type}' (10 Epochs)")

    model = GeoFSRFusionModel(scale=4, num_features=64, fusion_type=fusion_type).to(device)
    params = count_parameters(model)

    l1_loss = nn.L1Loss()
    freq_loss_fn = MultiBandFrequencyLoss(in_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    model.train()
    for epoch in range(1, 11):
        for batch in train_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            optimizer.zero_grad()
            sr = model(lr)

            l_pix = l1_loss(sr, hr)
            l_freq, _ = freq_loss_fn(sr, hr)
            loss = l_pix + 0.5 * l_freq

            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluation
    model.eval()
    n_samples = len(val_loader)
    m_psnr, m_ssim, m_lpips, m_miou, m_dice, m_bf1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total_time = 0.0

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            gt_mask = generate_ground_truth_mask(hr)

            t0 = time.time()
            sr = model(lr)
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

    avg_latency = total_time / n_samples
    res = {
        "fusion_type": fusion_type,
        "params": params,
        "psnr": round(m_psnr / n_samples, 2),
        "ssim": round(m_ssim / n_samples, 4),
        "lpips": round(m_lpips / n_samples, 4),
        "miou": round(m_miou / n_samples, 4),
        "dice": round(m_dice / n_samples, 4),
        "boundary_f1": round(m_bf1 / n_samples, 4),
        "latency_ms": round(avg_latency, 2)
    }

    # Save model checkpoint
    ckpt_dir = config["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"fusion_{fusion_type}_latest.pth")
    torch.save(model.state_dict(), ckpt_path)

    return res


def run_milestone6(config_path):
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

    fusion_types = ["concat", "learned", "cross_attention", "dual_cross_attention"]
    results = []

    print(f"\n=======================================================================")
    print(f"      MILESTONE 6 — Spatial-Frequency Fusion Mechanism Controlled Ablation ")
    print(f"=======================================================================\n")

    for f_type in fusion_types:
        res = train_and_eval_fusion(f_type, config, train_loader, val_loader, seg_head, device)
        results.append(res)

    print("\n--- Milestone 6 Quantitative Fusion Comparison Summary ---")
    print(f"{'Fusion Mechanism':<24} | {'Params':<8} | {'PSNR ↑':<8} | {'SSIM ↑':<8} | {'mIoU ↑':<8} | {'Dice ↑':<8} | {'Bound F1 ↑':<10} | {'Latency (ms)':<12}")
    print("-" * 115)

    for r in results:
        print(f"{r['fusion_type']:<24} | {r['params']:<8} | {r['psnr']:<8.2f} | {r['ssim']:<8.4f} | {r['miou']:<8.4f} | {r['dice']:<8.4f} | {r['boundary_f1']:<10.4f} | {r['latency_ms']:<12.2f}")

    print("-" * 115)

    # Save to experiments/fusion_ablation.csv
    csv_path = "experiments/fusion_ablation.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["fusion_type", "params", "psnr", "ssim", "lpips", "miou", "dice", "boundary_f1", "latency_ms"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[Milestone 6] Fusion ablation report saved to '{csv_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()
    run_milestone6(args.config)
