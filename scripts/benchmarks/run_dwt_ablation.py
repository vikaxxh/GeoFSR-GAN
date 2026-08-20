import os
import sys
import argparse
import yaml
import csv
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets import SatelliteDataset
from models import ImprovedSpatialSR, SpatialFrequencySR
from losses import SobelEdgeLoss, MultiBandFrequencyLoss
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


def run_milestone5(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["project"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cpu")

    # Load dataset
    dataset_cfg = config["dataset"]
    val_dataset = SatelliteDataset(
        data_dir=dataset_cfg["data_dir"],
        scale=dataset_cfg["scale"],
        hr_patch_size=dataset_cfg["hr_patch_size"],
        is_train=False,
        config=config
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    scale = dataset_cfg["scale"]
    model_a = ImprovedSpatialSR(scale=scale, num_features=64, num_blocks=3, growth_rate=32, res_scale=0.2).to(device)
    model_b = SpatialFrequencySR(scale=scale, num_features=64, num_blocks=3, growth_rate=32, res_scale=0.2).to(device)

    ckpt_dir = config["training"]["checkpoint_dir"]
    spat_ckpt = os.path.join(ckpt_dir, "improved_spatial_sr_latest.pth")
    if os.path.exists(spat_ckpt):
        model_a.load_state_dict(torch.load(spat_ckpt, map_location=device))

    freq_ckpt = os.path.join(ckpt_dir, "spatial_frequency_sr_latest.pth")
    if os.path.exists(freq_ckpt):
        model_b.load_state_dict(torch.load(freq_ckpt, map_location=device))

    seg_head = get_trained_segmentation_net(device=device, save_path="experiments/segmentation_head.pth", dataset=val_dataset)
    edge_loss_fn = SobelEdgeLoss().to(device)
    freq_loss_fn = MultiBandFrequencyLoss(in_channels=3).to(device)

    model_a.eval()
    model_b.eval()
    seg_head.eval()

    models = {
        "Model A (Spatial SR Only)": model_a,
        "Model B (Spatial + Frequency DWT)": model_b
    }

    metrics = {
        name: {
            "psnr": 0.0, "ssim": 0.0, "lpips": 0.0,
            "miou": 0.0, "dice": 0.0, "bf1": 0.0,
            "edge_loss": 0.0, "freq_loss": 0.0
        } for name in models.keys()
    }

    print(f"\n=======================================================================")
    print(f"      MILESTONE 5 — Frequency Branch (DWT) Validation & Controlled Ablation ")
    print(f"=======================================================================\n")

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            gt_mask = generate_ground_truth_mask(hr)

            for name, model in models.items():
                sr = model(lr)

                metrics[name]["psnr"] += calculate_psnr(sr, hr)
                metrics[name]["ssim"] += calculate_ssim(sr, hr)

                lp_score = calculate_lpips(sr, hr)
                metrics[name]["lpips"] += (lp_score if lp_score is not None else 0.0)

                pred_mask = torch.sigmoid(seg_head(sr))
                metrics[name]["miou"] += compute_miou(pred_mask, gt_mask)
                metrics[name]["dice"] += compute_dice_score(pred_mask, gt_mask)
                metrics[name]["bf1"] += compute_boundary_f1(pred_mask, gt_mask)

                e_loss = edge_loss_fn(sr, hr).item()
                f_loss, _ = freq_loss_fn(sr, hr)
                metrics[name]["edge_loss"] += e_loss
                metrics[name]["freq_loss"] += f_loss.item()

    n = len(val_loader)
    print(f"{'Model':<35} | {'PSNR ↑':<8} | {'SSIM ↑':<8} | {'mIoU ↑':<8} | {'Dice ↑':<8} | {'Bound F1 ↑':<10} | {'Edge Loss ↓':<12} | {'Freq L1 Error ↓':<15}")
    print("-" * 120)

    csv_rows = []
    for name, m in metrics.items():
        row = {
            "model": name,
            "psnr": round(m["psnr"] / n, 2),
            "ssim": round(m["ssim"] / n, 4),
            "lpips": round(m["lpips"] / n, 4),
            "miou": round(m["miou"] / n, 4),
            "dice": round(m["dice"] / n, 4),
            "boundary_f1": round(m["bf1"] / n, 4),
            "sobel_edge_loss": round(m["edge_loss"] / n, 5),
            "dwt_subband_l1_error": round(m["freq_loss"] / n, 5)
        }
        csv_rows.append(row)
        print(f"{row['model']:<35} | {row['psnr']:<8.2f} | {row['ssim']:<8.4f} | {row['miou']:<8.4f} | {row['dice']:<8.4f} | {row['boundary_f1']:<10.4f} | {row['sobel_edge_loss']:<12.5f} | {row['dwt_subband_l1_error']:<15.5f}")

    print("-" * 120)

    # Save to experiments/frequency_ablation.csv
    csv_path = "experiments/frequency_ablation.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["model", "psnr", "ssim", "lpips", "miou", "dice", "boundary_f1", "sobel_edge_loss", "dwt_subband_l1_error"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[Milestone 5] Frequency branch ablation report saved to '{csv_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()
    run_milestone5(args.config)
