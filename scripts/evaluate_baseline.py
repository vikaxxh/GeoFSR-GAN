import os
import sys
import json
import csv
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import SatelliteDataset
from models import BicubicBaseline, SimpleSpatialSR
from evaluation import calculate_psnr, calculate_ssim, calculate_lpips, save_baseline_comparison


def evaluate_baselines(config_path, exp_dir="experiments/baseline"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(exp_dir, exist_ok=True)
    data_dir = config["dataset"]["data_dir"]
    scale = config["dataset"]["scale"]
    hr_patch_size = config["dataset"]["hr_patch_size"]

    print(f"[Evaluate Baseline] Loading dataset from '{data_dir}'...")
    val_dataset = SatelliteDataset(data_dir=data_dir, scale=scale, hr_patch_size=hr_patch_size, is_train=False, config=config)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 1. Instantiate Models
    bicubic_model = BicubicBaseline(scale=scale)
    spatial_sr_model = SimpleSpatialSR(
        scale=scale,
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        num_features=config["model"]["spatial_encoder"]["num_features"]
    )

    checkpoint_path = os.path.join(config["training"]["checkpoint_dir"], "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"[Evaluate Baseline] Loading trained weights from '{checkpoint_path}'...")
        spatial_sr_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    else:
        print("[Evaluate Baseline] Warning: Trained checkpoint not found. Evaluating initialized model...")

    bicubic_model.eval()
    spatial_sr_model.eval()

    bicubic_metrics = {"psnr": [], "ssim": [], "lpips": []}
    spatial_sr_metrics = {"psnr": [], "ssim": [], "lpips": []}

    first_batch_sample = None

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            lr = batch["lr"]
            hr = batch["hr"]

            # Model outputs
            sr_bicubic = bicubic_model(lr)
            sr_spatial = spatial_sr_model(lr)

            if i == 0:
                first_batch_sample = (lr, sr_bicubic, sr_spatial, hr)

            # Compute Bicubic Metrics
            b_psnr = calculate_psnr(sr_bicubic, hr)
            b_ssim = calculate_ssim(sr_bicubic, hr)
            b_lpips = calculate_lpips(sr_bicubic, hr)
            bicubic_metrics["psnr"].append(b_psnr)
            bicubic_metrics["ssim"].append(b_ssim)
            if b_lpips is not None:
                bicubic_metrics["lpips"].append(b_lpips)

            # Compute Spatial SR Metrics
            s_psnr = calculate_psnr(sr_spatial, hr)
            s_ssim = calculate_ssim(sr_spatial, hr)
            s_lpips = calculate_lpips(sr_spatial, hr)
            spatial_sr_metrics["psnr"].append(s_psnr)
            spatial_sr_metrics["ssim"].append(s_ssim)
            if s_lpips is not None:
                spatial_sr_metrics["lpips"].append(s_lpips)

    # Compute Averages
    avg_results = {
        "Bicubic": {
            "PSNR": float(np.mean(bicubic_metrics["psnr"])),
            "SSIM": float(np.mean(bicubic_metrics["ssim"])),
            "LPIPS": float(np.mean(bicubic_metrics["lpips"])) if bicubic_metrics["lpips"] else "N/A"
        },
        "Spatial_SR_Baseline": {
            "PSNR": float(np.mean(spatial_sr_metrics["psnr"])),
            "SSIM": float(np.mean(spatial_sr_metrics["ssim"])),
            "LPIPS": float(np.mean(spatial_sr_metrics["lpips"])) if spatial_sr_metrics["lpips"] else "N/A"
        }
    }

    print("\n================ BASELINE EVALUATION REPORT ================")
    for model_name, metrics in avg_results.items():
        print(f"Model: {model_name:20s} | PSNR: {metrics['PSNR']:.2f} dB | SSIM: {metrics['SSIM']:.4f} | LPIPS: {metrics['LPIPS']}")
    print("===========================================================")

    # Save metrics JSON
    json_path = os.path.join(exp_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(avg_results, f, indent=4)
    print(f"[Evaluate Baseline] Saved metrics JSON to '{json_path}'.")

    # Save metrics CSV
    csv_path = os.path.join(exp_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "PSNR", "SSIM", "LPIPS"])
        for model_name, metrics in avg_results.items():
            writer.writerow([model_name, f"{metrics['PSNR']:.4f}", f"{metrics['SSIM']:.4f}", metrics['LPIPS']])
    print(f"[Evaluate Baseline] Saved metrics CSV to '{csv_path}'.")

    # Save Visual Comparison Grid
    if first_batch_sample is not None:
        vis_path = os.path.join(exp_dir, "visual_comparison.png")
        lr, sr_b, sr_s, hr = first_batch_sample
        save_baseline_comparison(lr[0], sr_b[0], sr_s[0], hr[0], save_path=vis_path, sample_idx=1)

    return avg_results


if __name__ == "__main__":
    import numpy as np
    parser = argparse.ArgumentParser(description="Evaluate GeoFSR-GAN Baselines.")
    parser.add_argument("--config", type=str, default="configs/cpu_debug.yaml", help="Path to config file.")
    parser.add_argument("--exp_dir", type=str, default="experiments/baseline", help="Output directory for results.")
    args = parser.parse_args()
    evaluate_baselines(args.config, exp_dir=args.exp_dir)
