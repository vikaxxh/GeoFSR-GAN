import os
import sys
import glob
import json
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.train_geofsr import train_geofsr
from datasets import SatelliteDataset
from models import GeoFSRGenerator, LightweightSegmentationUNet
from evaluation import calculate_psnr, calculate_ssim, compute_miou


def evaluate_ablation_checkpoint(config_path, checkpoint_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")

    val_dataset = SatelliteDataset(
        data_dir=config["dataset"]["data_dir"],
        scale=config["dataset"]["scale"],
        hr_patch_size=config["dataset"]["hr_patch_size"],
        is_train=False,
        config=config
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model_cfg = config["model"]
    generator = GeoFSRGenerator(
        scale=model_cfg["scale"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        num_features=model_cfg["spatial_encoder"]["num_features"],
        num_spatial_blocks=model_cfg["spatial_encoder"]["num_blocks"],
        fusion_type=model_cfg["fusion"]["type"]
    ).to(device)

    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()

    seg_head = LightweightSegmentationUNet(in_channels=3, num_classes=1, num_features=16).to(device)
    seg_head.eval()

    psnr_total, ssim_total, miou_total = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = generator(lr)

            psnr_total += calculate_psnr(sr, hr)
            ssim_total += calculate_ssim(sr, hr)

            mask_sr = torch.sigmoid(seg_head(sr))
            mask_hr = torch.sigmoid(seg_head(hr))
            miou_total += compute_miou(mask_sr, mask_hr)

    n = len(val_loader)
    return {
        "psnr": float(psnr_total / n),
        "ssim": float(ssim_total / n),
        "miou": float(miou_total / n)
    }


def run_all_ablations():
    ablation_dir = "configs/ablations"
    config_files = sorted(glob.glob(os.path.join(ablation_dir, "*.yaml")))
    
    # Include baseline full model
    config_files.append("configs/cpu_debug.yaml")

    results = {}

    print("\n=======================================================")
    print("      GeoFSR-GAN Automated Ablation Experiment Suite   ")
    print("=======================================================\n")

    for cfg_path in config_files:
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        print(f"\n[Ablation Variant] Running training for '{name}' ({cfg_path})...")

        ckpt_path = train_geofsr(cfg_path)
        metrics = evaluate_ablation_checkpoint(cfg_path, ckpt_path)

        results[name] = metrics
        print(f"[Results: {name}] PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f} | mIoU: {metrics['miou']:.4f}")

    # Output JSON & Markdown summary
    out_dir = "experiments"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "ablation_study_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    md_path = os.path.join(out_dir, "ablation_summary.md")
    with open(md_path, "w") as f:
        f.write("# GeoFSR-GAN Ablation Study Summary\n\n")
        f.write("| Variant | Configuration | PSNR (dB) | SSIM | Segmentation mIoU |\n")
        f.write("|---|---|:---:|:---:|:---:|\n")
        for variant, m in results.items():
            f.write(f"| **{variant}** | `{variant}.yaml` | {m['psnr']:.2f} | {m['ssim']:.4f} | {m['miou']:.4f} |\n")

    print(f"\n[Ablation Study Completed] Results saved to '{json_path}' and '{md_path}'.")


if __name__ == "__main__":
    run_all_ablations()
