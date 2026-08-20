import os
import sys
import argparse
import yaml
import csv
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import SatelliteDataset
from datasets.degradation import RealisticDegradation
from models import BicubicBaseline, SimpleSpatialSR, GeoFSRGenerator
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


def run_milestone3(config_path):
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
    bicubic = BicubicBaseline(scale=scale).to(device)
    spatial_sr = SimpleSpatialSR(scale=scale, num_features=config["model"]["spatial_encoder"]["num_features"]).to(device)
    geofsr = GeoFSRGenerator(
        scale=scale,
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        num_features=config["model"]["spatial_encoder"]["num_features"],
        num_spatial_blocks=config["model"]["spatial_encoder"]["num_blocks"],
        fusion_type=config["model"]["fusion"]["type"]
    ).to(device)

    # Load weights
    ckpt_dir = config["training"]["checkpoint_dir"]
    spatial_ckpt = os.path.join(ckpt_dir, "spatial_sr_latest.pth")
    if os.path.exists(spatial_ckpt):
        spatial_sr.load_state_dict(torch.load(spatial_ckpt, map_location=device))

    geofsr_ckpt = os.path.join(ckpt_dir, "geofsr_generator_latest.pth")
    if os.path.exists(geofsr_ckpt):
        geofsr.load_state_dict(torch.load(geofsr_ckpt, map_location=device))

    seg_head = get_trained_segmentation_net(device=device, save_path="experiments/segmentation_head.pth", dataset=val_dataset)

    bicubic.eval()
    spatial_sr.eval()
    geofsr.eval()
    seg_head.eval()

    degradation_pipeline = RealisticDegradation(config=config)
    modes = ["D1", "D2", "D3", "D4"]
    models = {
        "Bicubic": bicubic,
        "Spatial SR": spatial_sr,
        "GeoFSR-GAN": geofsr
    }

    print(f"\n=======================================================================")
    print(f"      MILESTONE 3 — Controlled Degradation Pipeline Audit & Ablation    ")
    print(f"=======================================================================\n")

    csv_rows = []

    for mode in modes:
        metrics = {
            m: {"psnr": 0.0, "ssim": 0.0, "miou": 0.0, "dice": 0.0, "bf1": 0.0}
            for m in models.keys()
        }

        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                hr = batch["hr"].to(device)
                gt_mask = generate_ground_truth_mask(hr)

                # Degrade HR with selected mode
                lr = degradation_pipeline.degrade(hr[0], scale=scale, seed=seed + idx, mode=mode).unsqueeze(0).to(device)

                for name, model in models.items():
                    sr = model(lr)
                    metrics[name]["psnr"] += calculate_psnr(sr, hr)
                    metrics[name]["ssim"] += calculate_ssim(sr, hr)

                    pred_mask = torch.sigmoid(seg_head(sr))
                    metrics[name]["miou"] += compute_miou(pred_mask, gt_mask)
                    metrics[name]["dice"] += compute_dice_score(pred_mask, gt_mask)
                    metrics[name]["bf1"] += compute_boundary_f1(pred_mask, gt_mask)

        n = len(val_loader)
        print(f"--- Degradation Strategy {mode} ---")
        print(f"{'Model':<12} | {'PSNR ↑':<8} | {'SSIM ↑':<8} | {'mIoU ↑':<8} | {'Dice ↑':<8} | {'Bound F1 ↑':<10}")
        print("-" * 65)

        for name in models.keys():
            m = metrics[name]
            psnr_avg = round(m["psnr"] / n, 2)
            ssim_avg = round(m["ssim"] / n, 4)
            miou_avg = round(m["miou"] / n, 4)
            dice_avg = round(m["dice"] / n, 4)
            bf1_avg = round(m["bf1"] / n, 4)

            print(f"{name:<12} | {psnr_avg:<8.2f} | {ssim_avg:<8.4f} | {miou_avg:<8.4f} | {dice_avg:<8.4f} | {bf1_avg:<10.4f}")

            csv_rows.append({
                "degradation_mode": mode,
                "model": name,
                "psnr": psnr_avg,
                "ssim": ssim_avg,
                "miou": miou_avg,
                "dice": dice_avg,
                "boundary_f1": bf1_avg
            })
        print("-" * 65 + "\n")

    # Save to experiments/degradation_ablation.csv
    csv_path = "experiments/degradation_ablation.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["degradation_mode", "model", "psnr", "ssim", "miou", "dice", "boundary_f1"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[Milestone 3] Degradation ablation report saved to '{csv_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()
    run_milestone3(args.config)
