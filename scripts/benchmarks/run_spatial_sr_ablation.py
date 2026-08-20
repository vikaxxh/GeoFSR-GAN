import os
import sys
import time
import argparse
import yaml
import csv
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets import SatelliteDataset
from models import BicubicBaseline, SimpleSpatialSR, ImprovedSpatialSR
from evaluation import (
    calculate_psnr,
    calculate_ssim,
    calculate_lpips,
    compute_miou,
    compute_dice_score,
    compute_precision_recall,
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


def run_milestone4(config_path):
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
    initial_spatial = SimpleSpatialSR(scale=scale, num_features=config["model"]["spatial_encoder"]["num_features"]).to(device)
    improved_spatial = ImprovedSpatialSR(scale=scale, num_features=64, num_blocks=3, growth_rate=32, res_scale=0.2).to(device)

    # Load checkpoints
    ckpt_dir = config["training"]["checkpoint_dir"]
    spatial_ckpt = os.path.join(ckpt_dir, "spatial_sr_latest.pth")
    if os.path.exists(spatial_ckpt):
        initial_spatial.load_state_dict(torch.load(spatial_ckpt, map_location=device))

    improved_ckpt = os.path.join(ckpt_dir, "improved_spatial_sr_latest.pth")
    if os.path.exists(improved_ckpt):
        improved_spatial.load_state_dict(torch.load(improved_ckpt, map_location=device))
        print(f"[Milestone 4] Loaded Improved Spatial SR checkpoint from '{improved_ckpt}'.")

    seg_head = get_trained_segmentation_net(device=device, save_path="experiments/segmentation_head.pth", dataset=val_dataset)

    bicubic.eval()
    initial_spatial.eval()
    improved_spatial.eval()
    seg_head.eval()

    models = {
        "Bicubic Baseline": bicubic,
        "Initial Spatial SR": initial_spatial,
        "Improved Spatial SR": improved_spatial
    }

    metrics = {
        name: {
            "psnr": 0.0, "ssim": 0.0, "lpips": 0.0,
            "miou": 0.0, "dice": 0.0, "prec": 0.0, "rec": 0.0, "bf1": 0.0,
            "latency_ms": 0.0, "params": count_parameters(model)
        } for name, model in models.items()
    }

    print(f"\n=======================================================================")
    print(f"      MILESTONE 4 — Spatial SR Architecture Improvement Evaluation      ")
    print(f"=======================================================================\n")

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            gt_mask = generate_ground_truth_mask(hr)

            for name, model in models.items():
                t0 = time.time()
                sr = model(lr)
                t1 = time.time()

                metrics[name]["latency_ms"] += (t1 - t0) * 1000.0

                metrics[name]["psnr"] += calculate_psnr(sr, hr)
                metrics[name]["ssim"] += calculate_ssim(sr, hr)

                lp_score = calculate_lpips(sr, hr)
                metrics[name]["lpips"] += (lp_score if lp_score is not None else 0.0)

                pred_mask = torch.sigmoid(seg_head(sr))
                metrics[name]["miou"] += compute_miou(pred_mask, gt_mask)
                metrics[name]["dice"] += compute_dice_score(pred_mask, gt_mask)
                prec, rec = compute_precision_recall(pred_mask, gt_mask)
                metrics[name]["prec"] += prec
                metrics[name]["rec"] += rec
                metrics[name]["bf1"] += compute_boundary_f1(pred_mask, gt_mask)

    n = len(val_loader)
    print(f"{'Model':<22} | {'Params':<8} | {'PSNR ↑':<8} | {'SSIM ↑':<8} | {'mIoU ↑':<8} | {'Dice ↑':<8} | {'Bound F1 ↑':<10} | {'Latency (ms)':<12}")
    print("-" * 105)

    csv_rows = []
    for name, m in metrics.items():
        row = {
            "model": name,
            "params": m["params"],
            "psnr": round(m["psnr"] / n, 2),
            "ssim": round(m["ssim"] / n, 4),
            "lpips": round(m["lpips"] / n, 4),
            "miou": round(m["miou"] / n, 4),
            "dice": round(m["dice"] / n, 4),
            "precision": round(m["prec"] / n, 4),
            "recall": round(m["rec"] / n, 4),
            "boundary_f1": round(m["bf1"] / n, 4),
            "latency_ms": round(m["latency_ms"] / n, 2)
        }
        csv_rows.append(row)
        print(f"{row['model']:<22} | {row['params']:<8} | {row['psnr']:<8.2f} | {row['ssim']:<8.4f} | {row['miou']:<8.4f} | {row['dice']:<8.4f} | {row['boundary_f1']:<10.4f} | {row['latency_ms']:<12.2f}")

    print("-" * 105)

    # Save CSV report
    csv_path = "experiments/spatial_sr_comparison.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["model", "params", "psnr", "ssim", "lpips", "miou", "dice", "precision", "recall", "boundary_f1", "latency_ms"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[Milestone 4] Spatial SR comparison report saved to '{csv_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()
    run_milestone4(args.config)
