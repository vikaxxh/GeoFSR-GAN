import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import SatelliteDataset
from models import BicubicBaseline, SimpleSpatialSR, GeoFSRGenerator
from evaluation import (
    calculate_psnr,
    calculate_ssim,
    calculate_lpips,
    compute_miou,
    compute_dice_score,
    compute_precision_recall,
    compute_boundary_f1,
    generate_ground_truth_mask,
    get_trained_segmentation_net,
    save_segmentation_audit_grid
)


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_all(config_path, model_path=None):
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

    # Instantiate models
    bicubic = BicubicBaseline(scale=dataset_cfg["scale"])
    
    spatial_sr = SimpleSpatialSR(
        scale=dataset_cfg["scale"],
        num_features=config["model"]["spatial_encoder"]["num_features"]
    )

    model_cfg = config["model"]
    geofsr = GeoFSRGenerator(
        scale=model_cfg["scale"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        num_features=model_cfg["spatial_encoder"]["num_features"],
        num_spatial_blocks=model_cfg["spatial_encoder"]["num_blocks"],
        fusion_type=model_cfg["fusion"]["type"]
    )

    if model_path and os.path.exists(model_path):
        geofsr.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[Evaluation] Loaded trained GeoFSR checkpoint from '{model_path}'.")
    else:
        print(f"[Evaluation] Using initial GeoFSR model weights.")

    # Load calibrated segmentation network for downstream evaluation
    seg_head = get_trained_segmentation_net(device=device, save_path="experiments/segmentation_head.pth", dataset=val_dataset)

    bicubic.eval()
    spatial_sr.eval()
    geofsr.eval()
    seg_head.eval()

    # Metrics containers
    models = ["Bicubic", "Spatial SR", "GeoFSR-GAN"]
    metrics = {m: {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "miou": 0.0, "dice": 0.0, "prec": 0.0, "rec": 0.0, "bf1": 0.0} for m in models}

    output_dir = "experiments/evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"   Milestone 1 — Comprehensive Benchmark Evaluation    ")
    print(f"=======================================================\n")

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            lr = batch["lr"]
            hr = batch["hr"]

            # Ground-truth binary target mask
            gt_mask = generate_ground_truth_mask(hr)

            sr_dict = {
                "Bicubic": bicubic(lr),
                "Spatial SR": spatial_sr(lr),
                "GeoFSR-GAN": geofsr(lr)
            }

            mask_dict = {}
            for name, sr in sr_dict.items():
                pred_mask = torch.sigmoid(seg_head(sr))
                mask_dict[name] = pred_mask

                metrics[name]["psnr"] += calculate_psnr(sr, hr)
                metrics[name]["ssim"] += calculate_ssim(sr, hr)
                
                lp_score = calculate_lpips(sr, hr)
                metrics[name]["lpips"] += (lp_score if lp_score is not None else 0.0)

                metrics[name]["miou"] += compute_miou(pred_mask, gt_mask)
                metrics[name]["dice"] += compute_dice_score(pred_mask, gt_mask)
                prec, rec = compute_precision_recall(pred_mask, gt_mask)
                metrics[name]["prec"] += prec
                metrics[name]["rec"] += rec
                metrics[name]["bf1"] += compute_boundary_f1(pred_mask, gt_mask)

            if idx == 0:
                # Save Milestone 1 Audit Grid
                sample_m_bicubic = {k: metrics["Bicubic"][k] / 1.0 for k in ["miou", "dice", "prec", "rec", "bf1"]}
                sample_m_spatial = {k: metrics["Spatial SR"][k] / 1.0 for k in ["miou", "dice", "prec", "rec", "bf1"]}
                sample_m_geofsr = {k: metrics["GeoFSR-GAN"][k] / 1.0 for k in ["miou", "dice", "prec", "rec", "bf1"]}

                grid_path = os.path.join(output_dir, "segmentation_audit_grid.png")
                save_segmentation_audit_grid(
                    hr=hr[0], bicubic=sr_dict["Bicubic"][0], spatial_sr=sr_dict["Spatial SR"][0], geofsr=sr_dict["GeoFSR-GAN"][0],
                    gt_mask=gt_mask[0], mask_bicubic=mask_dict["Bicubic"][0], mask_spatial=mask_dict["Spatial SR"][0], mask_geofsr=mask_dict["GeoFSR-GAN"][0],
                    metrics_bicubic=sample_m_bicubic, metrics_spatial=sample_m_spatial, metrics_geofsr=sample_m_geofsr,
                    save_path=grid_path
                )

    n_samples = len(val_loader)
    print("\n--- Milestone 1 Quantitative Benchmark Summary ---")
    print(f"{'Model':<12} | {'PSNR ↑':<8} | {'SSIM ↑':<8} | {'LPIPS ↓':<8} | {'mIoU ↑':<8} | {'Dice ↑':<8} | {'Prec ↑':<8} | {'Rec ↑':<8} | {'Bound F1 ↑':<10}")
    print("-" * 95)

    for name in models:
        m = metrics[name]
        psnr_avg = m["psnr"] / n_samples
        ssim_avg = m["ssim"] / n_samples
        lpips_avg = m["lpips"] / n_samples
        miou_avg = m["miou"] / n_samples
        dice_avg = m["dice"] / n_samples
        prec_avg = m["prec"] / n_samples
        rec_avg = m["rec"] / n_samples
        bf1_avg = m["bf1"] / n_samples

        print(f"{name:<12} | {psnr_avg:<8.2f} | {ssim_avg:<8.4f} | {lpips_avg:<8.4f} | {miou_avg:<8.4f} | {dice_avg:<8.4f} | {prec_avg:<8.4f} | {rec_avg:<8.4f} | {bf1_avg:<10.4f}")

    print("-" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 1 — Audit Evaluation Pipeline.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file.")
    parser.add_argument("--model_path", type=str, default="experiments/baseline/checkpoints/geofsr_generator_latest.pth", help="Path to checkpoint.")
    args = parser.parse_args()

    evaluate_all(args.config, args.model_path)
