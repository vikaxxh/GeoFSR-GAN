import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import SatelliteDataset
from models import BicubicBaseline, SimpleSpatialSR, GeoFSRGenerator, LightweightSegmentationUNet
from evaluation import calculate_psnr, calculate_ssim, compute_miou, save_comparison_grid


def evaluate_all(config_path, model_path=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

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

    seg_head = LightweightSegmentationUNet(in_channels=3, num_classes=1, num_features=16)

    bicubic.eval()
    spatial_sr.eval()
    geofsr.eval()
    seg_head.eval()

    # Metrics containers
    metrics = {
        "Bicubic": {"psnr": 0.0, "ssim": 0.0, "miou": 0.0},
        "Spatial SR": {"psnr": 0.0, "ssim": 0.0, "miou": 0.0},
        "GeoFSR-GAN": {"psnr": 0.0, "ssim": 0.0, "miou": 0.0}
    }

    output_dir = "experiments/evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"      GeoFSR-GAN Comprehensive Benchmark Evaluation     ")
    print(f"=======================================================\n")

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            lr = batch["lr"]
            hr = batch["hr"]

            sr_bicubic = bicubic(lr)
            sr_spatial = spatial_sr(lr)
            sr_geofsr = geofsr(lr)

            # Metrics
            metrics["Bicubic"]["psnr"] += calculate_psnr(sr_bicubic, hr)
            metrics["Bicubic"]["ssim"] += calculate_ssim(sr_bicubic, hr)

            metrics["Spatial SR"]["psnr"] += calculate_psnr(sr_spatial, hr)
            metrics["Spatial SR"]["ssim"] += calculate_ssim(sr_spatial, hr)

            metrics["GeoFSR-GAN"]["psnr"] += calculate_psnr(sr_geofsr, hr)
            metrics["GeoFSR-GAN"]["ssim"] += calculate_ssim(sr_geofsr, hr)

            # Downstream segmentation mIoU
            mask_hr = torch.sigmoid(seg_head(hr))
            metrics["Bicubic"]["miou"] += compute_miou(torch.sigmoid(seg_head(sr_bicubic)), mask_hr)
            metrics["Spatial SR"]["miou"] += compute_miou(torch.sigmoid(seg_head(sr_spatial)), mask_hr)
            metrics["GeoFSR-GAN"]["miou"] += compute_miou(torch.sigmoid(seg_head(sr_geofsr)), mask_hr)

            if idx == 0:
                # Save visual grid comparison
                images_dict = {
                    "LR (Bicubic x4)": sr_bicubic[0],
                    "Spatial SR": sr_spatial[0],
                    "GeoFSR-GAN": sr_geofsr[0],
                    "Ground Truth HR": hr[0]
                }
                grid_path = os.path.join(output_dir, "geofsr_comparison_grid.png")
                save_comparison_grid(images_dict, save_path=grid_path)
                print(f"[Grid Saved] Visual evaluation grid saved to '{grid_path}'.")

    n_samples = len(val_loader)
    print("\n--- Final Quantitative Benchmark Metrics ---")
    print(f"{'Model':<15} | {'PSNR (dB)':<10} | {'SSIM':<10} | {'Segmentation mIoU':<18}")
    print("-" * 65)

    for name, m in metrics.items():
        psnr_avg = m["psnr"] / n_samples
        ssim_avg = m["ssim"] / n_samples
        miou_avg = m["miou"] / n_samples
        print(f"{name:<15} | {psnr_avg:<10.2f} | {ssim_avg:<10.4f} | {miou_avg:<18.4f}")

    print("-" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GeoFSR-GAN against Baselines.")
    parser.add_argument("--config", type=str, default="configs/cpu_debug.yaml", help="Path to config file.")
    parser.add_argument("--model_path", type=str, default="experiments/cpu_debug/checkpoints/geofsr_generator_latest.pth", help="Path to checkpoint.")
    args = parser.parse_args()

    evaluate_all(args.config, args.model_path)
