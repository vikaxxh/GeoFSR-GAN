import os
import sys
import time
import argparse
import yaml
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import GeoFSRGenerator
from datasets import SatelliteDataset
from evaluation import calculate_psnr, calculate_ssim


def compare_fusion_mechanisms(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = config["dataset"]["data_dir"]
    scale = config["dataset"]["scale"]
    hr_patch_size = config["dataset"]["hr_patch_size"]

    dataset = SatelliteDataset(data_dir=data_dir, scale=scale, hr_patch_size=hr_patch_size, is_train=False, config=config)
    sample = dataset[0]
    lr = sample["lr"].unsqueeze(0)  # [1, 3, 24, 24]
    hr = sample["hr"].unsqueeze(0)  # [1, 3, 96, 96]

    fusion_types = ["concat", "learned", "attention", "cross_attention", "dual_cross_attention"]
    results = {}

    print(f"\n================ FUSION MECHANISM COMPARATIVE BENCHMARK ================")
    print(f"{'Fusion Type':22s} | {'Parameters':12s} | {'Latency (ms)':15s} | {'PSNR (dB)':10s} | {'SSIM':8s}")
    print("-" * 79)

    for f_type in fusion_types:
        model = GeoFSRGenerator(
            scale=scale,
            in_channels=3,
            out_channels=3,
            num_features=32,
            num_spatial_blocks=2,
            fusion_type=f_type
        )
        model.eval()

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Measure forward latency over 20 warm runs
        start_time = time.time()
        with torch.no_grad():
            for _ in range(20):
                sr = model(lr)
        end_time = time.time()
        avg_latency_ms = ((end_time - start_time) / 20.0) * 1000.0

        psnr_val = calculate_psnr(sr, hr)
        ssim_val = calculate_ssim(sr, hr)

        results[f_type] = {
            "parameters": param_count,
            "latency_ms": avg_latency_ms,
            "psnr": psnr_val,
            "ssim": ssim_val
        }

        print(f"{f_type.replace('_', ' ').title():22s} | {param_count:12,d} | {avg_latency_ms:15.2f} | {psnr_val:10.2f} | {ssim_val:8.4f}")

    print("========================================================================\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Spatial-Frequency Fusion Mechanisms.")
    parser.add_argument("--config", type=str, default="configs/cpu_debug.yaml", help="Path to config file.")
    args = parser.parse_args()
    compare_fusion_mechanisms(args.config)
