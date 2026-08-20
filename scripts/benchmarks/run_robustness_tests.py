import os
import sys
import json
import argparse
import yaml
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets import SatelliteDataset
from models import BicubicBaseline, SimpleSpatialSR, GeoFSRGenerator
from evaluation import (
    apply_gaussian_noise,
    apply_gaussian_blur,
    apply_jpeg_compression,
    evaluate_perturbation_resilience
)


def run_robustness_benchmark(config_path, model_path=None):
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

    # Models
    models = {
        "Bicubic": BicubicBaseline(scale=config["dataset"]["scale"]).to(device),
        "Spatial SR": SimpleSpatialSR(scale=config["dataset"]["scale"]).to(device),
        "GeoFSR-GAN": GeoFSRGenerator(
            scale=config["model"]["scale"],
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            num_features=config["model"]["spatial_encoder"]["num_features"],
            num_spatial_blocks=config["model"]["spatial_encoder"]["num_blocks"],
            fusion_type=config["model"]["fusion"]["type"]
        ).to(device)
    }

    if model_path and os.path.exists(model_path):
        models["GeoFSR-GAN"].load_state_dict(torch.load(model_path, map_location=device))
        print(f"[Robustness] Loaded GeoFSR-GAN weights from '{model_path}'.")

    for m in models.values():
        m.eval()

    # Perturbation sweeps
    noise_stds = [0.0, 0.01, 0.03, 0.05]
    blur_sigmas = [0.0, 0.5, 1.0, 1.5]
    jpeg_qualities = [100, 80, 60, 40]

    results = {
        "noise": {m_name: [] for m_name in models},
        "blur": {m_name: [] for m_name in models},
        "jpeg": {m_name: [] for m_name in models}
    }

    print("\n=======================================================")
    print("      GeoFSR-GAN Perturbation Robustness Benchmark      ")
    print("=======================================================\n")

    # 1. Noise Sweep
    print("[1/3] Running Gaussian Noise Perturbation Sweep...")
    for std in noise_stds:
        fn = (lambda tensor, s=std: apply_gaussian_noise(tensor, std=s)) if std > 0 else None
        for m_name, model in models.items():
            res = evaluate_perturbation_resilience(model, val_loader, fn, device=device)
            results["noise"][m_name].append(res["psnr"])

    # 2. Blur Sweep
    print("[2/3] Running Gaussian Blur Perturbation Sweep...")
    for sigma in blur_sigmas:
        fn = (lambda tensor, s=sigma: apply_gaussian_blur(tensor, sigma=s)) if sigma > 0 else None
        for m_name, model in models.items():
            res = evaluate_perturbation_resilience(model, val_loader, fn, device=device)
            results["blur"][m_name].append(res["psnr"])

    # 3. JPEG Compression Sweep
    print("[3/3] Running JPEG Compression Artifact Sweep...")
    for q in jpeg_qualities:
        fn = (lambda tensor, q_val=q: apply_jpeg_compression(tensor, quality=q_val)) if q < 100 else None
        for m_name, model in models.items():
            res = evaluate_perturbation_resilience(model, val_loader, fn, device=device)
            results["jpeg"][m_name].append(res["psnr"])

    # Save JSON metrics
    out_dir = "experiments"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "robustness_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # Plot Curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    colors = {"Bicubic": "#7f7f7f", "Spatial SR": "#1f77b4", "GeoFSR-GAN": "#2ca02c"}

    # Noise plot
    for m_name in models:
        axes[0].plot(noise_stds, results["noise"][m_name], marker="o", label=m_name, color=colors[m_name], linewidth=2)
    axes[0].set_title("Gaussian Noise Robustness", fontweight="bold")
    axes[0].set_xlabel("Noise Std (σ)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Blur plot
    for m_name in models:
        axes[1].plot(blur_sigmas, results["blur"][m_name], marker="s", label=m_name, color=colors[m_name], linewidth=2)
    axes[1].set_title("Gaussian Blur Robustness", fontweight="bold")
    axes[1].set_xlabel("Blur Sigma (σ)")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # JPEG plot
    for m_name in models:
        axes[2].plot(jpeg_qualities, results["jpeg"][m_name], marker="^", label=m_name, color=colors[m_name], linewidth=2)
    axes[2].set_title("JPEG Compression Robustness", fontweight="bold")
    axes[2].set_xlabel("JPEG Quality (100 = Best)")
    axes[2].set_ylabel("PSNR (dB)")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "robustness_curves.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Robustness Benchmark Complete] Metrics saved to '{json_path}', curves plot saved to '{plot_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Perturbation Robustness Benchmark.")
    parser.add_argument("--config", type=str, default="configs/cpu_debug.yaml")
    parser.add_argument("--model_path", type=str, default="experiments/cpu_debug/checkpoints/geofsr_generator_latest.pth")
    args = parser.parse_args()

    run_robustness_benchmark(args.config, args.model_path)
