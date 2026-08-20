import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import torchvision.transforms.functional as TF

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import SatelliteDataset


def inspect_dataset(config_path, output_vis_path="data/sample_inspection.png"):
    print(f"[Dataset Inspector] Loading configuration from '{config_path}'...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = config["dataset"]["data_dir"]
    scale = config["dataset"]["scale"]
    hr_patch_size = config["dataset"]["hr_patch_size"]
    batch_size = config["training"]["batch_size"]

    if not os.path.exists(data_dir):
        print(f"[Dataset Inspector] Data directory '{data_dir}' not found. Generating synthetic dataset...")
        from prepare_dataset import generate_synthetic_satellite_image
        os.makedirs(data_dir, exist_ok=True)
        for i in range(10):
            img = generate_synthetic_satellite_image(size=(512, 512), seed=42 + i)
            img.save(os.path.join(data_dir, f"sample_{i+1:02d}.png"))

    dataset = SatelliteDataset(
        data_dir=data_dir,
        scale=scale,
        hr_patch_size=hr_patch_size,
        is_train=True,
        config=config
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"\n================ DATASET INSPECTION REPORT ================")
    print(f"Total Dataset Samples: {len(dataset)}")
    print(f"Scale Factor: x{scale}")
    print(f"HR Patch Size: {hr_patch_size}x{hr_patch_size}")
    print(f"LR Patch Size: {hr_patch_size // scale}x{hr_patch_size // scale}")
    print(f"Batch Size: {batch_size}")
    print("==========================================================")

    for i, batch in enumerate(dataloader):
        lr, hr, filenames = batch["lr"], batch["hr"], batch["filename"]
        print(f"\nBatch {i+1}:")
        print(f"  - LR Tensor Shape: {lr.shape} | Dtype: {lr.dtype} | Range: [{lr.min().item():.4f}, {lr.max().item():.4f}]")
        print(f"  - HR Tensor Shape: {hr.shape} | Dtype: {hr.dtype} | Range: [{hr.min().item():.4f}, {hr.max().item():.4f}]")
        print(f"  - Files: {filenames}")

        # Bicubic upscale LR visually for inspection grid comparison
        lr_upscaled = torch.nn.functional.interpolate(lr, size=(hr_patch_size, hr_patch_size), mode="bicubic", align_corners=False)

        # Create side-by-side visual comparison (LR degraded upscaled vs HR ground truth)
        vis_grid = torch.cat([lr_upscaled, hr], dim=3) # Concatenate horizontally
        grid_image = make_grid(vis_grid, nrow=1, padding=2)
        os.makedirs(os.path.dirname(output_vis_path), exist_ok=True)
        save_image(grid_image, output_vis_path)
        print(f"\n[Dataset Inspector] Saved side-by-side visual inspection sample to '{output_vis_path}'.")
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect GeoFSR-GAN Satellite Dataset.")
    parser.add_argument("--config", type=str, default="configs/cpu_debug.yaml", help="Path to config file.")
    args = parser.parse_args()
    inspect_dataset(args.config)
