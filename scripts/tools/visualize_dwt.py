import os
import sys
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models import DWT2D
from datasets import SatelliteDataset


def visualize_dwt_subbands(config_path, output_path="data/dwt_subbands_visualization.png"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = config["dataset"]["data_dir"]
    dataset = SatelliteDataset(data_dir=data_dir, scale=4, hr_patch_size=128, is_train=False, config=config)
    sample = dataset[0]
    hr_tensor = sample["hr"].unsqueeze(0)  # [1, 3, 128, 128]

    dwt = DWT2D(in_channels=3)
    ll, lh, hl, hh = dwt(hr_tensor)

    def to_vis_np(tensor):
        img_np = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # Normalize for visual inspection
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        return img_np

    ll_vis = to_vis_np(ll)
    lh_vis = to_vis_np(lh)
    hl_vis = to_vis_np(hl)
    hh_vis = to_vis_np(hh)
    hr_vis = to_vis_np(hr_tensor)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=150)
    fig.suptitle("GeoFSR-GAN Discrete Wavelet Transform (DWT) Sub-band Decomposition", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(hr_vis)
    axes[0, 0].set_title("Original Satellite HR Image\nSpatial Domain [128x128]", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(ll_vis)
    axes[0, 1].set_title("LL Sub-band\nLow-Freq Approximation [64x64]", fontsize=10)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(lh_vis)
    axes[0, 2].set_title("LH Sub-band\nHorizontal High-Freq Edges [64x64]", fontsize=10)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(hl_vis)
    axes[1, 0].set_title("HL Sub-band\nVertical High-Freq Edges [64x64]", fontsize=10)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(hh_vis)
    axes[1, 1].set_title("HH Sub-band\nDiagonal High-Freq Details [64x64]", fontsize=10)
    axes[1, 1].axis("off")

    # Spectral energy summary
    axes[1, 2].axis("off")
    energy_text = (
        f"Sub-band Relative Energies:\n"
        f"  • LL Energy: {torch.norm(ll).item():.2f}\n"
        f"  • LH Energy: {torch.norm(lh).item():.2f}\n"
        f"  • HL Energy: {torch.norm(hl).item():.2f}\n"
        f"  • HH Energy: {torch.norm(hh).item():.2f}\n"
    )
    axes[1, 2].text(0.1, 0.4, energy_text, fontsize=11, family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[DWT Visualizer] Successfully saved sub-band visual grid to '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DWT Sub-bands for Satellite Imagery.")
    parser.add_argument("--config", type=str, default="configs/cpu_debug.yaml", help="Path to config file.")
    args = parser.parse_args()
    visualize_dwt_subbands(args.config)
