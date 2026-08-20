import os
import sys
import argparse
import yaml
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import SatelliteDataset
from losses import SobelEdgeFilter


def visualize_sobel_edges(config_path, output_path="data/sobel_edge_visualization.png"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = config["dataset"]["data_dir"]
    dataset = SatelliteDataset(data_dir=data_dir, scale=4, hr_patch_size=128, is_train=False, config=config)
    sample = dataset[0]
    hr_tensor = sample["hr"].unsqueeze(0)  # [1, 3, 128, 128]

    sobel = SobelEdgeFilter(in_channels=3)
    magnitude, gx, gy = sobel(hr_tensor)

    def to_vis(t):
        t_np = t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        t_np = (t_np - t_np.min()) / (t_np.max() - t_np.min() + 1e-8)
        return t_np

    hr_vis = to_vis(hr_tensor)
    gx_vis = to_vis(gx.abs())
    gy_vis = to_vis(gy.abs())
    mag_vis = to_vis(magnitude)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)
    fig.suptitle("GeoFSR-GAN Differentiable Sobel Edge Extraction", fontsize=14, fontweight="bold")

    axes[0].imshow(hr_vis)
    axes[0].set_title("Satellite Image HR\nSpatial Domain", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(gx_vis)
    axes[1].set_title("Horizontal Gradients |Gx|\nVertical Edges & Walls", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(gy_vis)
    axes[2].set_title("Vertical Gradients |Gy|\nHorizontal Edges & Roads", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(mag_vis)
    axes[3].set_title("Sobel Gradient Magnitude\nCombined Edge Map G", fontsize=10)
    axes[3].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[Sobel Visualizer] Saved edge extraction figure to '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Sobel Edge Extraction.")
    parser.add_argument("--config", type=str, default="configs/cpu_debug.yaml", help="Path to config file.")
    args = parser.parse_args()
    visualize_sobel_edges(args.config)
