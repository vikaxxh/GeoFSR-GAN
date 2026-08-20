import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from .image_metrics import tensor_to_numpy, calculate_psnr, calculate_ssim


def save_baseline_comparison(lr, bicubic, spatial_sr, hr, save_path, sample_idx=1):
    """
    Saves a 4-panel comparison figure: LR (Upscaled) | Bicubic | Spatial SR | HR Ground Truth
    with overlaid quantitative metrics.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if torch.is_tensor(lr):
        if lr.dim() == 3:
            lr = lr.unsqueeze(0)
        lr_up = torch.nn.functional.interpolate(lr, size=(hr.shape[-2], hr.shape[-1]), mode="bicubic", align_corners=False)
        lr_np = tensor_to_numpy(lr_up)
    else:
        lr_np = tensor_to_numpy(lr)

    bicubic_np = tensor_to_numpy(bicubic)
    spatial_sr_np = tensor_to_numpy(spatial_sr)
    hr_np = tensor_to_numpy(hr)

    bicubic_psnr = calculate_psnr(bicubic_np, hr_np)
    bicubic_ssim = calculate_ssim(bicubic_np, hr_np)

    sr_psnr = calculate_psnr(spatial_sr_np, hr_np)
    sr_ssim = calculate_ssim(spatial_sr_np, hr_np)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150)
    fig.suptitle(f"GeoFSR-GAN Baseline Comparison (Sample {sample_idx})", fontsize=14, fontweight="bold", y=0.98)

    axes[0].imshow(lr_np)
    axes[0].set_title("Degraded LR (Bicubic Up)\nLow-Res Input", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(bicubic_np)
    axes[1].set_title(f"Bicubic Baseline\nPSNR: {bicubic_psnr:.2f} dB | SSIM: {bicubic_ssim:.4f}", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(spatial_sr_np)
    axes[2].set_title(f"Spatial SR Baseline\nPSNR: {sr_psnr:.2f} dB | SSIM: {sr_ssim:.4f}", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(hr_np)
    axes[3].set_title("HR Ground Truth\nTarget Satellite Image", fontsize=10)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved baseline comparison figure to '{save_path}'.")


def save_comparison_grid(images_dict, save_path):
    """
    Saves a grid figure comparing arbitrary model output tensors.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_panels = len(images_dict)

    fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4.5), dpi=150)
    if num_panels == 1:
        axes = [axes]

    for ax, (title, img_tensor) in zip(axes, images_dict.items()):
        img_np = tensor_to_numpy(img_tensor)
        ax.imshow(img_np)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_segmentation_audit_grid(
    hr, bicubic, spatial_sr, geofsr,
    gt_mask, mask_bicubic, mask_spatial, mask_geofsr,
    metrics_bicubic, metrics_spatial, metrics_geofsr,
    save_path="experiments/evaluation_results/segmentation_audit_grid.png"
):
    """
    Milestone 1 Visualization Grid:
    Row 1: HR Image | Bicubic Image | Spatial SR Image | GeoFSR-GAN Image
    Row 2: Ground-Truth Mask | Bicubic Pred Mask | Spatial SR Pred Mask | GeoFSR-GAN Pred Mask
    With overlaid mIoU, Dice, Precision, Recall, and Boundary F1 score metrics.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), dpi=150)
    fig.suptitle("Milestone 1 — Downstream Segmentation & Visual Quality Audit Grid", fontsize=16, fontweight="bold", y=0.98)

    # Row 1: Images
    axes[0, 0].imshow(tensor_to_numpy(hr))
    axes[0, 0].set_title("Ground-Truth HR Image", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(tensor_to_numpy(bicubic))
    axes[0, 1].set_title("Bicubic Baseline Image", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(tensor_to_numpy(spatial_sr))
    axes[0, 2].set_title("Spatial SR Baseline Image", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(tensor_to_numpy(geofsr))
    axes[0, 3].set_title("GeoFSR-GAN Image", fontsize=11, fontweight="bold")
    axes[0, 3].axis("off")

    # Row 2: Binary Masks
    gt_mask_np = tensor_to_numpy(gt_mask)
    if gt_mask_np.ndim == 3:
        gt_mask_np = gt_mask_np[:, :, 0]

    axes[1, 0].imshow(gt_mask_np, cmap="gray")
    axes[1, 0].set_title("Ground-Truth Target Mask", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # Helper format string
    def fmt(m):
        return f"mIoU: {m['miou']:.4f} | Dice: {m['dice']:.4f}\nPrec: {m['prec']:.4f} | Rec: {m['rec']:.4f}\nBound F1: {m['bf1']:.4f}"

    axes[1, 1].imshow(tensor_to_numpy(mask_bicubic if mask_bicubic.dim()==3 else mask_bicubic[0]), cmap="gray")
    axes[1, 1].set_title(f"Bicubic Pred Mask\n{fmt(metrics_bicubic)}", fontsize=9)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(tensor_to_numpy(mask_spatial if mask_spatial.dim()==3 else mask_spatial[0]), cmap="gray")
    axes[1, 2].set_title(f"Spatial SR Pred Mask\n{fmt(metrics_spatial)}", fontsize=9)
    axes[1, 2].axis("off")

    axes[1, 3].imshow(tensor_to_numpy(mask_geofsr if mask_geofsr.dim()==3 else mask_geofsr[0]), cmap="gray")
    axes[1, 3].set_title(f"GeoFSR-GAN Pred Mask\n{fmt(metrics_geofsr)}", fontsize=9)
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Milestone 1 audit grid saved to '{save_path}'.")
