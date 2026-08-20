import io
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from .image_metrics import calculate_psnr, calculate_ssim


def apply_gaussian_noise(lr_tensor, std=0.03):
    """
    Applies additive Gaussian noise to image tensor [B, C, H, W] in [0, 1].
    """
    noise = torch.randn_like(lr_tensor) * std
    return torch.clamp(lr_tensor + noise, 0.0, 1.0)


def apply_gaussian_blur(lr_tensor, sigma=1.0):
    """
    Applies Gaussian spatial blur to image tensor [B, C, H, W] in [0, 1].
    """
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    return TF.gaussian_blur(lr_tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


def apply_jpeg_compression(lr_tensor, quality=60):
    """
    Simulates JPEG compression artifacts on PyTorch tensor [B, C, H, W] in [0, 1].
    """
    out_tensors = []
    for i in range(lr_tensor.shape[0]):
        img_pil = TF.to_pil_image(lr_tensor[i].cpu())
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        img_jpeg = Image.open(buffer).convert("RGB")
        out_tensors.append(TF.to_tensor(img_jpeg))

    return torch.stack(out_tensors, dim=0).to(lr_tensor.device)


def evaluate_perturbation_resilience(model, dataloader, perturbation_fn=None, device="cpu"):
    """
    Evaluates model PSNR and SSIM performance under specified input perturbation operator.
    """
    model.eval()
    psnr_total, ssim_total = 0.0, 0.0

    with torch.no_grad():
        for batch in dataloader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            if perturbation_fn is not None:
                lr = perturbation_fn(lr)

            sr = model(lr)

            psnr_total += calculate_psnr(sr, hr)
            ssim_total += calculate_ssim(sr, hr)

    n = len(dataloader)
    return {
        "psnr": float(psnr_total / n),
        "ssim": float(ssim_total / n)
    }
