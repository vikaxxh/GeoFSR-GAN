import math
import numpy as np
import torch
from skimage.metrics import structural_similarity as skimage_ssim

try:
    import lpips
    _LPIPS_AVAILABLE = True
    _lpips_fn = lpips.LPIPS(net='alex').eval()
except Exception:
    _LPIPS_AVAILABLE = False
    _lpips_fn = None


def tensor_to_numpy(img):
    """Converts PyTorch tensor [C, H, W] or [B, C, H, W] to uint8/float NumPy [H, W, C]."""
    if torch.is_tensor(img):
        if img.dim() == 4:
            img = img.squeeze(0)
        img = img.detach().cpu().numpy()

    if img.shape[0] in [1, 3]:  # Convert [C, H, W] -> [H, W, C]
        img = np.transpose(img, (1, 2, 0))

    return np.clip(img, 0.0, 1.0)


def calculate_psnr(sr, hr, max_val=1.0):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in dB.
    
    Args:
        sr: PyTorch Tensor or NumPy array [0.0, 1.0]
        hr: PyTorch Tensor or NumPy array [0.0, 1.0]
        max_val: Maximum pixel value (1.0 for float)
        
    Returns:
        float: PSNR in dB
    """
    sr_np = tensor_to_numpy(sr)
    hr_np = tensor_to_numpy(hr)

    # Crop to matching spatial resolution if necessary
    h = min(sr_np.shape[0], hr_np.shape[0])
    w = min(sr_np.shape[1], hr_np.shape[1])
    sr_np = sr_np[:h, :w]
    hr_np = hr_np[:h, :w]

    mse = np.mean((sr_np - hr_np) ** 2)
    if mse == 0:
        return 100.0  # Identical images
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def calculate_ssim(sr, hr):
    """
    Computes Structural Similarity Index Measure (SSIM).
    
    Args:
        sr: PyTorch Tensor or NumPy array [0.0, 1.0]
        hr: PyTorch Tensor or NumPy array [0.0, 1.0]
        
    Returns:
        float: SSIM score between -1.0 and 1.0
    """
    sr_np = tensor_to_numpy(sr)
    hr_np = tensor_to_numpy(hr)

    # Crop to matching spatial resolution if necessary
    h = min(sr_np.shape[0], hr_np.shape[0])
    w = min(sr_np.shape[1], hr_np.shape[1])
    sr_np = sr_np[:h, :w]
    hr_np = hr_np[:h, :w]

    # Determine channel axis
    if sr_np.ndim == 3:
        return float(skimage_ssim(sr_np, hr_np, channel_axis=2, data_range=1.0))
    return float(skimage_ssim(sr_np, hr_np, data_range=1.0))


def calculate_lpips(sr, hr):
    """
    Computes LPIPS perceptual similarity distance.
    Returns None if LPIPS module is unavailable.
    """
    if not _LPIPS_AVAILABLE or _lpips_fn is None:
        return None

    with torch.no_grad():
        if not torch.is_tensor(sr):
            sr = torch.from_numpy(sr).permute(2, 0, 1).unsqueeze(0).float()
        if not torch.is_tensor(hr):
            hr = torch.from_numpy(hr).permute(2, 0, 1).unsqueeze(0).float()

        if sr.dim() == 3:
            sr = sr.unsqueeze(0)
        if hr.dim() == 3:
            hr = hr.unsqueeze(0)

        # Scale from [0, 1] to [-1, 1] for LPIPS
        sr_norm = sr * 2.0 - 1.0
        hr_norm = hr * 2.0 - 1.0

        dist = _lpips_fn(sr_norm, hr_norm)
        return float(dist.item())
