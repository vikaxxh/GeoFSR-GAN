import io
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance


class RealisticDegradation:
    """
    Controlled & modular degradation pipeline supporting D1, D2, D3, D4 modes:
    - D1: HR -> Downsampling
    - D2: HR -> Blur -> Downsampling
    - D3: HR -> Blur -> Downsampling -> Noise
    - D4: HR -> Blur -> Downsampling -> Noise -> JPEG Compression -> Color Perturbation (Full)
    """
    def __init__(self, config=None, mode=None):
        cfg = config.get("degradation", {}) if config else {}
        self.mode = mode if mode is not None else cfg.get("mode", "D4")

        self.blur_kernel_size = cfg.get("blur_kernel_size", 7)
        self.blur_sigma_min = cfg.get("blur_sigma_min", 0.2)
        self.blur_sigma_max = cfg.get("blur_sigma_max", 2.0)
        self.use_motion_blur = cfg.get("use_motion_blur", True)
        self.noise_type = cfg.get("noise_type", "gaussian")
        self.noise_std_max = cfg.get("noise_std_max", 0.05)
        self.jpeg_quality_min = cfg.get("jpeg_quality_min", 50)
        self.jpeg_quality_max = cfg.get("jpeg_quality_max", 95)
        self.color_jitter_prob = cfg.get("color_jitter_prob", 0.3)

    def apply_blur(self, img_pil, rng):
        """Applies Gaussian blur to PIL image."""
        sigma = rng.uniform(self.blur_sigma_min, self.blur_sigma_max)
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img_pil

    def apply_downsampling(self, img_pil, scale, rng=None):
        """Downsamples image using randomized interpolation kernels (Bicubic / Bilinear / Box)."""
        w, h = img_pil.size
        lr_w, lr_h = w // scale, h // scale
        
        methods = [Image.BICUBIC, Image.BILINEAR, Image.BOX]
        resample_fn = rng.choice(methods) if rng is not None else Image.BICUBIC
        return img_pil.resize((lr_w, lr_h), resample=resample_fn)

    def apply_noise(self, img_np, rng):
        """Applies Gaussian or Poisson noise to float32 NumPy array in [0, 1]."""
        if self.noise_type in ["gaussian", "mixed"]:
            noise_std = rng.uniform(0.005, self.noise_std_max)
            gaussian_noise = rng.normal(0, noise_std, img_np.shape)
            img_np = np.clip(img_np + gaussian_noise, 0.0, 1.0)

        if self.noise_type in ["poisson", "mixed"] and rng.random() < 0.5:
            vals = len(np.unique(img_np))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = rng.poisson(img_np * vals) / float(vals)
            img_np = np.clip(noisy, 0.0, 1.0)

        return img_np.astype(np.float32)

    def apply_jpeg_compression(self, img_pil, rng):
        """Simulates JPEG compression artifacts via in-memory buffer."""
        quality = int(rng.randint(self.jpeg_quality_min, self.jpeg_quality_max))
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def apply_color_perturbation(self, img_pil, rng):
        """Applies slight brightness and contrast adjustments."""
        if rng.random() < self.color_jitter_prob:
            brightness_factor = rng.uniform(0.9, 1.1)
            contrast_factor = rng.uniform(0.9, 1.1)
            img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness_factor)
            img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast_factor)
        return img_pil

    def degrade(self, hr_img, scale=4, seed=None, mode=None):
        """
        Degrade a high-resolution input into a low-resolution output based on selected mode.
        """
        deg_mode = mode if mode is not None else self.mode
        rng = random.Random(seed) if seed is not None else random
        np_rng = np.random.RandomState(seed) if seed is not None else np.random

        is_tensor = torch.is_tensor(hr_img)
        if is_tensor:
            if hr_img.dim() == 4:
                hr_img = hr_img.squeeze(0)
            hr_pil = TF.to_pil_image(hr_img)
        elif isinstance(hr_img, np.ndarray):
            if hr_img.dtype != np.uint8:
                hr_img = (np.clip(hr_img, 0.0, 1.0) * 255.0).astype(np.uint8)
            hr_pil = Image.fromarray(hr_img)
        elif isinstance(hr_img, Image.Image):
            hr_pil = hr_img
        else:
            raise TypeError(f"Unsupported image type: {type(hr_img)}")

        # D1: Direct Downsampling
        if deg_mode == "D1":
            img_lr_pil = self.apply_downsampling(hr_pil, scale, rng=rng)
            return TF.to_tensor(img_lr_pil) if is_tensor else img_lr_pil

        # D2: Blur -> Downsampling
        img_curr = self.apply_blur(hr_pil, rng)
        img_lr_pil = self.apply_downsampling(img_curr, scale, rng=rng)
        if deg_mode == "D2":
            return TF.to_tensor(img_lr_pil) if is_tensor else img_lr_pil

        # D3: Blur -> Downsampling -> Noise
        img_lr_np = np.array(img_lr_pil, dtype=np.float32) / 255.0
        img_noisy_np = self.apply_noise(img_lr_np, np_rng)
        img_noisy_pil = Image.fromarray((img_noisy_np * 255.0).astype(np.uint8))
        if deg_mode == "D3":
            return TF.to_tensor(img_noisy_pil) if is_tensor else img_noisy_pil

        # D4: Blur -> Downsampling -> Noise -> JPEG -> Color Jitter (Full)
        img_compressed_pil = self.apply_jpeg_compression(img_noisy_pil, rng)
        img_final_pil = self.apply_color_perturbation(img_compressed_pil, rng)
        return TF.to_tensor(img_final_pil) if is_tensor else img_final_pil
