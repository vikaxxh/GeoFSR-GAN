import io
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance


class RealisticDegradation:
    """
    Modular, realistic degradation pipeline for satellite imagery.
    
    Order of degradation:
    HR Image -> Blur (Gaussian/Motion) -> Downsampling (Bicubic) -> Noise (Gaussian/Poisson) -> JPEG Compression -> Color Perturbation -> LR Image
    """
    def __init__(self, config=None):
        cfg = config.get("degradation", {}) if config else {}
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
        """Applies Gaussian or Motion blur to PIL image."""
        sigma = rng.uniform(self.blur_sigma_min, self.blur_sigma_max)
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img_pil

    def apply_downsampling(self, img_pil, scale):
        """Downsamples image using bicubic interpolation."""
        w, h = img_pil.size
        lr_w, lr_h = w // scale, h // scale
        return img_pil.resize((lr_w, lr_h), resample=Image.BICUBIC)

    def apply_noise(self, img_np, rng):
        """Applies Gaussian or Poisson noise to float32 NumPy array in [0, 1]."""
        if self.noise_type in ["gaussian", "mixed"]:
            noise_std = rng.uniform(0.005, self.noise_std_max)
            gaussian_noise = rng.normal(0, noise_std, img_np.shape)
            img_np = np.clip(img_np + gaussian_noise, 0.0, 1.0)

        if self.noise_type in ["poisson", "mixed"] and rng.random() < 0.5:
            # Scale up to photon count representation, apply Poisson, then scale back
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

    def degrade(self, hr_img, scale=4, seed=None):
        """
        Degrade a high-resolution input into a realistic low-resolution output.
        
        Args:
            hr_img: PIL Image or PyTorch Tensor [C, H, W] in [0, 1].
            scale: Downsampling scale factor (e.g. 2 or 4).
            seed: Optional integer seed for reproducibility.
            
        Returns:
            lr_img: Degraded low-resolution output matching hr_img type.
        """
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

        # 1. Blur
        img_blurred = self.apply_blur(hr_pil, rng)

        # 2. Downsampling (Bicubic)
        img_lr_pil = self.apply_downsampling(img_blurred, scale)

        # 3. Noise
        img_lr_np = np.array(img_lr_pil, dtype=np.float32) / 255.0
        img_noisy_np = self.apply_noise(img_lr_np, np_rng)

        # Convert back to PIL for JPEG compression
        img_noisy_pil = Image.fromarray((img_noisy_np * 255.0).astype(np.uint8))

        # 4. JPEG Compression
        img_compressed_pil = self.apply_jpeg_compression(img_noisy_pil, rng)

        # 5. Color Perturbation
        img_final_pil = self.apply_color_perturbation(img_compressed_pil, rng)

        if is_tensor:
            return TF.to_tensor(img_final_pil)
        return img_final_pil
