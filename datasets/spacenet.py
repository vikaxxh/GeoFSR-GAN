import os
import random
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from .degradation import RealisticDegradation
from .transforms import PairedTransform


class SatelliteDataset(Dataset):
    """
    Lazy-loading PyTorch dataset for satellite imagery super-resolution.
    Extracts patches dynamically to guarantee low RAM footprint on CPU systems.
    """
    def __init__(self, data_dir, scale=4, hr_patch_size=96, is_train=True, config=None, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.scale = scale
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale
        self.is_train = is_train
        self.seed = seed
        self.config = config

        self.degradation = RealisticDegradation(config=config)
        self.transform = PairedTransform(is_train=is_train)

        # Collect high-resolution image paths (lazy loading)
        self.image_paths = sorted(
            glob.glob(os.path.join(data_dir, "*.png")) +
            glob.glob(os.path.join(data_dir, "*.jpg")) +
            glob.glob(os.path.join(data_dir, "*.tif"))
        )

        subset_size = config.get("dataset", {}).get("subset_size", None) if config else None
        if subset_size is not None and subset_size > 0:
            self.image_paths = self.image_paths[:subset_size]

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No valid images (.png, .jpg, .tif) found in {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def extract_patch(self, hr_img, idx):
        """Extracts spatial patch from full-size HR image."""
        w, h = hr_img.size
        if w < self.hr_patch_size or h < self.hr_patch_size:
            # Resize small image up to patch size
            hr_img = hr_img.resize((max(w, self.hr_patch_size), max(h, self.hr_patch_size)), Image.BICUBIC)
            w, h = hr_img.size

        if self.is_train:
            # Random crop for training
            left = random.randint(0, w - self.hr_patch_size)
            top = random.randint(0, h - self.hr_patch_size)
        else:
            # Deterministic center crop for val/test
            rng = random.Random(self.seed + idx)
            left = (w - self.hr_patch_size) // 2
            top = (h - self.hr_patch_size) // 2

        hr_patch = hr_img.crop((left, top, left + self.hr_patch_size, top + self.hr_patch_size))
        return hr_patch

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        hr_full = Image.open(img_path).convert("RGB")

        # 1. Extract HR Patch
        hr_patch = self.extract_patch(hr_full, idx)
        hr_tensor = TF.to_tensor(hr_patch)

        # 2. Generate Degraded LR Patch using Realistic Degradation Pipeline
        sample_seed = (self.seed + idx) if not self.is_train else None
        lr_tensor = self.degradation.degrade(hr_tensor, scale=self.scale, seed=sample_seed)

        # 3. Apply Paired Transforms
        hr_tensor, lr_tensor = self.transform(hr_tensor, lr_tensor)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "filename": os.path.basename(img_path)
        }
