import os
import glob
import argparse
import yaml
from PIL import Image
import torchvision.transforms.functional as TF
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from datasets import RealisticDegradation


def main():
    parser = argparse.ArgumentParser(description="Generate degraded Low-Resolution images from High-Resolution folder.")
    parser.add_argument("--hr_dir", type=str, required=True, help="Directory containing HR input images.")
    parser.add_argument("--lr_dir", type=str, required=True, help="Directory to save degraded LR images.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor (e.g. 2 or 4).")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config file path.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    args = parser.parse_args()

    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    degradation = RealisticDegradation(config=config)
    os.makedirs(args.lr_dir, exist_ok=True)

    hr_paths = sorted(
        glob.glob(os.path.join(args.hr_dir, "*.png")) +
        glob.glob(os.path.join(args.hr_dir, "*.jpg")) +
        glob.glob(os.path.join(args.hr_dir, "*.tif"))
    )

    print(f"[Generate LR] Processing {len(hr_paths)} images from '{args.hr_dir}' to '{args.lr_dir}' (scale x{args.scale})...")

    for i, hr_path in enumerate(hr_paths):
        hr_img = Image.open(hr_path).convert("RGB")
        hr_tensor = TF.to_tensor(hr_img)
        lr_tensor = degradation.degrade(hr_tensor, scale=args.scale, seed=args.seed + i)
        lr_pil = TF.to_pil_image(lr_tensor)

        base_name = os.path.basename(hr_path)
        save_path = os.path.join(args.lr_dir, base_name)
        lr_pil.save(save_path)

    print(f"[Generate LR] Offline LR image batch generation complete.")


if __name__ == "__main__":
    main()
