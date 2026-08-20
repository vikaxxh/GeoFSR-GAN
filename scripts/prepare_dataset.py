import os
import argparse
import numpy as np
from PIL import Image, ImageDraw


def generate_synthetic_satellite_image(size=(512, 512), seed=42):
    """
    Generates realistic synthetic satellite imagery features:
    - Building footprints (polygons / rectangles with roof textures)
    - Road networks (intersecting grey lines)
    - Vegetation / terrain patches (greenish textures)
    """
    rng = np.random.RandomState(seed)
    
    # 1. Base terrain background
    base_color = rng.randint(40, 90, size=3)
    img_np = np.ones((size[1], size[0], 3), dtype=np.uint8) * base_color
    noise = rng.normal(0, 15, size=img_np.shape)
    img_np = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)
    
    # 2. Draw Road Network
    num_roads = rng.randint(3, 7)
    for _ in range(num_roads):
        x1, y1 = rng.randint(0, size[0]), rng.randint(0, size[1])
        x2, y2 = rng.randint(0, size[0]), rng.randint(0, size[1])
        road_width = rng.randint(12, 24)
        road_color = (rng.randint(60, 100), rng.randint(60, 100), rng.randint(60, 100))
        draw.line([(x1, y1), (x2, y2)], fill=road_color, width=road_width)
        
    # 3. Draw Building Footprints with Sharp Boundaries
    num_buildings = rng.randint(15, 30)
    for _ in range(num_buildings):
        bx = rng.randint(20, size[0] - 80)
        by = rng.randint(20, size[1] - 80)
        bw = rng.randint(30, 70)
        bh = rng.randint(30, 70)
        
        roof_color = (rng.randint(140, 220), rng.randint(100, 180), rng.randint(80, 160))
        outline_color = (rng.randint(20, 50), rng.randint(20, 50), rng.randint(20, 50))
        
        draw.rectangle([bx, by, bx + bw, by + bh], fill=roof_color, outline=outline_color, width=2)
        
    return img


def main():
    parser = argparse.ArgumentParser(description="Prepare synthetic satellite dataset for CPU debug testing.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of synthetic HR images to generate.")
    parser.add_argument("--output_dir", type=str, default="data/sample_dataset", help="Output directory path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Dataset Preparation] Generating {args.num_samples} synthetic satellite HR images in '{args.output_dir}'...")

    for i in range(args.num_samples):
        img_seed = args.seed + i
        img = generate_synthetic_satellite_image(size=(512, 512), seed=img_seed)
        filename = f"sat_sample_{i+1:03d}.png"
        filepath = os.path.join(args.output_dir, filename)
        img.save(filepath)

    print(f"[Dataset Preparation] Successfully created dataset with {args.num_samples} samples.")
    print("\n--- Real SpaceNet Dataset Download Instructions ---")
    print("To train on official SpaceNet high-resolution satellite data:")
    print("1. Install AWS CLI: https://aws.amazon.com/cli/")
    print("2. Download SpaceNet 2 (Building Footprints) or SpaceNet 3 (Road Networks):")
    print("   aws s3 ls --no-sign-request s3://spacenet-dataset/spacenet/SN2_buildings/")
    print("3. Unpack PNG/GeoTIFF images into 'data/spacenet/' directory.")


if __name__ == "__main__":
    main()
