import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path setup logic
current_dir = os.path.dirname(os.path.abspath(__file__))
while 'road_sign' not in os.listdir(current_dir):
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:
        break
    current_dir = parent_dir

road_sign_root = os.path.join(current_dir, 'road_sign')

def analyze_object_sizes(data_dir):
    coco_files = list(Path(data_dir).rglob("*annotations.json"))
    if not coco_files:
        print("No COCO JSON files found.")
        return
        
    artifacts_dir = Path(road_sign_root) / 'artifacts' / 'object_sizes'
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for coco_file in coco_files:
        print(f"\n--- Object Size Analysis for {coco_file.name} ---")
        with open(coco_file, 'r') as f:
            data = json.load(f)

        widths = []
        heights = []
        areas = []
        aspect_ratios = []

        for ann in data['annotations']:
            # COCO bbox format is [x_min, y_min, width, height]
            bbox = ann['bbox']
            if len(bbox) == 4:
                w, h = bbox[2], bbox[3]
                widths.append(w)
                heights.append(h)
                areas.append(w * h)
                if h > 0:
                    aspect_ratios.append(w / h)

        if not widths:
            print("No bounding boxes found.")
            continue

        widths = np.array(widths)
        heights = np.array(heights)
        areas = np.array(areas)
        aspect_ratios = np.array(aspect_ratios)

        print(f"Total Objects: {len(widths)}")
        print("\nWidths (pixels):")
        print(f"  Min: {np.min(widths):.1f}")
        print(f"  Max: {np.max(widths):.1f}")
        print(f"  Mean: {np.mean(widths):.1f}")
        print(f"  Median: {np.median(widths):.1f}")
        print(f"  90th Percentile: {np.percentile(widths, 90):.1f}")

        print("\nHeights (pixels):")
        print(f"  Min: {np.min(heights):.1f}")
        print(f"  Max: {np.max(heights):.1f}")
        print(f"  Mean: {np.mean(heights):.1f}")
        print(f"  Median: {np.median(heights):.1f}")
        print(f"  90th Percentile: {np.percentile(heights, 90):.1f}")
        
        print("\nAspect Ratios (Width / Height):")
        print(f"  Mean: {np.mean(aspect_ratios):.2f}")
        print(f"  Median: {np.median(aspect_ratios):.2f}")

        # Area Ratio Analysis
        patch_sizes = [512, 1024, 2048]
        print("\n--- Area Ratio Analysis ---")

        for patch_size in patch_sizes:
            patch_area = patch_size * patch_size
            area_ratios = areas / patch_area

            print(f"\nPatch Size: {patch_size}x{patch_size}")
            print(f"  Ratio Min: {np.min(area_ratios):.6f}")
            print(f"  Ratio Max: {np.max(area_ratios):.6f}")
            print(f"  Ratio Mean: {np.mean(area_ratios):.6f}")
            print(f"  Ratio Median: {np.median(area_ratios):.6f}")
            print(f"  Ratio 90th Pct: {np.percentile(area_ratios, 90):.6f}")
            print(f"  Objects > 50% of patch: {np.sum(area_ratios > 0.5)}")
            print(f"  Objects > 20% of patch: {np.sum(area_ratios > 0.2)}")
            print(f"  Objects < 1% of patch: {np.sum(area_ratios < 0.01)}")

            plt.figure(figsize=(8, 6))
            # Clip the data slightly for better histogram visualization if there are extreme outliers
            hist_data = np.clip(area_ratios, 0, 1.0) 
            plt.hist(hist_data, bins=np.arange(0, 1.05, 0.05), color='skyblue', edgecolor='black')
            plt.title(f'Area Ratio dist for {patch_size}x{patch_size}')
            plt.xlabel('Object Area / Patch Area')
            plt.ylabel('Frequency')
            plt.xlim(0, 1.0)
            plt.xticks(np.arange(0, 1.05, 0.05), rotation=45)

            plt.tight_layout()
            plot_path = artifacts_dir / f'area_ratio_analysis_{coco_file.stem}_{patch_size}.png'
            plt.savefig(plot_path)
            plt.close()
            print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    data_path = os.path.join(road_sign_root, 'data')
    analyze_object_sizes(data_path)
