import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Path setup logic
current_dir = os.path.dirname(os.path.abspath(__file__))
while 'road_sign' not in os.listdir(current_dir):
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:
        break
    current_dir = parent_dir

road_sign_root = os.path.join(current_dir, 'road_sign')
sys.path.insert(0, os.path.join(road_sign_root, 'scripts', 'training', 'patch_based'))

# Import the core logic
from prepare_patches import generate_positive_patches, update_labels_for_patch

# Use the same default config as the preparation script
DEFAULT_CONFIG = {
    "scales": [512, 1024, 2048],
    "min_scale_ratio_threshold": 0.5,
    "min_visibility_threshold": 0.4,
    "K2": 2,
    "target_size": [512, 512],
    "num_anchors": 5
}

def run_dry_run(source_data_dir, config=None):
    if config is None:
        config = DEFAULT_CONFIG
        
    print("--- Starting Dry Run for Adaptive Patching Algorithm ---")
    print(f"Configuration: {config}")
    
    img_dir = os.path.join(source_data_dir, 'train', 'images')
    lbl_dir = os.path.join(source_data_dir, 'labels', 'annotations')
    
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"Source directories missing. Checked: {img_dir} and {lbl_dir}")
        return
        
    pairs = []
    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            stem = Path(img_file).stem
            label_file = os.path.join(lbl_dir, f"{stem}.txt")
            if os.path.exists(label_file):
                pairs.append((os.path.join(img_dir, img_file), label_file))
                
    if not pairs:
        print("No valid image/label pairs found.")
        return

    total_orig_objects = 0
    patch_size_counts = {s: 0 for s in config["scales"]}
    final_area_ratios = []
    
    for img_path, lbl_path in tqdm(pairs, desc="Simulating Patches"):
        with Image.open(img_path) as img:
            w, h = img.size
            
        gt_boxes = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = [float(x) for x in line.split()]
                if len(parts) >= 5:
                    cls, ncx, ncy, nw, nh = parts[:5]
                    gt_boxes.append([cls, ncx * w, ncy * h, nw * w, nh * h])
                    
        if not gt_boxes:
            continue
            
        gt_boxes = np.array(gt_boxes)
        total_orig_objects += len(gt_boxes)

        # Simulate positive patches generation
        pos_coords = generate_positive_patches((h, w), gt_boxes, config)
        
        for coords in pos_coords:
            x1, y1, x2, y2 = coords
            patch_w = x2 - x1
            
            # Record the patch size chosen
            if patch_w in patch_size_counts:
                patch_size_counts[patch_w] += 1
            else:
                # Fallback for dynamic scales
                patch_size_counts.setdefault(patch_w, 0)
                patch_size_counts[patch_w] += 1
                
            # Simulate updating labels (this handles the visibility threshold now!)
            patch_labels = update_labels_for_patch(coords, gt_boxes, config)
            
            for lbl in patch_labels:
                _, _, _, w_norm, h_norm = lbl
                final_area_ratios.append(w_norm * h_norm)

    final_area_ratios = np.array(final_area_ratios)
    
    print("\n--- Dry Run Statistics ---")
    print(f"Total Original Objects: {total_orig_objects}")
    print(f"Total Positive Patches Generated: {sum(patch_size_counts.values())}")
    print(f"Total Valid Objects inside Patches: {len(final_area_ratios)}")
    print("\nPatch Size Distribution:")
    for size in sorted(patch_size_counts.keys()):
        print(f"  {size}x{size}: {patch_size_counts[size]} patches")
        
    print("\nFinal Object Area Ratios (Object Area / Patch Area):")
    if len(final_area_ratios) > 0:
        print(f"  Min: {np.min(final_area_ratios):.6f}")
        print(f"  Max: {np.max(final_area_ratios):.6f}")
        print(f"  Mean: {np.mean(final_area_ratios):.6f}")

        print("\n--- Area Ratio Quantiles (5 to 100) ---")
        for q in range(5, 101, 5):
            val = np.percentile(final_area_ratios, q)
            print(f"  {q:3d}th Percentile: {val:.6f}")

        print("\n--- Patches per Area Ratio Bin (step 0.05) ---")
        bins = np.arange(0.0, 1.05, 0.05)
        counts, _ = np.histogram(final_area_ratios, bins=bins)
        for i in range(len(counts)):
            lower = bins[i]
            upper = bins[i+1]
            print(f"  [{lower:.2f}, {upper:.2f}): {counts[i]:4d} patches")

        print(f"\n  Objects > 50% of patch: {np.sum(final_area_ratios > 0.5)} ({(np.sum(final_area_ratios > 0.5) / len(final_area_ratios)) * 100:.1f}%)")
        print(f"  Objects > 20% of patch: {np.sum(final_area_ratios > 0.2)} ({(np.sum(final_area_ratios > 0.2) / len(final_area_ratios)) * 100:.1f}%)")
        print(f"  Objects < 1% of patch: {np.sum(final_area_ratios < 0.01)} ({(np.sum(final_area_ratios < 0.01) / len(final_area_ratios)) * 100:.1f}%)")
        print(f"  Objects < 0.1% (sub-pixel danger): {np.sum(final_area_ratios < 0.001)} ({(np.sum(final_area_ratios < 0.001) / len(final_area_ratios)) * 100:.1f}%)")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    hist_data = np.clip(final_area_ratios, 0, 1.0)
    plt.hist(hist_data, bins=np.arange(0, 1.05, 0.05), color='lightgreen', edgecolor='black')
    plt.title('Expected Final Area Ratio Distribution (Adaptive Patching)')
    plt.xlabel('Object Area / Final Patch Area')
    plt.ylabel('Frequency')
    plt.xlim(0, 1.0)
    plt.xticks(np.arange(0, 1.05, 0.05), rotation=45)
    
    plt.tight_layout()
    artifacts_dir = Path(road_sign_root) / 'artifacts' / 'patch_visualization'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plot_path = artifacts_dir / 'adaptive_patch_dry_run_ratio_dist_updated.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    SOURCE_DIR = os.path.join(road_sign_root, 'data')
    run_dry_run(SOURCE_DIR, DEFAULT_CONFIG)
