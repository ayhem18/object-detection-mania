import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any

from dl_lib.common_tools.path_utils import get_data_dir

from dl_lib.etalon_object_detection.modules.path_layout import (
    full_dataset_analysis_dir,
    images_dir,
    labels_data_dir,
    master_labels_path,
    resolve_latest_dataset_hash,
)

def resolve_paths(dataset_hash: str) -> Dict[str, Path]:
    """Resolves all required project paths based on the dataset hash."""
    return {
        "root": get_data_dir().parent,
        "images": images_dir(),
        "cache": labels_data_dir(dataset_hash),
        "master_json": master_labels_path(dataset_hash),
        "artifacts": full_dataset_analysis_dir(dataset_hash, "image_shapes"),
    }

def imread_unicode(path: str) -> np.ndarray:
    """Read an image with Unicode path support on Windows."""
    try:
        with open(path, 'rb') as f:
            chunk = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(chunk, cv2.IMREAD_UNCHANGED)
        return img
    except Exception:
        return None

def run_image_shape_analysis(config: Dict[str, Any]):
    dataset_hash = config["dataset_hash"]
    paths = resolve_paths(dataset_hash)
    
    if not paths["master_json"].exists():
        print(f"Error: {paths['master_json']} not found. Run obj_det_prepare.py first.")
        return

    with open(paths["master_json"], 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    samples = data.get("samples", [])
    print(f"Analyzing shapes for {len(samples)} images in dataset: {dataset_hash}")

    x_dims = []
    y_dims = []
    
    for s in tqdm(samples, desc="Reading Image Shapes"):
        img_path = paths["images"] / s['img_path']
        
        # We don't need to load the full image, just the header, 
        # but for simplicity and robustness with different formats, imread is okay here.
        # Alternatively, we could read from the cached label JSONs if they store it.
        
        lbl_path = paths["cache"] / s['lbl_path']
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                lbl_data = json.load(f)
            x_dims.append(lbl_data.get("image_x_dim"))
            y_dims.append(lbl_data.get("image_y_dim"))
        else:
            # Fallback to reading file if not in cache
            img = imread_unicode(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                x_dims.append(w)
                y_dims.append(h)

    x_dims = np.array([x for x in x_dims if x is not None])
    y_dims = np.array([y for y in y_dims if y is not None])

    if len(x_dims) == 0:
        print("No image data found.")
        return

    # 1. Console Summary
    print("\n" + "="*50)
    print("           IMAGE SHAPE STATISTICS           ")
    print("="*50)
    print(f"{'Metric':<15} | {'X-Dim':<12} | {'Y-Dim':<12}")
    print("-" * 50)
    print(f"{'Mean':<15} | {np.mean(x_dims):>12.2f} | {np.mean(y_dims):>12.2f}")
    print(f"{'Median':<15} | {np.median(x_dims):>12.2f} | {np.median(y_dims):>12.2f}")
    print(f"{'Min':<15} | {np.min(x_dims):>12.2f} | {np.min(y_dims):>12.2f}")
    print(f"{'Max':<15} | {np.max(x_dims):>12.2f} | {np.max(y_dims):>12.2f}")
    print(f"{'Std Dev':<15} | {np.std(x_dims):>12.2f} | {np.std(y_dims):>12.2f}")
    
    # Most common shapes
    unique_shapes, counts = np.unique(np.stack([x_dims, y_dims], axis=1), axis=0, return_counts=True)
    sorted_idx = np.argsort(-counts)
    
    print("-" * 50)
    print("Top 3 most common shapes (X x Y):")
    for i in range(min(3, len(unique_shapes))):
        idx = sorted_idx[i]
        shape = unique_shapes[idx]
        print(f"  {shape[0]} x {shape[1]}: {counts[idx]} images")
    print("="*50 + "\n")

    # 2. Visualizations
    paths["artifacts"].mkdir(parents=True, exist_ok=True)
    
    # 2.1 Histograms
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(x_dims, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"X-Dim Distribution")
    plt.xlabel("Pixels")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(y_dims, bins=30, color='salmon', edgecolor='black')
    plt.title(f"Y-Dim Distribution")
    plt.xlabel("Pixels")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(paths["artifacts"] / "shape_histograms.png")
    plt.close()

    # 2.2 Aspect Ratio Distribution
    aspect_ratios = x_dims / y_dims
    plt.figure(figsize=(8, 6))
    plt.hist(aspect_ratios, bins=30, color='lightgreen', edgecolor='black')
    plt.axvline(1.0, color='red', linestyle='--', label='Square')
    plt.title("Aspect Ratio Distribution (X/Y)")
    plt.xlabel("Ratio")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(paths["artifacts"] / "aspect_ratio_dist.png")
    plt.close()

    print(f"Visualizations saved to: {paths['artifacts']}")

def main():
    config = {
        "dataset_hash": "latest"
    }
    
    if config["dataset_hash"] == "latest":
        latest = resolve_latest_dataset_hash()
        if latest:
            config["dataset_hash"] = latest
            print(f"Using latest dataset hash: {config['dataset_hash']}")
    
    run_image_shape_analysis(config)

if __name__ == "__main__":
    main()
