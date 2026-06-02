"""
Dry-run statistics for the adaptive patching algorithm on the original (or resized) dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from home_made_od.general.path_utils import (
    DATASET_VERSION_ORIGINAL,
    DATASET_VERSION_PATCH,
    get_train_image_label_pairs,
    legacy_flat_data_dir,
    patch_visualization_dir,
    resolve_latest_dataset_hash,
)
from prepare_patches import generate_positive_patches, update_labels_for_patch

DEFAULT_PATCH_CONFIG = {
    "scales": [512, 1024, 2048],
    "min_scale_ratio_threshold": 0.5,
    "min_visibility_threshold": 0.4,
    "K2": 2,
    "target_size": [512, 512],
    "num_anchors": 5,
}


def run_dry_run(
    source_version: str,
    source_dataset_hash: str,
    config: dict | None = None,
) -> None:
    if config is None:
        config = dict(DEFAULT_PATCH_CONFIG)

    print("--- Adaptive patching dry run ---")
    print(f"Source: {source_version}/{source_dataset_hash}")
    print(f"Configuration: {config}")

    pairs = get_train_image_label_pairs(source_version, source_dataset_hash)  # type: ignore[arg-type]
    if not pairs:
        raise ValueError("No image/label pairs found for dry run.")

    patch_size_counts: dict[int, int] = {s: 0 for s in config["scales"]}
    final_area_ratios: list[float] = []

    for img_path, lbl_path in tqdm(pairs, desc="Simulating patches"):
        with Image.open(img_path) as img:
            w, h = img.size

        gt_boxes = []
        with open(lbl_path, encoding="utf-8") as f:
            for line in f:
                parts = [float(x) for x in line.split()]
                if len(parts) >= 5:
                    cls, ncx, ncy, nw, nh = parts[:5]
                    gt_boxes.append([cls, ncx * w, ncy * h, nw * w, nh * h])
        gt_boxes = np.array(gt_boxes)

        pos_coords = generate_positive_patches((h, w), gt_boxes, config)
        for coords in pos_coords:
            x1, y1, x2, y2 = coords
            patch_w = x2 - x1
            patch_size_counts[patch_w] = patch_size_counts.get(patch_w, 0) + 1

            patch_labels = update_labels_for_patch(coords, gt_boxes, config)
            pw, ph = x2 - x1, y2 - y1
            for lbl in patch_labels:
                _, _, _, w_norm, h_norm = lbl
                final_area_ratios.append(w_norm * h_norm)

    print(f"Total positive patches: {sum(patch_size_counts.values())}")
    print(f"Valid objects in patches: {len(final_area_ratios)}")
    print("\nPatch size distribution:")
    for size in sorted(patch_size_counts.keys()):
        print(f"  {size}x{size}: {patch_size_counts[size]} patches")

    if final_area_ratios:
        ratios = np.array(final_area_ratios)
        print("\nObject area / patch area ratios:")
        print(f"  Mean: {ratios.mean():.4f}")
        print(f"  Median: {np.median(ratios):.4f}")
        print(f"  > 50% of patch: {np.sum(ratios > 0.5)}")
        print(f"  > 20% of patch: {np.sum(ratios > 0.2)}")
        print(f"  < 1% of patch: {np.sum(ratios < 0.01)}")

        out_dir = patch_visualization_dir()
        plt.figure(figsize=(10, 6))
        plt.hist(ratios, bins=50, color="skyblue", edgecolor="black")
        plt.title("Expected area ratio distribution (dry run)")
        plt.xlabel("Object area / patch area")
        plt.ylabel("Count")
        plt.grid(axis="y", alpha=0.75)
        plot_path = out_dir / "adaptive_patch_dry_run_ratio_dist.png"
        plt.savefig(plot_path)
        print(f"\nSaved plot to {plot_path}")


def main() -> None:
    from home_made_od.general.path_utils import register_original_dataset_from_dir

    source_hash = resolve_latest_dataset_hash(DATASET_VERSION_ORIGINAL)
    if source_hash is None:
        legacy = legacy_flat_data_dir()
        if (legacy / "train" / "images").is_dir():
            source_hash = register_original_dataset_from_dir(legacy, copy=True)
        else:
            raise FileNotFoundError("No original dataset found.")

    run_dry_run(DATASET_VERSION_ORIGINAL, source_hash)


if __name__ == "__main__":
    main()
