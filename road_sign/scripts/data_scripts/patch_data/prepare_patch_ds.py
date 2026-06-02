"""
Build the adaptive patch dataset under ``road_sign/data/patch_based/{dataset_hash}/``.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from home_made_od.general.path_utils import (
    DATASET_VERSION_ORIGINAL,
    DATASET_VERSION_PATCH,
    compute_patch_dataset_hash,
    dataset_root,
    get_train_image_label_pairs,
    iter_train_image_label_pairs,
    legacy_flat_data_dir,
    patch_config_path,
    resolve_latest_dataset_hash,
    road_sign_root,
    train_images_dir,
    train_labels_dir,
    write_dataset_config,
)
from mypt.code_utils.pytorch_utils import seed_everything

from prepare_patches import (
    generate_negative_patches,
    generate_positive_patches,
    update_labels_for_patch,
)

DEFAULT_PATCH_CONFIG: dict = {
    "scales": [512, 1024, 2048],
    "min_scale_ratio_threshold": 0.5,
    "min_visibility_threshold": 0.4,
    "K2": 2,
    "target_size": [512, 512],
    "num_anchors": 5,
}


def process_single_image(args: tuple) -> bool:
    img_path, label_path, target_img_dir, target_label_dir, config = args

    img = cv2.imread(img_path)
    if img is None:
        return False

    h, w = img.shape[:2]
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parts = [float(x) for x in line.split()]
                if len(parts) >= 5:
                    cls, ncx, ncy, nw, nh = parts[:5]
                    gt_boxes.append([cls, ncx * w, ncy * h, nw * w, nh * h])
    gt_boxes = np.array(gt_boxes)

    pos_coords = generate_positive_patches((h, w), gt_boxes, config)
    neg_coords = generate_negative_patches(img, gt_boxes, config)
    all_patches = [("pos", c) for c in pos_coords] + [("neg", c) for c in neg_coords]

    base_name = Path(img_path).stem
    img_subdir = os.path.join(target_img_dir, base_name)
    lbl_subdir = os.path.join(target_label_dir, base_name)
    os.makedirs(img_subdir, exist_ok=True)
    os.makedirs(lbl_subdir, exist_ok=True)

    for i, (type_prefix, coords) in enumerate(all_patches):
        x1, y1, x2, y2 = coords
        patch_img = img[y1:y2, x1:x2]
        if patch_img.size == 0:
            continue

        patch_resized = cv2.resize(
            patch_img,
            tuple(config["target_size"]),
            interpolation=cv2.INTER_LINEAR,
        )
        patch_filename = f"patch_{i}_{type_prefix}.png"
        cv2.imwrite(os.path.join(img_subdir, patch_filename), patch_resized)

        patch_labels = update_labels_for_patch(coords, gt_boxes, config)
        out_label_path = os.path.join(lbl_subdir, f"patch_{i}_{type_prefix}.txt")
        with open(out_label_path, "w", encoding="utf-8") as f_out:
            for lbl in patch_labels:
                cls_id, cx_norm, cy_norm, w_norm, h_norm = lbl
                f_out.write(
                    f"{int(cls_id)} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                )

    return True


def prepare_patch_dataset(
    source_version: str,
    source_dataset_hash: str,
    patch_config: dict | None = None,
) -> str:
    if patch_config is None:
        patch_config = dict(DEFAULT_PATCH_CONFIG)

    pairs = get_train_image_label_pairs(source_version, source_dataset_hash)  # type: ignore[arg-type]
    if not pairs:
        raise ValueError(
            f"No train pairs for {source_version}/{source_dataset_hash}"
        )

    dataset_hash = compute_patch_dataset_hash(source_dataset_hash, patch_config)
    target_root = dataset_root(DATASET_VERSION_PATCH, dataset_hash)
    target_img_dir = train_images_dir(DATASET_VERSION_PATCH, dataset_hash)
    target_label_dir = train_labels_dir(DATASET_VERSION_PATCH, dataset_hash)
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_label_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Patch dataset {dataset_hash} ---")
    print(f"Source: {source_version}/{source_dataset_hash}")
    print(f"Target: {target_root}")

    with open(patch_config_path(dataset_hash), "w", encoding="utf-8") as f:
        yaml.dump(patch_config, f)

    tasks = [
        (img, lbl, str(target_img_dir), str(target_label_dir), patch_config)
        for img, lbl in pairs
    ]

    successful_images = 0
    with ProcessPoolExecutor() as executor:
        for success in tqdm(executor.map(process_single_image, tasks), total=len(tasks)):
            if success:
                successful_images += 1

    write_dataset_config(
        DATASET_VERSION_PATCH,
        dataset_hash,
        {
            "version": DATASET_VERSION_PATCH,
            "dataset_hash": dataset_hash,
            "source_version": source_version,
            "source_dataset_hash": source_dataset_hash,
            "patch_config": patch_config,
            "source_image_count": successful_images,
        },
    )

    num_imgs = sum(len(files) for _, _, files in os.walk(target_img_dir))
    num_lbls = sum(len(files) for _, _, files in os.walk(target_label_dir))
    print(f"Processed {successful_images} source images.")
    print(f"Generated {num_imgs} patch images and {num_lbls} label files.")
    print(
        "Next: run create_split.py --dataset-version patch_based "
        f"--dataset-hash {dataset_hash} to create splits and anchors."
    )
    return dataset_hash


def import_legacy_patch_root(
    legacy_root: Path,
    source_dataset_hash: str,
    patch_config: dict | None = None,
) -> str:
    """Register legacy ``data_patch_adaptive/{config_hash}/`` into ``patch_based/{dataset_hash}/``."""
    if patch_config is None:
        config_path = legacy_root / "config.yaml"
        if config_path.is_file():
            with open(config_path, encoding="utf-8") as f:
                patch_config = yaml.safe_load(f)
        else:
            patch_config = dict(DEFAULT_PATCH_CONFIG)

    dataset_hash = compute_patch_dataset_hash(source_dataset_hash, patch_config)
    target_root = dataset_root(DATASET_VERSION_PATCH, dataset_hash)

    import shutil

    if legacy_root.resolve() != target_root.resolve():
        target_root.parent.mkdir(parents=True, exist_ok=True)
        if target_root.exists():
            shutil.rmtree(target_root)
        shutil.copytree(legacy_root, target_root)

    pairs = list(iter_train_image_label_pairs(DATASET_VERSION_PATCH, dataset_hash))
    write_dataset_config(
        DATASET_VERSION_PATCH,
        dataset_hash,
        {
            "version": DATASET_VERSION_PATCH,
            "dataset_hash": dataset_hash,
            "source_version": DATASET_VERSION_ORIGINAL,
            "source_dataset_hash": source_dataset_hash,
            "patch_config": patch_config,
            "imported_from_legacy": str(legacy_root),
            "source_image_count": len({Path(img).parent.name for img, _ in pairs}),
        },
    )
    print(f"Imported legacy patch data -> {target_root} ({dataset_hash})")
    return dataset_hash


def main() -> None:
    from home_made_od.general.path_utils import register_original_dataset_from_dir

    seed_everything(42)

    source_hash = resolve_latest_dataset_hash(DATASET_VERSION_ORIGINAL)
    if source_hash is None:
        legacy_original = legacy_flat_data_dir()
        if (legacy_original / "train" / "images").is_dir():
            print(f"Registering legacy original data at {legacy_original}")
            source_hash = register_original_dataset_from_dir(legacy_original, copy=True)
        else:
            raise FileNotFoundError("No original dataset found. Run prepare_data.py first.")

    if resolve_latest_dataset_hash(DATASET_VERSION_PATCH) is None:
        legacy_patch_root = road_sign_root() / "data_patch_adaptive"
        if legacy_patch_root.is_dir():
            for child in sorted(legacy_patch_root.iterdir()):
                if child.is_dir() and len(child.name) == 32:
                    import_legacy_patch_root(child, source_hash)
                    return
            import_legacy_patch_root(legacy_patch_root, source_hash)
            return

    prepare_patch_dataset(
        DATASET_VERSION_ORIGINAL,
        source_hash,
        patch_config=dict(DEFAULT_PATCH_CONFIG),
    )


if __name__ == "__main__":
    main()
