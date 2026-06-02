"""
Resize the road-sign training set to a fixed resolution and register it under
``road_sign/data/resized_ds/{dataset_hash}/``.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

from home_made_od.general.path_utils import (
    DATASET_VERSION_ORIGINAL,
    DATASET_VERSION_RESIZED,
    compute_resized_dataset_hash,
    dataset_root,
    get_train_image_label_pairs,
    iter_train_image_label_pairs,
    legacy_flat_data_dir,
    resolve_latest_dataset_hash,
    train_images_dir,
    train_labels_dir,
    write_dataset_config,
)


def process_single_pair(args: tuple) -> bool:
    img_path, label_path, target_img_dir, target_label_dir, target_size = args

    img = cv2.imread(img_path)
    if img is None:
        return False

    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    filename = Path(img_path).name
    out_img_path = Path(target_img_dir) / filename
    cv2.imwrite(str(out_img_path), img_resized)

    label_filename = Path(label_path).name
    out_label_path = Path(target_label_dir) / label_filename
    shutil.copy2(label_path, out_label_path)
    return True


def resize_dataset(
    source_version: str,
    source_dataset_hash: str,
    target_size: tuple[int, int] = (512, 512),
) -> str:
    source_root = dataset_root(source_version, source_dataset_hash)  # type: ignore[arg-type]
    pairs = get_train_image_label_pairs(source_version, source_dataset_hash)  # type: ignore[arg-type]
    if not pairs:
        raise ValueError(f"No train image/label pairs found under {source_root}")

    dataset_hash = compute_resized_dataset_hash(source_dataset_hash, target_size)
    target_root = dataset_root(DATASET_VERSION_RESIZED, dataset_hash)
    target_img_dir = train_images_dir(DATASET_VERSION_RESIZED, dataset_hash)
    target_label_dir = train_labels_dir(DATASET_VERSION_RESIZED, dataset_hash)
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_label_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resizing {source_version}/{source_dataset_hash} -> resized/{dataset_hash}")
    print(f"Target size: {target_size}")
    print(f"Output: {target_root}")

    tasks = [
        (img, lbl, str(target_img_dir), str(target_label_dir), target_size)
        for img, lbl in pairs
    ]

    successful = 0
    with ProcessPoolExecutor() as executor:
        for ok in tqdm(executor.map(process_single_pair, tasks), total=len(tasks)):
            if ok:
                successful += 1

    write_dataset_config(
        DATASET_VERSION_RESIZED,
        dataset_hash,
        {
            "version": DATASET_VERSION_RESIZED,
            "dataset_hash": dataset_hash,
            "source_version": source_version,
            "source_dataset_hash": source_dataset_hash,
            "target_size": list(target_size),
            "train_image_count": successful,
        },
    )

    print(f"Successfully resized {successful}/{len(pairs)} images.")
    return dataset_hash


def import_legacy_flat_resized(
    legacy_dir: Path,
    source_dataset_hash: str,
    target_size: tuple[int, int] = (512, 512),
) -> str:
    """
    Register an existing flat resized directory (e.g. legacy ``road_sign/data_512``)
    into the hashed ``resized_ds`` layout without re-encoding images.
    """
    dataset_hash = compute_resized_dataset_hash(source_dataset_hash, target_size)
    target_root = dataset_root(DATASET_VERSION_RESIZED, dataset_hash)

    if legacy_dir.resolve() != target_root.resolve():
        target_root.parent.mkdir(parents=True, exist_ok=True)
        if target_root.exists():
            shutil.rmtree(target_root)
        shutil.copytree(legacy_dir, target_root)

    pairs = list(iter_train_image_label_pairs(DATASET_VERSION_RESIZED, dataset_hash))
    write_dataset_config(
        DATASET_VERSION_RESIZED,
        dataset_hash,
        {
            "version": DATASET_VERSION_RESIZED,
            "dataset_hash": dataset_hash,
            "source_version": DATASET_VERSION_ORIGINAL,
            "source_dataset_hash": source_dataset_hash,
            "target_size": list(target_size),
            "train_image_count": len(pairs),
            "imported_from_legacy": str(legacy_dir),
        },
    )
    print(f"Imported legacy resized data -> {target_root} ({dataset_hash})")
    return dataset_hash


def main() -> None:
    from home_made_od.general.path_utils import register_original_dataset_from_dir

    target_size = (512, 512)
    source_hash = resolve_latest_dataset_hash(DATASET_VERSION_ORIGINAL)

    if source_hash is None:
        legacy_original = legacy_flat_data_dir()
        if (legacy_original / "train" / "images").is_dir():
            print(f"No hashed original dataset found; registering legacy {legacy_original}")
            source_hash = register_original_dataset_from_dir(legacy_original, copy=True)
        else:
            raise FileNotFoundError(
                "No original dataset found. Run prepare_data.py first."
            )

    if resolve_latest_dataset_hash(DATASET_VERSION_RESIZED) is None:
        legacy_resized = legacy_flat_data_dir().parent / "data_512"
        if legacy_resized.is_dir():
            import_legacy_flat_resized(legacy_resized, source_hash, target_size=target_size)
            return

    resize_dataset(
        DATASET_VERSION_ORIGINAL,
        source_hash,
        target_size=target_size,
    )


if __name__ == "__main__":
    main()
