"""
Download the Kaggle road-sign dataset, keep only the train split, convert COCO to YOLO,
and register the canonical layout::

    road_sign/data/original_ds/{dataset_hash}/
        class_mapping.json
        dataset_config.json
        images/{image_id}.png
        labels/{image_id}.txt
"""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import kagglehub
from dotenv import load_dotenv
from tqdm import tqdm

from home_made_od.general.path_utils import (
    CLASS_MAPPING_FILENAME,
    ROAD_SIGN_KAGGLE_SOURCE,
    DATASET_VERSION_ORIGINAL,
    dataset_images_dir,
    dataset_labels_dir,
    dataset_root,
    original_data_sanity_dir,
    register_original_dataset_from_dir,
    road_sign_data_root,
)

STAGING_DIR_NAME = "_download_staging"
TRAIN_DIR_NAME = "train"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def download_dataset(staging_dir: Path) -> Path:
    load_dotenv()

    print("Downloading dataset...")
    cache_path = Path(kagglehub.competition_download("traffic-sign-object-detection-challenge"))
    print(f"Kaggle cache: {cache_path}")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    for item in cache_path.iterdir():
        dst = staging_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    print(f"Staging copy: {staging_dir}")
    return staging_dir


def prune_staging_dir(staging_dir: Path) -> None:
    """Keep only ``train/`` and ``class_mapping.json``; discard everything else."""
    kept = {TRAIN_DIR_NAME, CLASS_MAPPING_FILENAME}
    for item in list(staging_dir.iterdir()):
        if item.name in kept:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    print(f"Pruned staging dir (kept {sorted(kept)}): {staging_dir}")


def _find_train_coco_json(train_dir: Path) -> Path:
    candidates = [
        train_dir / "annotations.json",
        train_dir / "annotation.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = [p for p in train_dir.glob("*.json") if p.name != CLASS_MAPPING_FILENAME]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No COCO JSON found under {train_dir}")
    raise ValueError(f"Ambiguous COCO JSON files under {train_dir}: {matches}")


def _build_train_image_index(train_dir: Path) -> dict[str, Path]:
    """Map COCO ``file_name`` -> on-disk path (single directory scan)."""
    index: dict[str, Path] = {}
    for folder in (train_dir / "images", train_dir):
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index[path.name] = path
    return index


def _export_image_to_png(args: tuple[str, str]) -> bool:
    """
    Worker: copy PNG as-is, or decode JPEG/other once and write PNG.

    Must stay top-level for Windows multiprocessing pickling.
    """
    src_path, dst_path = args
    src = Path(src_path)
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst_path)
        return True

    img = cv2.imread(src_path)
    if img is None:
        return False
    # Low compression = faster writes; training only needs lossless-enough PNGs.
    cv2.imwrite(dst_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return True


def _format_yolo_labels(anns: list, width: int, height: int) -> str:
    lines: list[str] = []
    for ann in anns:
        class_id = ann["category_id"]
        x_min, y_min, bw, bh = ann["bbox"]
        x_center = (x_min + bw / 2) / width
        y_center = (y_min + bh / 2) / height
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {bw / width:.6f} {bh / height:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def build_canonical_dataset(staging_dir: Path) -> None:
    """
    Convert train COCO annotations to YOLO and write flat ``images/`` + ``labels/``.
    Removes the ``train/`` folder when done.
    """
    train_dir = staging_dir / TRAIN_DIR_NAME
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Expected {train_dir} after pruning.")

    coco_path = _find_train_coco_json(train_dir)
    print(f"Converting {coco_path}...")

    with open(coco_path, encoding="utf-8") as f:
        data = json.load(f)

    images_out = staging_dir / "images"
    labels_out = staging_dir / "labels"
    if images_out.exists():
        shutil.rmtree(images_out)
    if labels_out.exists():
        shutil.rmtree(labels_out)
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    categories = data.get("categories", [])
    id_to_name = {cat["id"]: cat["name"] for cat in categories}
    with open(staging_dir / CLASS_MAPPING_FILENAME, "w", encoding="utf-8") as f:
        json.dump(id_to_name, f, indent=4)
    print(f"Saved class mapping to {staging_dir / CLASS_MAPPING_FILENAME}")

    images = {img["id"]: img for img in data["images"]}
    img_annots: dict[int, list] = {}
    for ann in data["annotations"]:
        img_annots.setdefault(ann["image_id"], []).append(ann)

    src_index = _build_train_image_index(train_dir)
    image_tasks: list[tuple[str, str]] = []
    missing: list[str] = []

    for img_id, anns in img_annots.items():
        img_info = images[img_id]
        file_name = img_info["file_name"]
        image_id = Path(file_name).stem

        src_img = src_index.get(file_name)
        if src_img is None:
            missing.append(file_name)
            continue

        image_tasks.append(
            (str(src_img), str(images_out / f"{image_id}.png"))
        )
        label_text = _format_yolo_labels(anns, img_info["width"], img_info["height"])
        (labels_out / f"{image_id}.txt").write_text(label_text, encoding="utf-8")

    if missing:
        print(f"Warning: {len(missing)} images not found (first: {missing[0]})")

    workers = min(16, os.cpu_count() or 4)
    chunk = max(1, len(image_tasks) // (workers * 4))
    print(f"Exporting {len(image_tasks)} images with {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(
            tqdm(
                executor.map(_export_image_to_png, image_tasks, chunksize=chunk),
                total=len(image_tasks),
                desc="Writing images",
            )
        )
    saved = sum(results)

    shutil.rmtree(train_dir)
    print(f"Wrote {saved} image/label pairs under {images_out} and {labels_out}")
    print("Removed train/ staging folder.")


def visualize_original_data(data_dir: Path, max_images: int = 10) -> None:
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"Skipping visualization; expected {images_dir} and {labels_dir}.")
        return

    class_mapping: dict[int | str, str] = {}
    mapping_path = data_dir / CLASS_MAPPING_FILENAME
    if mapping_path.is_file():
        with open(mapping_path, encoding="utf-8") as f:
            raw = json.load(f)
        class_mapping = {int(k) if str(k).isdigit() else k: v for k, v in raw.items()}

    out_dir = original_data_sanity_dir()
    print(f"Visualizing up to {max_images} samples to {out_dir}")

    count = 0
    for img_path in sorted(images_dir.iterdir()):
        if count >= max_images:
            break
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.is_file():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                name = class_mapping.get(cls_id, str(cls_id))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    img,
                    name,
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

        cv2.imwrite(str(out_dir / f"org_{img_path.name}"), img)
        count += 1

    print("Finished visualizing original data.")


def main() -> None:
    staging_dir = road_sign_data_root() / STAGING_DIR_NAME

    download_dataset(staging_dir)
    prune_staging_dir(staging_dir)
    build_canonical_dataset(staging_dir)

    dataset_hash = register_original_dataset_from_dir(
        staging_dir,
        source=ROAD_SIGN_KAGGLE_SOURCE,
    )
    final_root = dataset_root(DATASET_VERSION_ORIGINAL, dataset_hash)
    print(f"Registered original dataset: {dataset_hash}")
    print(f"Location: {final_root}")
    print(f"Images: {dataset_images_dir(DATASET_VERSION_ORIGINAL, dataset_hash)}")
    print(f"Labels: {dataset_labels_dir(DATASET_VERSION_ORIGINAL, dataset_hash)}")

    visualize_original_data(final_root, max_images=5)


if __name__ == "__main__":
    main()
