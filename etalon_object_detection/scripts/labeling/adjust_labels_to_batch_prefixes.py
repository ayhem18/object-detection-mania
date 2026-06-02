import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

from dl_lib.common_tools.path_utils import get_data_dir
from dl_lib.etalon_object_detection.modules.path_layout import get_data_maps

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

CVAT_PREFIX = "digital_lab_wired_etalons/data_as_images"
YOLO_CLASS_ID_WIRED = 0


def infer_location(index: Dict[str, str]) -> str:
    sample_key = next(iter(index))
    if sample_key.startswith("/mnt") or sample_key.startswith("\\\\"):
        return "remote"
    return "local"


def build_dcm_name_to_batch_map(
    index: Dict[str, str],
    data_maps: Dict[str, str],
    images_dir: Path,
) -> Dict[str, str]:
    """
    Maps cleaned DCM folder names to batch prefixes.

    Uses index.json DCM source paths first, then falls back to the data_as_images
    folder layout when index values already include a batch prefix.
    """
    sorted_sources = sorted(data_maps.keys(), key=len, reverse=True)
    mapping: Dict[str, str] = {}

    for dcm_path, rel_value in index.items():
        if "/" in rel_value:
            batch, dcm_name = rel_value.split("/", 1)
            mapping[dcm_name] = batch
            continue

        dcm_name = rel_value
        for source_dir in sorted_sources:
            if dcm_path.startswith(source_dir):
                mapping[dcm_name] = data_maps[source_dir]
                break

    for batch_dir in images_dir.iterdir():
        if not batch_dir.is_dir():
            continue
        has_pngs = any(child.suffix.lower() == ".png" for child in batch_dir.iterdir() if child.is_file())
        if has_pngs:
            continue
        for dcm_dir in batch_dir.iterdir():
            if dcm_dir.is_dir():
                mapping.setdefault(dcm_dir.name, batch_dir.name)

    return mapping


def copy_labels_with_batch_prefixes(
    source_labels_dir: Path,
    output_labels_dir: Path,
    dcm_to_batch: Dict[str, str],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Copies label files into ``output_labels_dir/{batch}/{dcm_name}/frame_xxx.txt``.

    Returns processed frame entries for CVAT train.txt and a list of skipped DCM names.
    """
    if output_labels_dir.exists():
        shutil.rmtree(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    processed_frames: List[Tuple[str, str]] = []
    skipped_dcm_names: List[str] = []

    for dcm_dir in sorted(source_labels_dir.iterdir()):
        if not dcm_dir.is_dir():
            continue

        dcm_name = dcm_dir.name
        batch = dcm_to_batch.get(dcm_name)
        if batch is None:
            skipped_dcm_names.append(dcm_name)
            logger.warning("No batch mapping found for '%s'. Skipping.", dcm_name)
            continue

        rel_dcm = f"{batch}/{dcm_name}"
        dest_dcm_dir = output_labels_dir / batch / dcm_name
        dest_dcm_dir.mkdir(parents=True, exist_ok=True)

        label_files = sorted(
            f for f in dcm_dir.iterdir() if f.is_file() and f.name.startswith("frame_") and f.suffix == ".txt"
        )
        if not label_files:
            logger.warning("No label files found in '%s'. Skipping.", dcm_dir)
            continue

        for label_file in label_files:
            shutil.copy2(label_file, dest_dcm_dir / label_file.name)
            processed_frames.append((rel_dcm, f"{label_file.stem}.png"))

    return processed_frames, skipped_dcm_names


def create_cvat_zip(batch_data_dir: str, output_zip: str) -> None:
    print(f"Creating ZIP archive: {output_zip}")
    with zipfile.ZipFile(output_zip, "w") as zf:
        for root, _, files in os.walk(batch_data_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path_in_zip = os.path.relpath(abs_path, batch_data_dir)
                zf.write(abs_path, rel_path_in_zip)


def package_labels_for_cvat(
    labels_dir: Path,
    processed_frames: List[Tuple[str, str]],
    output_zip: Path,
) -> None:
    """Packages batch-prefixed labels into a CVAT/YOLO upload zip."""
    temp_root = labels_dir.parent / "cvat_upload_temp"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    dest_labels_dir = temp_root / "labels" / "train" / CVAT_PREFIX
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    for item in labels_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, dest_labels_dir / item.name)

    train_txt_path = temp_root / "train.txt"
    with open(train_txt_path, "w", encoding="utf-8") as f:
        for rel_dcm, frame_name in processed_frames:
            line = f"data/images/train/{CVAT_PREFIX}/{rel_dcm}/{frame_name}".replace("\\", "/")
            f.write(line + "\n")

    data_yaml_content = f"names:\n  {YOLO_CLASS_ID_WIRED}: wired\npath: .\ntrain: train.txt\n"
    with open(temp_root / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml_content)

    create_cvat_zip(str(temp_root), str(output_zip))
    shutil.rmtree(temp_root)


def adjust_labels_to_batch_prefixes(
    source_labels_dir: Path | None = None,
    output_labels_dir: Path | None = None,
    output_zip: Path | None = None,
    location: str | None = None,
) -> Dict[str, int]:
    data_dir = get_data_dir()
    source_labels_dir = source_labels_dir or (data_dir / "labeling" / "labels")
    output_labels_dir = output_labels_dir or (data_dir / "labeling" / "labels_with_batch_prefixes")
    output_zip = output_zip or (data_dir / "labeling" / "cvat_labels_upload.zip")
    images_dir = data_dir / "labeling" / "data_as_images"
    index_path = images_dir / "index.json"

    if not source_labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {source_labels_dir}")
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index file not found: {index_path}. Run initial_data_extraction.py first."
        )

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    resolved_location = location or infer_location(index)
    data_maps = get_data_maps(resolved_location)
    dcm_to_batch = build_dcm_name_to_batch_map(index, data_maps, images_dir)

    logger.info("Resolved location: %s", resolved_location)
    logger.info("Built batch mapping for %d DCM names.", len(dcm_to_batch))

    processed_frames, skipped_dcm_names = copy_labels_with_batch_prefixes(
        source_labels_dir, output_labels_dir, dcm_to_batch
    )

    if not processed_frames:
        raise RuntimeError("No label files were copied. Check batch mappings and source labels.")

    package_labels_for_cvat(output_labels_dir, processed_frames, output_zip)

    stats = {
        "copied_frames": len(processed_frames),
        "skipped_dcm_folders": len(skipped_dcm_names),
        "mapped_dcm_names": len(dcm_to_batch),
    }
    print(f"Prefixed labels copied to {output_labels_dir}")
    print(f"CVAT upload zip created at {output_zip}")
    print(
        f"Done: {stats['copied_frames']} frames copied, "
        f"{stats['skipped_dcm_folders']} DCM folders skipped."
    )
    if skipped_dcm_names:
        print("Skipped DCM folders:", ", ".join(skipped_dcm_names))

    return stats


def main() -> None:
    adjust_labels_to_batch_prefixes()


if __name__ == "__main__":
    main()
