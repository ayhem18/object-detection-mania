"""
Build object-detection training cache from PNG frames and YOLO labels.

Uses batch-aware relative keys from
:mod:`dl_lib.etalon_object_detection.modules.path_layout`::

    batch_{name}/dcm_stem/frame_{NNN}
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from dl_lib.common_tools.path_utils import get_data_dir
from dl_lib.etalon_object_detection.modules.path_layout import (
    RelativeKeyError,
    batch_name_from_relative_key,
    cached_label_path_from_relative_key,
    dcm_relative_key_from_frame_key,
    images_dir,
    is_frame_relative_key,
    labels_dir,
    load_index_maps,
    relative_key_from_local_path,
)

logger = logging.getLogger(__name__)

# ``batch_{name}/dcm_stem`` — DCM-level relative key from index.json
RelativeKeyFilter = Callable[[str], bool]

FULL_DATASET_VERSION = "full_dataset"
SMALL_DIAMETERS_VERSION = "small_diameteres"
SMALL_DIAMETERS_BATCH_SUBSTRING = "small_diameteres"


class DatasetVersion(NamedTuple):
    version_name: str
    relative_key_filter: Optional[RelativeKeyFilter]


def convert_bbox(
    cx: float,
    cy: float,
    x_dim_norm: float,
    y_dim_norm: float,
    img_y_dim: int,
    img_x_dim: int,
    target_format: str,
) -> List[float]:
    """Convert normalized YOLO coords to the requested format."""
    cx_abs = cx * img_x_dim
    cy_abs = cy * img_y_dim
    x_dim_abs = x_dim_norm * img_x_dim
    y_dim_abs = y_dim_norm * img_y_dim

    min_x = cx_abs - x_dim_abs / 2
    max_x = cx_abs + x_dim_abs / 2
    min_y = cy_abs - y_dim_abs / 2
    max_y = cy_abs + y_dim_abs / 2

    if target_format == "min_x_max_x_min_y_max_y":
        return [min_x, max_x, min_y, max_y]
    if target_format == "coco":
        return [min_x, min_y, x_dim_abs, y_dim_abs]
    if target_format == "pascal":
        return [min_x, min_y, max_x, max_y]
    if target_format == "yolo":
        return [cx, cy, x_dim_norm, y_dim_norm]
    raise ValueError(f"Unknown format: {target_format}")


def process_single_label_file(
    label_path: Path,
    img_y_dim: int,
    img_x_dim: int,
    target_format: str,
) -> Dict[str, Any]:
    """Read a YOLO ``.txt`` file and convert boxes to the target format."""
    boxes: List[List[float]] = []
    classes: List[int] = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            c_id = int(parts[0])
            cx, cy, x_dim_norm, y_dim_norm = map(float, parts[1:5])
            boxes.append(
                convert_bbox(
                    cx=cx,
                    cy=cy,
                    x_dim_norm=x_dim_norm,
                    y_dim_norm=y_dim_norm,
                    img_y_dim=img_y_dim,
                    img_x_dim=img_x_dim,
                    target_format=target_format,
                )
            )
            classes.append(c_id)

    return {
        "format": target_format,
        "image_x_dim": img_x_dim,
        "image_y_dim": img_y_dim,
        "boxes": boxes,
        "classes": classes,
    }


def _read_image_dims(img_path: Path) -> Optional[Tuple[int, int]]:
    """Return ``(y_dim, x_dim)`` using imdecode for Unicode paths on Windows."""
    try:
        with open(img_path, "rb") as f:
            chunk = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(chunk, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error("Failed to decode image: %s", img_path)
            return None
        img_y_dim, img_x_dim = img.shape[:2]
        return img_y_dim, img_x_dim
    except Exception:
        logger.error("Failed to read image %s", img_path, exc_info=True)
        return None


def log_unlinked_label_keys(
    label_lookup: Dict[str, Path],
    image_lookup: Dict[str, Path],
) -> int:
    """Log an error for each label frame key that has no matching PNG."""
    unlinked = sorted(set(label_lookup.keys()) - set(image_lookup.keys()))
    for frame_key in unlinked:
        logger.error(
            "Label has no matching image for relative key '%s' (%s)",
            frame_key,
            label_lookup[frame_key],
        )
    if unlinked:
        logger.error(
            "Labels without matching images: %d / %d",
            len(unlinked),
            len(label_lookup),
        )
    return len(unlinked)


def _iter_frame_assets(
    root: Path,
    *,
    extensions: set[str],
) -> Iterator[Tuple[str, Path]]:
    """Yield ``(frame_relative_key, absolute_path)`` under ``root``."""
    root = root.resolve()

    def _on_walk_error(exc: OSError) -> None:
        logger.warning("Skipping unreadable path during scan: %s", exc)

    for dirpath, _, filenames in os.walk(root, onerror=_on_walk_error):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() not in extensions:
                continue
            try:
                frame_key = relative_key_from_local_path(path, root)
            except RelativeKeyError:
                continue
            if is_frame_relative_key(frame_key):
                yield frame_key, path


def build_frame_image_lookup(
    images_root: Path,
    selected_dcm_keys: set[str],
) -> Dict[str, Path]:
    """Map frame relative keys to PNG paths, restricted to selected DCM keys."""
    lookup: Dict[str, Path] = {}
    for frame_key, path in _iter_frame_assets(images_root, extensions={".png"}):
        dcm_key = dcm_relative_key_from_frame_key(frame_key)
        if dcm_key not in selected_dcm_keys:
            continue
        lookup[frame_key] = path
    return lookup


def build_frame_label_lookup(
    labels_root: Path,
    selected_dcm_keys: set[str] | None = None,
) -> Dict[str, Path]:
    """Map frame relative keys to YOLO ``.txt`` paths."""
    lookup: Dict[str, Path] = {}
    for frame_key, path in _iter_frame_assets(labels_root, extensions={".txt"}):
        if selected_dcm_keys is not None:
            dcm_key = dcm_relative_key_from_frame_key(frame_key)
            if dcm_key not in selected_dcm_keys:
                continue
        lookup[frame_key] = path
    return lookup


def make_master_sample(
    frame_key: str,
    cache_dir: Path,
    rel_to_dcm: Dict[str, str],
) -> Dict[str, str]:
    dcm_key = dcm_relative_key_from_frame_key(frame_key)
    lbl_path = cached_label_path_from_relative_key(frame_key, cache_dir)
    return {
        "img_path": f"{frame_key}.png",
        "lbl_path": lbl_path.relative_to(cache_dir).as_posix(),
        "dcm_path": rel_to_dcm[dcm_key],
    }


def process_and_cache_labels(
    image_lookup: Dict[str, Path],
    label_lookup: Dict[str, Path],
    cache_dir: Path,
    rel_to_dcm: Dict[str, str],
    target_format: str,
    include_empty_frames: bool,
) -> List[Dict[str, str]]:
    """
    Convert YOLO labels to cached JSON and build master-index entries.

    When ``include_empty_frames`` is False, only labeled frames are cached.
    Otherwise every frame in ``image_lookup`` is cached (empty boxes if unlabeled).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    master_samples: List[Dict[str, str]] = []

    if include_empty_frames:
        frame_keys = sorted(image_lookup.keys())
    else:
        frame_keys = sorted(set(image_lookup.keys()) & set(label_lookup.keys()))

    desc = "Caching labels & background" if include_empty_frames else "Converting labels"
    for frame_key in tqdm(frame_keys, desc=desc):
        try:
            img_path = image_lookup.get(frame_key)
            if img_path is None:
                continue

            dims = _read_image_dims(img_path)
            if dims is None:
                continue
            img_y_dim, img_x_dim = dims

            label_path = label_lookup.get(frame_key)
            if label_path is not None and label_path.is_file():
                label_data = process_single_label_file(
                    label_path, img_y_dim, img_x_dim, target_format
                )
            elif include_empty_frames:
                label_data = {
                    "format": target_format,
                    "image_x_dim": img_x_dim,
                    "image_y_dim": img_y_dim,
                    "boxes": [],
                    "classes": [],
                }
            else:
                continue

            json_path = cached_label_path_from_relative_key(frame_key, cache_dir)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(label_data, f, indent=4, ensure_ascii=False)

            master_samples.append(make_master_sample(frame_key, cache_dir, rel_to_dcm))
        except Exception:
            logger.error("Failed to process frame '%s'", frame_key, exc_info=True)

    return master_samples


def build_master_index(
    master_samples: List[Dict[str, str]],
    cache_dir: Path,
    target_format: str,
) -> None:
    master_json_path = cache_dir / "master_labels.json"
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump({"samples": master_samples, "format": target_format}, f, indent=4, ensure_ascii=False)
    print(
        f"Preparation complete! Master index saved to {master_json_path} "
        f"with {len(master_samples)} samples."
    )


def get_dataset_hash(
    dcm_relative_keys: List[str],
    include_empty_frames: bool,
    version_name: str,
) -> str:
    hasher = hashlib.md5()
    for rel_key in sorted(dcm_relative_keys):
        hasher.update(rel_key.encode("utf-8"))
    hasher.update(str(include_empty_frames).encode("utf-8"))
    hasher.update(version_name.encode("utf-8"))
    return hasher.hexdigest()


def select_dcm_entries(
    rel_to_dcm: Dict[str, str],
    relative_key_filter: Optional[RelativeKeyFilter] = None,
) -> Tuple[List[str], set[str]]:
    """Return selected original DCM paths and their DCM-level relative keys."""
    if relative_key_filter is None:
        selected_keys = sorted(rel_to_dcm.keys())
    else:
        selected_keys = sorted(k for k in rel_to_dcm if relative_key_filter(k))
    selected_dcm_paths = [rel_to_dcm[k] for k in selected_keys]
    selected_dcm_keys = set(selected_keys)
    return selected_dcm_paths, selected_dcm_keys


def relative_key_filter_by_batch(batch_name: str) -> RelativeKeyFilter:
    """Keep DCMs whose relative key is under ``batch_name/``."""
    prefix = f"{batch_name}/"
    return lambda dcm_relative_key: dcm_relative_key.startswith(prefix)


def relative_key_filter_by_batch_substring(substring: str) -> RelativeKeyFilter:
    """Keep DCMs whose batch segment contains ``substring``."""
    return lambda dcm_relative_key: substring in batch_name_from_relative_key(dcm_relative_key)


def list_batch_names(rel_to_dcm: Dict[str, str]) -> List[str]:
    """Return sorted ``batch_{name}`` values present in the index."""
    return sorted({batch_name_from_relative_key(k) for k in rel_to_dcm})


def standard_dataset_versions(rel_to_dcm: Dict[str, str]) -> List[DatasetVersion]:
    """
    Built-in dataset versions:

    - ``full_dataset`` — all indexed DCMs
    - one version per batch (version name = batch folder name)
    - ``small_diameters`` — batches whose name contains ``small_diameters``
    """
    versions: List[DatasetVersion] = [
        DatasetVersion(FULL_DATASET_VERSION, None),
    ]

    # a version for each batch
    for batch_name in list_batch_names(rel_to_dcm):
        versions.append(
            DatasetVersion(batch_name, relative_key_filter_by_batch(batch_name))
        )

    # version with only small diameters
    versions.append(
        DatasetVersion(
            SMALL_DIAMETERS_VERSION,
            relative_key_filter_by_batch_substring(SMALL_DIAMETERS_BATCH_SUBSTRING),
        )
    )
    return versions


def build_all_standard_versions(
    include_empty_frames: bool = False,
    *,
    images_root: str | Path | None = None,
    labels_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    target_format: str = "pascal",
) -> Dict[str, str]:
    """Build every standard dataset version; returns ``version_name -> dataset_hash``."""
    rel_to_dcm, _ = load_index_maps()
    results: Dict[str, str] = {}
    for version in standard_dataset_versions(rel_to_dcm):
        print(f"\n=== Building dataset version: {version.version_name} ===")
        results[version.version_name] = get_labels(
            images_root=images_root,
            labels_root=labels_root,
            cache_root=cache_root,
            target_format=target_format,
            include_empty_frames=include_empty_frames,
            relative_key_filter=version.relative_key_filter,
            version_name=version.version_name,
        )
    return results


def get_labels(
    images_root: str | Path | None = None,
    labels_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    target_format: str = "pascal",
    include_empty_frames: bool = False,
    relative_key_filter: Optional[RelativeKeyFilter] = None,
    *,
    version_name: str,
) -> str:
    """
    Main orchestration for the object-detection dataset preparation pipeline.

    Parameters
    ----------
    relative_key_filter :
        ``Callable[[dcm_relative_key], bool]`` selecting which DCMs belong to
        this dataset version. ``dcm_relative_key`` has the form
        ``batch_{name}/dcm_stem`` (as in ``index.json`` values).
    version_name :
        Required human-readable dataset version label, stored in
        ``dataset_config.json`` and mixed into the hash.

    Returns
    -------
    str
        ``dataset_hash`` for the prepared cache directory.
    """
    if not version_name.strip():
        raise ValueError("version_name is required and must be non-empty.")

    images_root = Path(images_root or images_dir())
    labels_root = Path(labels_root or labels_dir())
    cache_root = Path(cache_root or (get_data_dir() / "labeling" / "cache"))

    rel_to_dcm, _ = load_index_maps()
    selected_dcm_paths, selected_dcm_keys = select_dcm_entries(rel_to_dcm, relative_key_filter)

    if not selected_dcm_keys:
        raise RuntimeError("Relative-key filter excluded every entry in index.json.")

    dataset_hash = get_dataset_hash(
        sorted(selected_dcm_keys), include_empty_frames, version_name
    )
    hashed_cache_dir = cache_root / dataset_hash
    labels_data_dir = hashed_cache_dir / "labels_data"
    labels_data_dir.mkdir(parents=True, exist_ok=True)
    hashed_cache_dir.mkdir(parents=True, exist_ok=True)

    config_path = hashed_cache_dir / "dataset_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_hash": dataset_hash,
                "version_name": version_name,
                "dcm_paths": selected_dcm_paths,
                "dcm_relative_keys": sorted(selected_dcm_keys),
                "include_empty_frames": include_empty_frames,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Dataset configuration saved. Hash: {dataset_hash}")
    print(f"Version: {version_name}")
    print(f"Selected {len(selected_dcm_paths)} DCM files.")

    print("1. Indexing PNG frames for selected DCMs...")
    image_lookup = build_frame_image_lookup(images_root, selected_dcm_keys)
    print(f"   Found {len(image_lookup)} PNG frames locally.")

    print("2. Indexing YOLO label files...")
    label_lookup = build_frame_label_lookup(labels_root, selected_dcm_keys)
    print(f"   Found {len(label_lookup)} label files.")
    log_unlinked_label_keys(label_lookup, image_lookup)

    mode = "INCLUDING empty frames" if include_empty_frames else "ONLY frames with objects"
    print(f"3. Parsing labels and converting to '{target_format}' format ({mode})...")
    master_samples = process_and_cache_labels(
        image_lookup=image_lookup,
        label_lookup=label_lookup,
        cache_dir=labels_data_dir,
        rel_to_dcm=rel_to_dcm,
        target_format=target_format,
        include_empty_frames=include_empty_frames,
    )

    print("4. Building master index...")
    build_master_index(master_samples, labels_data_dir, target_format)
    return dataset_hash


def main_server(include_empty_frames: bool = False, *, version_name: str) -> None:
    server_root = Path("/home/abouabid/seam_etalon_detector") / "data" / "labeling"
    get_labels(
        images_root=server_root / "data_as_images",
        labels_root=server_root / "labels",
        cache_root=server_root / "cache",
        target_format="pascal",
        include_empty_frames=include_empty_frames,
        version_name=version_name,
    )


def main_local(
    version_name: str,
    include_empty_frames: bool = False,
    relative_key_filter: Optional[RelativeKeyFilter] = None,
) -> str:
    return get_labels(
        target_format="pascal",
        include_empty_frames=include_empty_frames,
        relative_key_filter=relative_key_filter,
        version_name=version_name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s")
    hashes = build_all_standard_versions(include_empty_frames=False)
    print("\nBuilt dataset versions:")
    for name, dataset_hash in hashes.items():
        print(f"  {name}: {dataset_hash}")
