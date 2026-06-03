"""
Path helpers for the object-detection-mania monorepo.

Road-sign layout (single source of truth)::

    road_sign/data/
        original_ds/{dataset_hash}/
            dataset_config.json
            class_mapping.json
            images/{image_id}.png
            labels/{image_id}.txt
        resized_ds/{dataset_hash}/
            dataset_config.json
            images/ or train/images/
            labels/
            splits/{split_hash}/
                split_config.json
                train_files.json
                val_files.json
        patch_based/{dataset_hash}/
            ...

    road_sign/artifacts/{model_name}/anchor_configs/{config_hash}.json
        (registered clustering recipes: method + method_parameters)

    road_sign/artifacts/{model_name}/{dataset_type}/{dataset_hash}/{split_hash}/
        anchors/anchors.json          (YOLO KMeans)
        anchors/anchor_config.json    (RetinaNet per-split optimized anchors)
        {experiment_hash}/
            experiment_config.json
            checkpoints/
            metrics/
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Literal, Sequence, Tuple

RoadSignDatasetVersion = Literal["original", "resized", "patch_based"]

ROAD_SIGN_KAGGLE_SOURCE = "kaggle:traffic-sign-object-detection-challenge"
DATASET_VERSION_ORIGINAL: RoadSignDatasetVersion = "original"
DATASET_VERSION_RESIZED: RoadSignDatasetVersion = "resized"
DATASET_VERSION_PATCH: RoadSignDatasetVersion = "patch_based"

_VERSION_DIR_NAMES: Dict[RoadSignDatasetVersion, str] = {
    DATASET_VERSION_ORIGINAL: "original_ds",
    DATASET_VERSION_RESIZED: "resized_ds",
    DATASET_VERSION_PATCH: "patch_based",
}

DATASET_CONFIG_FILENAME = "dataset_config.json"
PATCH_CONFIG_FILENAME = "patch_config.yaml"
CLASS_MAPPING_FILENAME = "class_mapping.json"
ANCHORS_FILENAME = "anchors.json"
RETINANET_ANCHOR_CONFIG_FILENAME = "anchor_config.json"
SPLIT_CONFIG_FILENAME = "split_config.json"
TRAIN_FILES_FILENAME = "train_files.json"
VAL_FILES_FILENAME = "val_files.json"
EXPERIMENT_CONFIG_FILENAME = "experiment_config.json"

SPLITS_DIR_NAME = "splits"
LEGACY_SPLIT_METADATA_DIR_NAME = "split_metadata"

IMAGE_FILE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def get_project_root() -> Path:
    """
    Dynamically finds the home_made_od package root (directory containing ``src``).
    """
    current_dir = Path(__file__).resolve().parent
    while "src" not in os.listdir(current_dir):
        parent_dir = current_dir.parent
        if parent_dir == current_dir:
            raise RuntimeError("Could not find project root (directory containing 'src').")
        current_dir = parent_dir
    return current_dir


def get_monorepo_root() -> Path:
    """Return the object-detection-mania workspace root."""
    current = Path(__file__).resolve().parent
    for _ in range(12):
        if (current / "road_sign").is_dir() and (current / "home_made_od").is_dir():
            return current
        if current.parent == current:
            break
        current = current.parent
    raise RuntimeError(
        "Could not find monorepo root (expected directories 'road_sign' and 'home_made_od')."
    )


def road_sign_root() -> Path:
    return get_monorepo_root() / "road_sign"


def road_sign_data_root() -> Path:
    return road_sign_root() / "data"


def road_sign_artifacts_root() -> Path:
    return road_sign_root() / "artifacts"


REGISTERED_ANCHOR_CONFIGS_DIR_NAME = "anchor_configs"


def registered_anchor_configs_root(model_name: str = "retinanet") -> Path:
    """
    Directory for registered anchor-clustering recipes (dataset-independent).

    Each recipe is ``{config_hash}.json`` (method + validated method_parameters).
    """
    return road_sign_artifacts_root() / model_name / REGISTERED_ANCHOR_CONFIGS_DIR_NAME


def registered_anchor_config_path(
    config_hash: str,
    model_name: str = "retinanet",
) -> Path:
    """Path to one registered recipe: ``artifacts/{model}/anchor_configs/{hash}.json``."""
    return registered_anchor_configs_root(model_name) / f"{config_hash}.json"


def legacy_flat_data_dir() -> Path:
    """Pre-refactor flat dataset location (``road_sign/data/``)."""
    return road_sign_data_root()


def dataset_version_root(version: RoadSignDatasetVersion) -> Path:
    return road_sign_data_root() / _VERSION_DIR_NAMES[version]


def dataset_root(version: RoadSignDatasetVersion, dataset_hash: str) -> Path:
    return dataset_version_root(version) / dataset_hash


def dataset_config_path(version: RoadSignDatasetVersion, dataset_hash: str) -> Path:
    return dataset_root(version, dataset_hash) / DATASET_CONFIG_FILENAME


def patch_config_path(dataset_hash: str) -> Path:
    return dataset_root(DATASET_VERSION_PATCH, dataset_hash) / PATCH_CONFIG_FILENAME


def class_mapping_path(version: RoadSignDatasetVersion, dataset_hash: str) -> Path:
    return dataset_root(version, dataset_hash) / CLASS_MAPPING_FILENAME


def dataset_type_dir_name(version: RoadSignDatasetVersion) -> str:
    """Directory name under ``road_sign/data/`` and ``artifacts/{model}/``."""
    return _VERSION_DIR_NAMES[version]


def dataset_images_dir(version: RoadSignDatasetVersion, dataset_hash: str) -> Path:
    """Training images directory (flat ``images/`` for original, else ``train/images/``)."""
    root = dataset_root(version, dataset_hash)
    if version == DATASET_VERSION_ORIGINAL:
        return root / "images"
    return root / "train" / "images"


def dataset_labels_dir(version: RoadSignDatasetVersion, dataset_hash: str) -> Path:
    """Training labels directory (flat ``labels/`` for original, else nested train labels)."""
    root = dataset_root(version, dataset_hash)
    if version == DATASET_VERSION_ORIGINAL:
        return root / "labels"
    canonical = root / "labels" / "train"
    if canonical.is_dir():
        return canonical
    legacy = root / "labels" / "annotations"
    return legacy if legacy.is_dir() else canonical


def train_images_dir(version: RoadSignDatasetVersion, dataset_hash: str) -> Path:
    return dataset_images_dir(version, dataset_hash)


def train_labels_dir(version: RoadSignDatasetVersion, dataset_hash: str) -> Path:
    return dataset_labels_dir(version, dataset_hash)


def test_images_dir(version: RoadSignDatasetVersion, dataset_hash: str) -> Path:
    return dataset_root(version, dataset_hash) / "test" / "images"


def splits_root(version: RoadSignDatasetVersion, dataset_hash: str) -> Path:
    return dataset_root(version, dataset_hash) / SPLITS_DIR_NAME


def split_dir(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
) -> Path:
    return splits_root(version, dataset_hash) / split_hash


def split_config_path(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
) -> Path:
    return split_dir(version, dataset_hash, split_hash) / SPLIT_CONFIG_FILENAME


def model_artifact_root(
    model_name: str,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
) -> Path:
    return (
        road_sign_artifacts_root()
        / model_name
        / dataset_type_dir_name(version)
        / dataset_hash
        / split_hash
    )


def model_anchors_dir(
    model_name: str,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
) -> Path:
    return model_artifact_root(model_name, version, dataset_hash, split_hash) / "anchors"


def model_anchors_json_path(
    model_name: str,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
) -> Path:
    """YOLO KMeans anchor file (``anchors.json``)."""
    return model_anchors_dir(model_name, version, dataset_hash, split_hash) / ANCHORS_FILENAME


def model_retinanet_anchor_config_path(
    model_name: str,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
) -> Path:
    """RetinaNet per-FPN-level anchor config (``anchor_config.json``)."""
    return (
        model_anchors_dir(model_name, version, dataset_hash, split_hash)
        / RETINANET_ANCHOR_CONFIG_FILENAME
    )


def model_experiment_dir(
    model_name: str,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    experiment_hash: str,
) -> Path:
    return model_artifact_root(model_name, version, dataset_hash, split_hash) / experiment_hash


def model_checkpoints_dir(
    model_name: str,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    experiment_hash: str,
) -> Path:
    return model_experiment_dir(
        model_name, version, dataset_hash, split_hash, experiment_hash
    ) / "checkpoints"


def model_metrics_dir(
    model_name: str,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    experiment_hash: str,
) -> Path:
    return model_experiment_dir(
        model_name, version, dataset_hash, split_hash, experiment_hash
    ) / "metrics"


def patch_visualization_dir(name: str = "patch_visualization") -> Path:
    path = road_sign_artifacts_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def original_data_sanity_dir(name: str = "org_data_sa") -> Path:
    path = road_sign_artifacts_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def canonical_json_hash(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def compute_original_dataset_hash(
    *,
    source: str,
    class_mapping: Dict[str, Any],
    image_stems: Sequence[str],
) -> str:
    payload = {
        "version": DATASET_VERSION_ORIGINAL,
        "source": source,
        "class_mapping": class_mapping,
        "image_stems": sorted(image_stems),
    }
    return canonical_json_hash(payload)


def compute_resized_dataset_hash(
    source_dataset_hash: str,
    target_size: Tuple[int, int],
) -> str:
    payload = {
        "version": DATASET_VERSION_RESIZED,
        "source_dataset_hash": source_dataset_hash,
        "target_size": [int(target_size[0]), int(target_size[1])],
    }
    return canonical_json_hash(payload)


def compute_patch_dataset_hash(
    source_dataset_hash: str,
    patch_config: Dict[str, Any],
) -> str:
    payload = {
        "version": DATASET_VERSION_PATCH,
        "source_dataset_hash": source_dataset_hash,
        "patch_config": patch_config,
    }
    return canonical_json_hash(payload)


def compute_split_hash(train_ids: Sequence[str], val_ids: Sequence[str]) -> str:
    hash_str = (
        "train__"
        + "|".join(sorted(train_ids))
        + "__val__"
        + "|".join(sorted(val_ids))
    )
    return hashlib.md5(hash_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Dataset discovery / config I/O
# ---------------------------------------------------------------------------


def write_dataset_config(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    config: Dict[str, Any],
) -> Path:
    path = dataset_config_path(version, dataset_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    return path


def load_dataset_config(version: RoadSignDatasetVersion, dataset_hash: str) -> Dict[str, Any]:
    path = dataset_config_path(version, dataset_hash)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_latest_dataset_hash(version: RoadSignDatasetVersion) -> str | None:
    root = dataset_version_root(version)
    if not root.is_dir():
        return None
    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir() and len(child.name) == 32 and (child / DATASET_CONFIG_FILENAME).is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).name


def resolve_split_dir(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
) -> Path | None:
    """Return an existing split directory (``splits/`` or legacy ``split_metadata/``)."""
    for candidate in (
        split_dir(version, dataset_hash, split_hash),
        dataset_root(version, dataset_hash)
        / LEGACY_SPLIT_METADATA_DIR_NAME
        / split_hash,
    ):
        if (candidate / TRAIN_FILES_FILENAME).is_file():
            return candidate
    return None


def resolve_latest_split_hash(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
) -> str | None:
    root = splits_root(version, dataset_hash)
    if not root.is_dir():
        legacy_root = dataset_root(version, dataset_hash) / LEGACY_SPLIT_METADATA_DIR_NAME
        root = legacy_root if legacy_root.is_dir() else root
    if not root.is_dir():
        return None
    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir()
        and len(child.name) == 32
        and (child / TRAIN_FILES_FILENAME).is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).name


def load_split_lists(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
) -> Tuple[list[str], list[str], Dict[str, Any]]:
    split_path = resolve_split_dir(version, dataset_hash, split_hash)
    if split_path is None:
        raise FileNotFoundError(
            f"Split not found for {version}/{dataset_hash}/{split_hash}"
        )
    with open(split_path / TRAIN_FILES_FILENAME, encoding="utf-8") as f:
        train_ids = json.load(f)
    with open(split_path / VAL_FILES_FILENAME, encoding="utf-8") as f:
        val_ids = json.load(f)
    config: Dict[str, Any] = {}
    config_file = split_path / SPLIT_CONFIG_FILENAME
    if config_file.is_file():
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)
    return train_ids, val_ids, config


# ---------------------------------------------------------------------------
# Samples, pairs, splits
# ---------------------------------------------------------------------------


def _image_stems_in_dir(images_dir: Path) -> list[str]:
    if not images_dir.is_dir():
        return []
    stems: list[str] = []
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_FILE_SUFFIXES:
            stems.append(path.stem)
    return sorted(stems)


def list_split_unit_ids(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
) -> list[str]:
    """
    IDs used for train/val splitting (leakage-safe grouping key).

    Flat datasets: image stem. Patch datasets: parent image stem (subdir name).
    """
    img_dir = dataset_images_dir(version, dataset_hash)
    if version == DATASET_VERSION_PATCH:
        if not img_dir.is_dir():
            return []
        return sorted(d.name for d in img_dir.iterdir() if d.is_dir())
    return _image_stems_in_dir(img_dir)


def collect_original_sample_manifest(data_dir: Path) -> Tuple[list[str], Dict[str, Any] | None]:
    flat_images = data_dir / "images"
    if flat_images.is_dir():
        image_stems = _image_stems_in_dir(flat_images)
    else:
        image_stems = _image_stems_in_dir(data_dir / "train" / "images")

    class_mapping: Dict[str, Any] | None = None
    mapping_path = data_dir / CLASS_MAPPING_FILENAME
    if mapping_path.is_file():
        with open(mapping_path, encoding="utf-8") as f:
            class_mapping = json.load(f)

    return image_stems, class_mapping


def build_original_dataset_config(
    *,
    source: str,
    class_mapping: Dict[str, Any],
    image_stems: Sequence[str],
) -> Tuple[str, Dict[str, Any]]:
    dataset_hash = compute_original_dataset_hash(
        source=source,
        class_mapping=class_mapping,
        image_stems=image_stems,
    )
    config = {
        "version": DATASET_VERSION_ORIGINAL,
        "dataset_hash": dataset_hash,
        "source": source,
        "image_count": len(image_stems),
        "class_mapping": class_mapping,
    }
    return dataset_hash, config


def register_original_dataset_from_dir(
    data_dir: Path,
    *,
    source: str = ROAD_SIGN_KAGGLE_SOURCE,
    copy: bool = False,
) -> str:
    """
    Compute the original-dataset hash from on-disk content, ensure files live under
    ``original_ds/{dataset_hash}/``, and write ``dataset_config.json``.

    When ``copy`` is True (legacy import), the source directory is left unchanged.
    """
    import shutil

    image_stems, class_mapping = collect_original_sample_manifest(data_dir)
    if not image_stems:
        raise ValueError(
            f"No images found under {data_dir / 'images'} "
            f"(or legacy {data_dir / 'train' / 'images'})."
        )

    dataset_hash, config = build_original_dataset_config(
        source=source,
        class_mapping=class_mapping or {},
        image_stems=image_stems,
    )

    target_root = dataset_root(DATASET_VERSION_ORIGINAL, dataset_hash)
    if data_dir.resolve() != target_root.resolve():
        target_root.parent.mkdir(parents=True, exist_ok=True)
        if target_root.exists():
            shutil.rmtree(target_root)
        if copy:
            shutil.copytree(data_dir, target_root)
        else:
            shutil.move(str(data_dir), str(target_root))

    write_dataset_config(DATASET_VERSION_ORIGINAL, dataset_hash, config)
    return dataset_hash


def iter_image_label_pairs(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    unit_ids: Sequence[str] | None = None,
) -> Iterator[Tuple[Path, Path]]:
    """
    Yield ``(image_path, label_path)`` for all samples, optionally filtered by split unit IDs.
    """
    img_dir = dataset_images_dir(version, dataset_hash)
    lbl_dir = dataset_labels_dir(version, dataset_hash)
    allowed = set(unit_ids) if unit_ids is not None else None

    if not img_dir.is_dir():
        return

    if version == DATASET_VERSION_PATCH:
        subdirs = sorted(img_dir.iterdir()) if allowed is None else [
            img_dir / uid for uid in sorted(allowed) if (img_dir / uid).is_dir()
        ]
        for subdir in subdirs:
            if not subdir.is_dir():
                continue
            lbl_subdir = lbl_dir / subdir.name
            if not lbl_subdir.is_dir():
                continue
            for img_path in sorted(subdir.iterdir()):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in IMAGE_FILE_SUFFIXES:
                    continue
                label_path = lbl_subdir / f"{img_path.stem}.txt"
                if label_path.is_file():
                    yield img_path, label_path
        return

    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_FILE_SUFFIXES:
            continue
        if allowed is not None and img_path.stem not in allowed:
            continue
        label_path = lbl_dir / f"{img_path.stem}.txt"
        if label_path.is_file():
            yield img_path, label_path


def iter_train_image_label_pairs(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
) -> Iterator[Tuple[Path, Path]]:
    """Yield all training pairs (no split filter)."""
    yield from iter_image_label_pairs(version, dataset_hash)


def get_train_image_label_pairs(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    unit_ids: Sequence[str] | None = None,
) -> list[Tuple[str, str]]:
    return [
        (str(img), str(lbl))
        for img, lbl in iter_image_label_pairs(version, dataset_hash, unit_ids)
    ]


def create_and_cache_split(
    *,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    sample_ids: Sequence[str] | None = None,
    val_ratio: float = 0.15,
    seed: int = 42,
    train_ratio: float | None = None,
) -> Tuple[list[str], list[str], str]:
    """
    Random train/val split over split-unit IDs (image stems or patch parent stems).

    Writes ``splits/{split_hash}/`` under the dataset directory.
    Returns ``(train_ids, val_ids, split_hash)``.
    """
    if train_ratio is not None:
        val_ratio = 1.0 - train_ratio

    unique_ids = sorted(set(sample_ids or list_split_unit_ids(version, dataset_hash)))
    if not unique_ids:
        raise ValueError("Cannot split an empty sample list.")

    shuffled = list(unique_ids)
    random.seed(seed)
    random.shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_ids = sorted(shuffled[:val_size])
    train_ids = sorted(shuffled[val_size:]) if val_size else shuffled

    split_hash = compute_split_hash(train_ids, val_ids)
    out_dir = split_dir(version, dataset_hash, split_hash)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_config = {
        "split_hash": split_hash,
        "dataset_version": version,
        "dataset_hash": dataset_hash,
        "val_ratio": val_ratio,
        "train_ratio": 1.0 - val_ratio,
        "seed": seed,
        "train_count": len(train_ids),
        "val_count": len(val_ids),
    }
    with open(out_dir / SPLIT_CONFIG_FILENAME, "w", encoding="utf-8") as f:
        json.dump(split_config, f, indent=4, ensure_ascii=False)
    with open(out_dir / TRAIN_FILES_FILENAME, "w", encoding="utf-8") as f:
        json.dump(train_ids, f, indent=4, ensure_ascii=False)
    with open(out_dir / VAL_FILES_FILENAME, "w", encoding="utf-8") as f:
        json.dump(val_ids, f, indent=4, ensure_ascii=False)

    print(
        f"Split cached at {out_dir} | hash={split_hash} | "
        f"train={len(train_ids)} val={len(val_ids)}"
    )
    return train_ids, val_ids, split_hash


def split_unit_filters(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
) -> Tuple[Callable[[str], bool], Callable[[str], bool]]:
    train_set = set(train_ids)
    val_set = set(val_ids)
    return lambda uid: uid in train_set, lambda uid: uid in val_set


def collect_normalized_wh_from_pairs(
    pairs: Sequence[Tuple[str, str]],
) -> list[Tuple[float, float]]:
    wh_list: list[Tuple[float, float]] = []
    for _, label_path in pairs:
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    wh_list.append((float(parts[3]), float(parts[4])))
    return wh_list


def compute_and_cache_anchors(
    model_name: str,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    *,
    num_anchors: int = 5,
    seed: int = 42,
    force: bool = False,
) -> list[list[float]]:
    """
    KMeans-IoU anchors from the training split labels, saved under
    ``artifacts/{model}/{dataset_type}/{dataset_hash}/{split_hash}/anchors/``.
    """
    from home_made_od.yolo_family.anchors.anchor_utils import generate_anchors

    out_path = model_anchors_json_path(model_name, version, dataset_hash, split_hash)
    if out_path.is_file() and not force:
        with open(out_path, encoding="utf-8") as f:
            payload = json.load(f)
        print(f"Loading cached anchors from {out_path}")
        return payload["anchors"]

    train_ids, _, _ = load_split_lists(version, dataset_hash, split_hash)
    train_pairs = get_train_image_label_pairs(version, dataset_hash, train_ids)
    wh_list = collect_normalized_wh_from_pairs(train_pairs)
    if not wh_list:
        raise ValueError(
            f"No boxes found on training split {split_hash} for anchor generation."
        )

    anchors = generate_anchors(wh_list, num_anchors=num_anchors, seed=seed)
    anchors_list = anchors.tolist()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchors": anchors_list,
        "model_name": model_name,
        "dataset_version": version,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "num_anchors": num_anchors,
        "seed": seed,
        "train_unit_count": len(train_ids),
        "box_count": len(wh_list),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    print(f"Saved anchors to {out_path}")
    return anchors_list


def load_anchors(
    model_name: str,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
) -> list[list[float]]:
    path = model_anchors_json_path(model_name, version, dataset_hash, split_hash)
    with open(path, encoding="utf-8") as f:
        return json.load(f)["anchors"]


def write_json_file(path: Path, payload: Dict[str, Any], *, indent: int = 4) -> Path:
    """Write a JSON file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False)
    return path


def read_json_file(path: Path) -> Dict[str, Any]:
    """Load a JSON object from *path*."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def compute_experiment_hash(config: Dict[str, Any]) -> str:
    """MD5 of experiment-defining parameters (for run artifact subfolders)."""
    return canonical_json_hash(config)


def write_experiment_config(
    model_name: str,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    experiment_hash: str,
    config: Dict[str, Any],
) -> Path:
    run_dir = model_experiment_dir(
        model_name, version, dataset_hash, split_hash, experiment_hash
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / EXPERIMENT_CONFIG_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Legacy YOLOv2 synthetic toy paths (unchanged)
# ---------------------------------------------------------------------------


def get_yolo_v2_artifacts_dir() -> Path:
    """Returns the path to the YOLOv2 synthetic artifacts directory."""
    root = get_project_root()
    path = root / "src" / "object_detection_mania" / "artifacts" / "yolo_v2_artifacts" / "synthetic"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_yolo_v2_config_dir() -> Path:
    """Returns the path to the YOLOv2 synthetic configs directory."""
    path = get_yolo_v2_artifacts_dir() / "configs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_yolo_v2_visualization_dir(subfolder: str) -> Path:
    """Returns a visualization subfolder path within YOLOv2 artifacts."""
    path = get_yolo_v2_artifacts_dir() / "visualizations" / subfolder
    path.mkdir(parents=True, exist_ok=True)
    return path
