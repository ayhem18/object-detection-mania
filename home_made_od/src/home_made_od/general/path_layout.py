"""
Canonical paths for the etalon object-detection labeling workflow.

Cache layout (per dataset)::

    labeling/cache/{dataset_hash}/
        dataset_config.json
        labels_data/
            master_labels.json
            ...
        split_metadata/
            {split_hash}/
                train_files.json
                val_files.json

Artifact layout::

    labeling/artifacts/{dataset_hash}/...
    labeling/artifacts/runs/{dataset_hash}/{split_hash}/{experiment_hash}/...

Relative keys (strict format — enforced by helpers)::

    DCM level:   batch_{name}/dcm_stem
    Frame level: batch_{name}/dcm_stem/frame_{number}

    Examples::

        batch_large_diameters/01_TR_SFCC_ZEPP_DRT_08981
        batch_large_diameters/01_TR_SFCC_ZEPP_DRT_08981/frame_001

    ``index.json`` maps original DCM path <-> DCM-level relative key.
    Frame PNGs:  ``data_as_images/{frame_relative_key}.png``
    YOLO labels: ``labels/{frame_relative_key}.txt``
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterator

from dl_lib.common_tools.path_utils import get_data_dir

SPLIT_ARTIFACT_RESERVED_DIRS = frozenset({"full_dataset"})


def labeling_root() -> Path:
    return get_data_dir() / "labeling"


def anchor_configs_root() -> Path:
    """Dataset-independent anchor recipes (method + method_parameters)."""
    return labeling_root() / "anchor_configs"


def registered_anchor_config_path(config_hash: str) -> Path:
    return anchor_configs_root() / f"{config_hash}.json"


def dataset_cache_root(dataset_hash: str) -> Path:
    return labeling_root() / "cache" / dataset_hash


def labels_data_dir(dataset_hash: str) -> Path:
    """
    Directory with ``master_labels.json`` and per-frame label JSON files.

    Falls back to the legacy cache root when ``labels_data/`` is absent.
    """
    root = dataset_cache_root(dataset_hash)
    nested = root / "labels_data"
    if nested.is_dir() and (nested / "master_labels.json").is_file():
        return nested
    if (root / "master_labels.json").is_file():
        return root
    return nested


def master_labels_path(dataset_hash: str) -> Path:
    return labels_data_dir(dataset_hash) / "master_labels.json"


def dataset_hash_from_master_labels(master_json_path: str | Path) -> str:
    path = Path(master_json_path)
    if path.parent.name == "labels_data":
        return path.parent.parent.name
    return path.parent.name


def split_metadata_root(dataset_hash: str) -> Path:
    return dataset_cache_root(dataset_hash) / "split_metadata"


def split_metadata_dir(dataset_hash: str, split_hash: str) -> Path:
    return split_metadata_root(dataset_hash) / split_hash


def legacy_split_dir(dataset_hash: str, split_hash: str) -> Path:
    """Pre-refactor location: ``cache/{dataset_hash}/splits/{split_hash}/``."""
    return dataset_cache_root(dataset_hash) / "splits" / split_hash


def resolve_split_metadata_dir(dataset_hash: str, split_hash: str) -> Path | None:
    """Return an existing split-metadata directory (new or legacy), if any."""
    new_dir = split_metadata_dir(dataset_hash, split_hash)
    if (new_dir / "train_files.json").is_file():
        return new_dir
    old_dir = legacy_split_dir(dataset_hash, split_hash)
    if (old_dir / "train_files.json").is_file():
        return old_dir
    return None


def artifacts_root() -> Path:
    return labeling_root() / "artifacts"


def dataset_artifacts_root(dataset_hash: str) -> Path:
    return artifacts_root() / dataset_hash


def full_dataset_artifacts_root(dataset_hash: str) -> Path:
    return dataset_artifacts_root(dataset_hash) / "full_dataset"


def split_artifacts_root(dataset_hash: str, split_hash: str) -> Path:
    return dataset_artifacts_root(dataset_hash) / split_hash


def anchor_artifacts_root(
    dataset_hash: str,
    split_hash: str,
    anchor_config_hash: str,
) -> Path:
    """Root for all phases tied to one (dataset, split, anchor config) triple."""
    return split_artifacts_root(dataset_hash, split_hash) / anchor_config_hash


def anchors_data_dir(
    dataset_hash: str,
    split_hash: str,
    anchor_config_hash: str,
) -> Path:
    return anchor_artifacts_root(dataset_hash, split_hash, anchor_config_hash) / "anchors_data"


def anchors_data_manifest_path(
    dataset_hash: str,
    split_hash: str,
    anchor_config_hash: str,
) -> Path:
    return anchors_data_dir(dataset_hash, split_hash, anchor_config_hash) / ANCHOR_MANIFEST_FILENAME


ANCHOR_MANIFEST_FILENAME = "anchor_config.json"


def sweeps_capacity_subset_dir(
    dataset_hash: str,
    split_hash: str,
    anchor_config_hash: str,
    subset_hash: str,
) -> Path:
    return (
        anchor_artifacts_root(dataset_hash, split_hash, anchor_config_hash)
        / "sweeps"
        / "capacity"
        / subset_hash
    )


def sweeps_capacity_run_dir(
    dataset_hash: str,
    split_hash: str,
    anchor_config_hash: str,
    subset_hash: str,
    freeze_layers: int,
) -> Path:
    return (
        sweeps_capacity_subset_dir(
            dataset_hash, split_hash, anchor_config_hash, subset_hash
        )
        / f"frozen_layers_{freeze_layers}"
    )


def sweeps_lr_find_dir(
    dataset_hash: str,
    split_hash: str,
    anchor_config_hash: str,
) -> Path:
    return (
        anchor_artifacts_root(dataset_hash, split_hash, anchor_config_hash)
        / "sweeps"
        / "lr_find"
    )


def sweeps_lr_find_run_dir(
    dataset_hash: str,
    split_hash: str,
    anchor_config_hash: str,
    freeze_layers: int,
) -> Path:
    return sweeps_lr_find_dir(dataset_hash, split_hash, anchor_config_hash) / (
        f"frozen_layers_{freeze_layers}"
    )


def full_training_run_dir(
    dataset_hash: str,
    split_hash: str,
    anchor_config_hash: str,
    experiment_hash: str,
) -> Path:
    return (
        anchor_artifacts_root(dataset_hash, split_hash, anchor_config_hash)
        / "full_training"
        / experiment_hash
    )


def error_analysis_dir(
    dataset_hash: str,
    split_hash: str,
    anchor_config_hash: str,
    experiment_hash: str,
    ea_hash: str | None = None,
) -> Path:
    base = (
        anchor_artifacts_root(dataset_hash, split_hash, anchor_config_hash)
        / "error_analysis"
        / experiment_hash
    )
    if ea_hash is not None:
        return base / ea_hash
    return base


def anchor_analysis_dir(
    dataset_hash: str,
    split_hash: str,
    anchor_config_hash: str,
    name: str,
) -> Path:
    return anchor_artifacts_root(dataset_hash, split_hash, anchor_config_hash) / "analysis" / name


def split_analysis_dir(
    dataset_hash: str,
    split_hash: str,
    anchor_config_hash: str,
    name: str,
) -> Path:
    """Alias for :func:`anchor_analysis_dir` (split-scoped analysis under an anchor config)."""
    return anchor_analysis_dir(dataset_hash, split_hash, anchor_config_hash, name)


def full_dataset_analysis_dir(dataset_hash: str, name: str) -> Path:
    return full_dataset_artifacts_root(dataset_hash) / "analysis" / name


def images_dir() -> Path:
    return labeling_root() / "data_as_images"


def labels_dir() -> Path:
    return labeling_root() / "labels"


INDEX_FILENAME = "index.json"
CVAT_DATA_AS_IMAGES_PREFIX = "digital_lab_wired_etalons/data_as_images"

# Strict relative-key grammar (enforced by require_* helpers).
BATCH_DIR_PATTERN = re.compile(r"^batch_[^/]+$")
FRAME_NAME_PATTERN = re.compile(r"^frame_\d{3}$")
DCM_RELATIVE_KEY_PATTERN = re.compile(r"^batch_[^/]+/[^/]+$")
FRAME_RELATIVE_KEY_PATTERN = re.compile(r"^batch_[^/]+/[^/]+/frame_\d{3}$")


class RelativeKeyError(ValueError):
    """Raised when a path/key does not match the batch/DCM/frame relative-key format."""


def index_path() -> Path:
    return images_dir() / INDEX_FILENAME


def normalize_relative_key(key: str | Path) -> str:
    """Normalize slashes; does not validate format."""
    return Path(str(key).replace("\\", "/")).as_posix().strip("/")


def is_batch_dir_name(name: str) -> bool:
    return bool(BATCH_DIR_PATTERN.match(name))


def is_frame_name(name: str) -> bool:
    return bool(FRAME_NAME_PATTERN.match(name))


def is_dcm_relative_key(key: str | Path) -> bool:
    return bool(DCM_RELATIVE_KEY_PATTERN.match(normalize_relative_key(key)))


def is_frame_relative_key(key: str | Path) -> bool:
    return bool(FRAME_RELATIVE_KEY_PATTERN.match(normalize_relative_key(key)))


def require_dcm_relative_key(key: str | Path) -> str:
    """
    Validate and return a DCM-level key: ``batch_{name}/dcm_stem``.
    """
    normalized = normalize_relative_key(key)
    if not is_dcm_relative_key(normalized):
        raise RelativeKeyError(
            f"Invalid DCM relative key '{key}'. "
            "Expected format: batch_{name}/dcm_stem "
            f"(e.g. batch_large_diameters/01_TR_SFCC_ZEPP_DRT_08981)."
        )
    return normalized


def require_frame_relative_key(key: str | Path) -> str:
    """
    Validate and return a frame-level key: ``batch_{name}/dcm_stem/frame_{number}``.
    """
    normalized = normalize_relative_key(key)
    if not is_frame_relative_key(normalized):
        raise RelativeKeyError(
            f"Invalid frame relative key '{key}'. "
            "Expected format: batch_{name}/dcm_stem/frame_{number} "
            f"(e.g. batch_large_diameters/01_TR_SFCC_ZEPP_DRT_08981/frame_001)."
        )
    return normalized


def make_dcm_relative_key(batch: str, dcm_stem: str) -> str:
    """Build and validate ``batch_{name}/dcm_stem``."""
    return require_dcm_relative_key(f"{batch}/{dcm_stem}")


def make_frame_relative_key(batch: str, dcm_stem: str, frame_name: str) -> str:
    """Build and validate ``batch_{name}/dcm_stem/frame_{number}``."""
    return require_frame_relative_key(f"{batch}/{dcm_stem}/{frame_name}")


def _frame_stem_from_filename(filename: str) -> str | None:
    """Return ``frame_{number}`` from a frame asset filename, or ``None``."""
    stem = Path(filename).stem
    if stem.endswith("_lbl"):
        stem = stem[: -len("_lbl")]
    if is_frame_name(stem):
        return stem
    return None


def relative_key_from_local_path(local_path: str | Path, root: str | Path) -> str:
    """
    Convert a path under ``root`` to a validated relative key.

    Frame assets (``.png`` / ``.txt`` / ``frame_*_lbl.json``) yield frame-level keys.
    Two-segment paths yield DCM-level keys.
    """
    rel = Path(local_path).resolve().relative_to(Path(root).resolve()).as_posix()
    norm = normalize_relative_key(rel)
    parts = norm.split("/")

    if len(parts) >= 3:
        batch, dcm_stem, leaf = parts[0], parts[1], parts[-1]
        frame_stem = _frame_stem_from_filename(leaf)
        if frame_stem is None:
            raise RelativeKeyError(
                f"Cannot derive frame relative key from '{local_path}'. "
                f"Leaf '{leaf}' is not frame_{{number}} (optionally with _lbl suffix)."
            )
        return make_frame_relative_key(batch, dcm_stem, frame_stem)

    if len(parts) == 2:
        return require_dcm_relative_key(norm)

    raise RelativeKeyError(
        f"Cannot derive relative key from '{local_path}' under '{root}'. "
        "Expected batch_{{name}}/dcm_stem or batch_{{name}}/dcm_stem/frame_{{number}}."
    )


def local_path_from_relative_key(
    relative_key: str | Path,
    root: str | Path,
    *,
    extension: str | None = None,
) -> Path:
    """Resolve a validated frame relative key under ``root``."""
    key = require_frame_relative_key(relative_key)
    if extension is not None:
        ext = extension if extension.startswith(".") else f".{extension}"
        return Path(root) / f"{key}{ext}"
    return Path(root) / key


def image_path_from_relative_key(
    relative_key: str | Path,
    images_root: str | Path | None = None,
) -> Path:
    return local_path_from_relative_key(relative_key, images_root or images_dir(), extension=".png")


def label_path_from_relative_key(
    relative_key: str | Path,
    labels_root: str | Path | None = None,
) -> Path:
    return local_path_from_relative_key(relative_key, labels_root or labels_dir(), extension=".txt")


def cached_label_path_from_relative_key(
    frame_relative_key: str | Path,
    cache_root: str | Path,
) -> Path:
    """``batch/.../frame_001`` -> ``cache_root/batch/.../frame_001_lbl.json``."""
    key = require_frame_relative_key(frame_relative_key)
    frame_name = frame_name_from_relative_key(key)
    dcm_key = dcm_relative_key_from_frame_key(key)
    return Path(cache_root) / dcm_key / f"{frame_name}_lbl.json"


def dcm_relative_key_from_frame_key(frame_relative_key: str | Path) -> str:
    """``batch_{name}/dcm_stem/frame_{number}`` -> ``batch_{name}/dcm_stem``."""
    key = require_frame_relative_key(frame_relative_key)
    return require_dcm_relative_key(Path(key).parent)


def frame_name_from_relative_key(frame_relative_key: str | Path) -> str:
    """``batch_{name}/dcm_stem/frame_{number}`` -> ``frame_{number}``."""
    key = require_frame_relative_key(frame_relative_key)
    frame_name = Path(key).name
    if not is_frame_name(frame_name):
        raise RelativeKeyError(f"Invalid frame name in key '{key}'.")
    return frame_name


def batch_name_from_relative_key(key: str | Path) -> str:
    """Return the ``batch_{name}`` segment from a DCM- or frame-level key."""
    normalized = require_dcm_relative_key(
        key if is_dcm_relative_key(key) else dcm_relative_key_from_frame_key(key)
    )
    return normalized.split("/", 1)[0]


def dcm_stem_from_relative_key(key: str | Path) -> str:
    """Return the ``dcm_stem`` segment from a DCM- or frame-level key."""
    normalized = require_dcm_relative_key(
        key if is_dcm_relative_key(key) else dcm_relative_key_from_frame_key(key)
    )
    return normalized.split("/", 1)[1]


def load_index(path: str | Path | None = None) -> dict[str, str]:
    """Load ``data_as_images/index.json`` (original DCM path -> ``batch/dcm_stem``)."""
    index_file = Path(path) if path is not None else index_path()
    if not index_file.is_file():
        raise FileNotFoundError(f"Index file not found: {index_file}")
    with open(index_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index_maps(
    path: str | Path | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Build bidirectional maps from ``index.json``.

    Returns
    -------
    rel_to_dcm :
        ``batch/dcm_stem`` -> original DCM location (index key)
    dcm_to_rel :
        original DCM location -> ``batch/dcm_stem``
    """
    dcm_to_rel: dict[str, str] = {}
    for dcm_path, rel_value in load_index(path).items():
        dcm_to_rel[dcm_path] = require_dcm_relative_key(rel_value)

    rel_to_dcm = {rel: dcm for dcm, rel in dcm_to_rel.items()}
    return rel_to_dcm, dcm_to_rel


def build_dcm_stem_to_rel_map(rel_to_dcm: dict[str, str]) -> dict[str, str]:
    """Map cleaned DCM folder name -> ``batch/dcm_stem`` (raises if ambiguous)."""
    stem_to_rel: dict[str, str] = {}
    for rel_key in rel_to_dcm:
        stem = Path(rel_key).name
        if stem in stem_to_rel and stem_to_rel[stem] != rel_key:
            raise ValueError(
                f"Ambiguous DCM stem '{stem}' maps to both "
                f"'{stem_to_rel[stem]}' and '{rel_key}'."
            )
        stem_to_rel[stem] = rel_key
    return stem_to_rel


def dcm_relative_key_from_cvat_path(path: str | Path) -> str | None:
    """
    Extract ``batch_{name}/dcm_stem`` from a CVAT/YOLO path that includes
    ``digital_lab_wired_etalons/data_as_images/...``.
    """
    path_str = str(path).replace("\\", "/")
    marker = f"{CVAT_DATA_AS_IMAGES_PREFIX}/"
    if marker not in path_str:
        return None
    suffix = path_str.split(marker, 1)[1]
    parts = Path(suffix).parts
    if len(parts) < 2:
        return None
    try:
        return require_dcm_relative_key(f"{parts[0]}/{parts[1]}")
    except RelativeKeyError:
        return None


def resolve_dcm_relative_key(
    dcm_stem: str,
    *,
    path_hint: str | Path | None = None,
    rel_to_dcm: dict[str, str] | None = None,
    stem_to_rel: dict[str, str] | None = None,
    index_file: str | Path | None = None,
) -> str | None:
    """
    Resolve a DCM folder name to ``batch_{name}/dcm_stem`` using CVAT path hints and/or index maps.
    """
    if path_hint is not None:
        from_cvat = dcm_relative_key_from_cvat_path(path_hint)
        if from_cvat is not None:
            return from_cvat

    if stem_to_rel is None or rel_to_dcm is None:
        rel_to_dcm, _ = load_index_maps(index_file)
        stem_to_rel = build_dcm_stem_to_rel_map(rel_to_dcm)

    candidate = stem_to_rel.get(dcm_stem)
    if candidate is None:
        return None
    return require_dcm_relative_key(candidate)


def get_data_maps(location: str) -> dict[str, str]:
    """
    Maps DCM source folders to batch subdirectory names under ``data_as_images``.

    Each key is the folder containing DCM data; each value is the target batch name.
    """
    if location == "local":
        base = get_data_dir() / "data_dcm"
        return {
            str(base / "Мерник"): "batch_large_diameters",
            # str(base / "folder_name_1"): "batch_small_diameteres_1",
            # str(base / "folder_name_2"): "batch_small_diameteres_2",
        }
    if location == "remote":
        remote_base = "/mnt/datastore/svarka/dataset/dl_data_v3/Сортированные снимки"
        return {
            f"{remote_base}/Мерник": "batch_large_diameters",
            f"{remote_base}/Суперимпоз": "batch_small_diameteres_1",
            f"{remote_base}/Эллипс": "batch_small_diameteres_2",
        }
    raise ValueError(f"Unknown location: '{location}'. Expected 'local' or 'remote'.")


def segmentation_weights_path() -> Path:
    return get_data_dir() / "weights" / "latest_segmentation_model.pt"


def resolve_latest_dataset_hash() -> str | None:
    cache_dir = labeling_root() / "cache"
    if not cache_dir.exists():
        return None
    subdirs = [d for d in cache_dir.iterdir() if d.is_dir() and len(d.name) == 32]
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime).name


def resolve_dataset_hash_by_version_name(version_name: str) -> str:
    """Look up a cache ``dataset_hash`` by ``dataset_config.json`` ``version_name``."""
    cache_dir = labeling_root() / "cache"
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Labeling cache not found: {cache_dir}")

    matches: list[str] = []
    for subdir in cache_dir.iterdir():
        if not subdir.is_dir() or len(subdir.name) != 32:
            continue
        config_path = subdir / "dataset_config.json"
        if not config_path.is_file():
            continue
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("version_name") == version_name:
            matches.append(subdir.name)

    if not matches:
        raise FileNotFoundError(
            f"No dataset cache found for version_name={version_name!r}. "
            "Run obj_det_ds_prep.py first."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple dataset caches match version_name={version_name!r}: {matches}"
        )
    return matches[0]


def iter_split_artifact_dirs(dataset_hash: str) -> Iterator[Path]:
    """Yield split artifact directories under ``artifacts/{dataset_hash}/``."""
    root = dataset_artifacts_root(dataset_hash)
    if not root.is_dir():
        return
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name in SPLIT_ARTIFACT_RESERVED_DIRS:
            continue
        if len(child.name) == 32:
            yield child


def iter_anchor_artifact_dirs(
    dataset_hash: str,
    split_hash: str,
) -> Iterator[Path]:
    """Yield anchor-config artifact roots under ``artifacts/{dataset_hash}/{split_hash}/``."""
    root = split_artifacts_root(dataset_hash, split_hash)
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child.is_dir() and len(child.name) == 32:
            yield child


def legacy_runs_dataset_root(dataset_hash: str) -> Path:
    """Legacy artifact root kept for manual migration only."""
    return runs_root() / dataset_hash


def runs_root() -> Path:
    """Root of downloaded / legacy full-training runs."""
    return artifacts_root() / "runs"


def legacy_runs_split_root(dataset_hash: str, split_hash: str) -> Path:
    return runs_root() / dataset_hash / split_hash


def legacy_run_dir(dataset_hash: str, split_hash: str, experiment_hash: str) -> Path:
    """
    One full-training experiment under the legacy ``runs/`` tree::

        runs/{dataset_hash}/{split_hash}/{experiment_hash}/
            experiment_config.json
            anchor_config.json
            anchor_training_spec.json
            checkpoints/best_model.pt
            metrics/  tb_logs/  viz/
    """
    return legacy_runs_split_root(dataset_hash, split_hash) / experiment_hash


EXPERIMENT_CONFIG_FILENAME = "experiment_config.json"
ANCHOR_TRAINING_SPEC_FILENAME = "anchor_training_spec.json"
RUN_CHECKPOINT_REL_PATH = Path("checkpoints") / "best_model.pt"


def run_checkpoint_path(run_dir: Path) -> Path:
    return Path(run_dir) / RUN_CHECKPOINT_REL_PATH


def localize_labeling_path(path: str | Path) -> Path:
    """Map an absolute server path to the local ``data/labeling/...`` tree."""
    path_str = str(path).replace("\\", "/")
    for marker in ("/data/labeling/", "/labeling/"):
        if marker in path_str:
            suffix = path_str.split(marker, 1)[1]
            return labeling_root() / suffix
    return Path(path)


def resolve_existing_labeling_path(path: str | Path) -> Path | None:
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    localized = localize_labeling_path(candidate)
    if localized.is_file():
        return localized.resolve()
    return None


def resolve_run_anchor_manifest(
    run_dir: Path,
    explicit: str | Path | None = None,
) -> Path:
    """
    Resolve ``anchor_config.json`` for inference from a legacy run directory.

    Prefers the manifest copied into ``run_dir``; falls back to split-level
    ``anchor_configs/`` and server paths remapped via :func:`localize_labeling_path`.
    """
    run_dir = Path(run_dir).resolve()
    if explicit:
        resolved = resolve_existing_labeling_path(explicit)
        if resolved is not None:
            return resolved

    run_manifest = run_dir / ANCHOR_MANIFEST_FILENAME
    if run_manifest.is_file():
        return run_manifest.resolve()

    spec_path = run_dir / ANCHOR_TRAINING_SPEC_FILENAME
    if spec_path.is_file():
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        resolved = resolve_existing_labeling_path(spec.get("manifest_path", ""))
        if resolved is not None:
            return resolved

    exp_config_path = run_dir / EXPERIMENT_CONFIG_FILENAME
    if exp_config_path.is_file():
        with open(exp_config_path, "r", encoding="utf-8") as f:
            exp_config = json.load(f)

        anchor_config_hash = exp_config.get("anchor_config_hash")
        split_hash = exp_config.get("split_hash")
        dataset_hash = exp_config.get("dataset_hash")
        if anchor_config_hash and split_hash and dataset_hash:
            split_manifest = (
                legacy_runs_split_root(dataset_hash, split_hash)
                / "anchor_configs"
                / anchor_config_hash
                / ANCHOR_MANIFEST_FILENAME
            )
            if split_manifest.is_file():
                return split_manifest.resolve()

        anchor_spec = exp_config.get("anchor_training_spec", {})
        resolved = resolve_existing_labeling_path(anchor_spec.get("manifest_path", ""))
        if resolved is not None:
            return resolved

    raise FileNotFoundError(
        f"Could not resolve anchor manifest for run directory {run_dir}. "
        f"Expected {run_manifest.name} in the run directory."
    )


def resolve_run_img_size(
    run_dir: Path,
    img_size: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    if img_size is not None:
        return img_size

    exp_config_path = Path(run_dir) / EXPERIMENT_CONFIG_FILENAME
    if not exp_config_path.is_file():
        return None

    with open(exp_config_path, "r", encoding="utf-8") as f:
        exp_config = json.load(f)
    model_img_size = exp_config.get("model_params", {}).get("img_size")
    if model_img_size is not None:
        return tuple(model_img_size)
    return None
