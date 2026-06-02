"""
Shared utilities for RetinaNet error-analysis scripts.

Experiment path resolution, validation data loading, and output directory naming.
No model forward logic — see ``modules/retinanet/error_analysis/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dl_lib.common_tools.path_utils import get_data_dir
from dl_lib.etalon_object_detection.modules.ds_utils import (
    DL_LIB_ETALON_FIXED_SIZE,
    WeldingDetectionDataset,
    collate_fn,
    get_path_split_callables,
)


# ---------------------------------------------------------------------------
# Experiment paths (same contract as retinanet_inference.py)
# ---------------------------------------------------------------------------


def resolve_experiment_paths(
    dataset_hash: str = "latest",
    split_hash: str = "latest",
    exp_hash: str = "latest",
    artifacts_subdir: str = "error_analysis",
) -> Dict[str, Path]:
    """
    Autodiscover dataset / split / experiment hashes and return canonical paths.

    Parameters
    ----------
    artifacts_subdir :
        Subfolder under ``exp_dir`` for EA outputs (default ``error_analysis``).
        Inference scripts may use e.g. ``val_inference``.
    """
    base_data = get_data_dir()
    runs_root = base_data / "labeling" / "artifacts" / "runs"

    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found at {runs_root}")

    if dataset_hash == "latest":
        ds_dirs = [d for d in runs_root.iterdir() if d.is_dir() and len(d.name) == 32]
        if not ds_dirs:
            raise FileNotFoundError("No dataset runs found.")
        dataset_hash = max(ds_dirs, key=lambda d: d.stat().st_mtime).name

    ds_path = runs_root / dataset_hash

    if split_hash == "latest":
        split_dirs = [d for d in ds_path.iterdir() if d.is_dir() and len(d.name) == 32]
        if not split_dirs:
            raise FileNotFoundError(f"No splits found for dataset {dataset_hash}")
        split_hash = max(split_dirs, key=lambda d: d.stat().st_mtime).name

    split_path = ds_path / split_hash

    if exp_hash == "latest":
        exp_dirs = [d for d in split_path.iterdir() if d.is_dir() and len(d.name) == 32]
        if not exp_dirs:
            raise FileNotFoundError(f"No experiments found for split {split_hash}")
        exp_hash = max(exp_dirs, key=lambda d: d.stat().st_mtime).name

    exp_path = split_path / exp_hash
    cache_root = base_data / "labeling" / "cache" / dataset_hash

    return {
        "exp_dir": exp_path,
        "config": exp_path / "experiment_config.json",
        "checkpoint": exp_path / "checkpoints" / "best_model.pt",
        "anchor_config": split_path / "anchor_config.json",
        "master_json": cache_root / "master_labels.json",
        "images_dir": base_data / "labeling" / "data_as_images",
        "cache_dir": cache_root,
        "artifacts_root": exp_path / artifacts_subdir,
    }


# ---------------------------------------------------------------------------
# Output directory naming
# ---------------------------------------------------------------------------


def fmt_param_float(value: float) -> str:
    """Filesystem-safe float (0.1 -> ``0p10``)."""
    return f"{value:.2f}".replace(".", "p")


def build_objectness_ea_output_dir(
    artifacts_root: Path,
    iou_overlap_min: float,
    iou_eval: float,
    nms_thresh: float,
    score_thresh: float,
    conf_threshold_for_breakdown: float,
    fg_iou_threshold: float,
    run_layer_b: bool,
    k_extremes: int,
    run_grid_metrics: bool = False,
) -> Path:
    """
    One folder per objectness EA configuration, e.g.::

        objectness_ea/iouOm0p10_iouEv0p50_nms1p00_sc0p01_..._layerAB_grid/
    """
    tags = ["layerAB" if run_layer_b else "layerA"]
    if run_grid_metrics:
        tags.append("grid")
    layer_tag = "_".join(tags)

    name = (
        f"objectness_ea/"
        f"iouOm{fmt_param_float(iou_overlap_min)}_iouEv{fmt_param_float(iou_eval)}_"
        f"nms{fmt_param_float(nms_thresh)}_sc{fmt_param_float(score_thresh)}_"
        f"confBd{fmt_param_float(conf_threshold_for_breakdown)}_"
        f"fgIoU{fmt_param_float(fg_iou_threshold)}_k{k_extremes}_{layer_tag}"
    )
    return artifacts_root / name


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


OBJECTNESS_EA_REQUIRED_KEYS: Tuple[str, ...] = (
    "dataset_hash",
    "split_hash",
    "exp_hash",
    "iou_overlap_min",
    "iou_eval",
    "nms_thresh",
    "score_thresh",
    "conf_threshold_for_breakdown",
    "fg_iou_threshold",
    "k_extremes",
)


def validate_objectness_ea_config(config: Dict[str, Any]) -> None:
    """Raise ``KeyError`` if required objectness EA keys are missing."""
    missing = [k for k in OBJECTNESS_EA_REQUIRED_KEYS if k not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")


# ---------------------------------------------------------------------------
# Experiment config & validation loader
# ---------------------------------------------------------------------------


def load_experiment_config(paths: Dict[str, Path]) -> Dict[str, Any]:
    """Load ``experiment_config.json`` for the resolved experiment."""
    with open(paths["config"], "r", encoding="utf-8") as f:
        return json.load(f)


def get_eval_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_val_dataloader(
    paths: Dict[str, Path],
    experiment_config: Dict[str, Any],
    batch_size: int,
) -> Tuple[WeldingDetectionDataset, DataLoader]:
    """Reconstruct the validation split DataLoader from a trained experiment."""
    _, val_filter, _ = get_path_split_callables(
        master_json_path=str(paths["master_json"]),
        val_ratio=experiment_config["split_params"]["val_ratio"],
        seed=experiment_config["split_params"]["seed"],
        save_dir=str(paths["cache_dir"]),
    )

    val_dataset = WeldingDetectionDataset(
        images_dir=str(paths["images_dir"]),
        cache_dir=str(paths["cache_dir"]),
        target_size=DL_LIB_ETALON_FIXED_SIZE,
        initial_2_ds_cls_ids=experiment_config["initial_2_ds_cls_ids"],
        transformations=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        path_filter=val_filter,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return val_dataset, val_loader


def assert_checkpoint_exists(paths: Dict[str, Path]) -> None:
    if not paths["checkpoint"].exists():
        raise FileNotFoundError(f"Checkpoint not found: {paths['checkpoint']}")


def image_stem_from_sample_path(img_path_rel: str) -> str:
    """Stable filename stem: ``dcm_name_frame_stem``."""
    p = Path(img_path_rel)
    return f"{p.parent.name}_{p.stem}"


def default_conf_thresholds() -> List[float]:
    return [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
