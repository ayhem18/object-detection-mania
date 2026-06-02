"""Shared helpers for parameter sweep scripts."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dl_lib.etalon_object_detection.modules.ds_utils import (
    WeldingDetectionDataset,
    collate_fn,
)
from dl_lib.etalon_object_detection.modules.path_layout import (
    batch_name_from_relative_key,
    dcm_stem_from_relative_key,
)
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_train import _single_iteration

logger = logging.getLogger(__name__)

SUBSET_SAMPLING_STRATEGY = "per_data_batch_dcm_round_robin"


def compute_subset_hash(subset_params: Dict[str, Any], split_hash: str) -> str:
    payload = {
        "split_hash": split_hash,
        "sampling_strategy": SUBSET_SAMPLING_STRATEGY,
        "max_samples_per_batch": subset_params["max_samples"],
        "seed": subset_params["seed"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def frame_relative_key_from_img_path(img_path: str) -> str:
    """``batch/dcm_stem/frame_001.png`` -> ``batch/dcm_stem/frame_001``."""
    return img_path[: -len(".png")] if img_path.endswith(".png") else img_path


def group_train_paths_by_data_batch(train_paths: List[str]) -> Dict[str, List[str]]:
    """Group ``master_labels.json`` frame paths by top-level ``batch_{name}`` folder."""
    by_batch: Dict[str, List[str]] = {}
    for img_path in train_paths:
        frame_key = frame_relative_key_from_img_path(img_path)
        batch_name = batch_name_from_relative_key(frame_key)
        by_batch.setdefault(batch_name, []).append(img_path)
    return {batch: sorted(paths) for batch, paths in sorted(by_batch.items())}


def _rng_for_data_batch(seed: int, batch_name: str) -> random.Random:
    digest = hashlib.md5(f"{seed}\0{batch_name}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _select_paths_within_data_batch(
    batch_paths: List[str],
    max_samples: int,
    rng: random.Random,
) -> List[str]:
    """
    Select up to ``max_samples`` frames from one data batch.

    Frames are drawn round-robin across DCM stems so consecutive picks prefer
    different ``dcm_stem`` folders when possible.
    """
    dcm_to_paths: Dict[str, List[str]] = {}
    for img_path in batch_paths:
        frame_key = frame_relative_key_from_img_path(img_path)
        dcm_key = f"{batch_name_from_relative_key(frame_key)}/{dcm_stem_from_relative_key(frame_key)}"
        dcm_to_paths.setdefault(dcm_key, []).append(img_path)

    for paths in dcm_to_paths.values():
        rng.shuffle(paths)

    dcm_order = sorted(dcm_to_paths.keys())
    rng.shuffle(dcm_order)

    selected: List[str] = []
    while len(selected) < max_samples:
        progressed = False
        for dcm_key in dcm_order:
            if len(selected) >= max_samples:
                break
            if dcm_to_paths[dcm_key]:
                selected.append(dcm_to_paths[dcm_key].pop())
                progressed = True
        if not progressed:
            break
    return sorted(selected)


def select_subset_paths_by_data_batch(
    train_paths: List[str],
    max_samples_per_batch: int,
    seed: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Build an overfitting subset with a fixed quota **per data batch**.

    Uses ``master_labels.json`` relative keys to discover ``batch_{name}`` folders
    and ``dcm_stem`` groups, then selects frames round-robin across DCMs within
    each batch.
    """
    if max_samples_per_batch <= 0:
        raise ValueError("subset_params.max_samples must be positive (per data batch).")

    by_batch = group_train_paths_by_data_batch(train_paths)
    if not by_batch:
        raise ValueError("No training paths available for subset selection.")

    batch_details: Dict[str, Dict[str, Any]] = {}
    selected_paths: List[str] = []

    for batch_name, batch_paths in by_batch.items():
        dcm_stems = sorted(
            {
                dcm_stem_from_relative_key(frame_relative_key_from_img_path(p))
                for p in batch_paths
            }
        )
        batch_rng = _rng_for_data_batch(seed, batch_name)
        batch_selected = _select_paths_within_data_batch(
            batch_paths,
            max_samples_per_batch,
            batch_rng,
        )
        selected_paths.extend(batch_selected)
        batch_details[batch_name] = {
            "num_train_frames": len(batch_paths),
            "num_dcm_stems": len(dcm_stems),
            "dcm_stems": dcm_stems,
            "num_selected": len(batch_selected),
            "selected_paths": batch_selected,
        }

    metadata: Dict[str, Any] = {
        "sampling_strategy": SUBSET_SAMPLING_STRATEGY,
        "max_samples_per_batch": max_samples_per_batch,
        "seed": seed,
        "num_data_batches": len(by_batch),
        "num_selected_total": len(selected_paths),
        "data_batches": batch_details,
        "paths": sorted(selected_paths),
    }
    return sorted(selected_paths), metadata


def get_no_aug_transforms() -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or ("cuda error" in message and "memory" in message)


def probe_batch_size(
    *,
    build_model: Callable[[], torch.nn.Module],
    dataset: WeldingDetectionDataset,
    batch_size: int,
    device: torch.device,
) -> bool:
    model = build_model()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    try:
        images, targets = next(iter(loader))
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        batch_metrics = _single_iteration(model, images, targets)
        if not math.isfinite(batch_metrics["loss"]):
            return False
        batch_metrics["loss_tensor"].backward()
        optimizer.step()
        return True
    except RuntimeError as exc:
        if is_oom_error(exc):
            return False
        raise
    finally:
        del model
        clear_cuda_cache()


def find_max_batch_size(
    *,
    build_model: Callable[[], torch.nn.Module],
    dataset: WeldingDetectionDataset,
    device: torch.device,
    min_batch_size: int = 1,
    max_batch_size: int = 64,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Binary search for the largest batch size that completes one train step."""
    if len(dataset) == 0:
        return 0, []

    max_batch_size = min(max_batch_size, len(dataset))
    probe_log: List[Dict[str, Any]] = []

    if not torch.cuda.is_available():
        batch_size = max_batch_size
        probe_log.append(
            {
                "batch_size": batch_size,
                "success": True,
                "note": "CPU run; skipped OOM probing.",
            }
        )
        return batch_size, probe_log

    low = min_batch_size
    high = max_batch_size
    best = 0

    while low <= high:
        mid = (low + high) // 2
        success = probe_batch_size(
            build_model=build_model,
            dataset=dataset,
            batch_size=mid,
            device=device,
        )
        probe_log.append({"batch_size": mid, "success": success})
        logger.info("Batch size probe: %d -> %s", mid, "ok" if success else "OOM")

        if success:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best, probe_log


def load_capacity_sweep_runs(capacity_summary_path: Path) -> List[Dict[str, Any]]:
    """
    Load ``(freeze_layers, batch_size)`` pairs from a capacity ``sweep_summary.json``.

    Skips rows with missing or non-positive ``max_batch_size``.
    """
    with open(capacity_summary_path, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    runs: List[Dict[str, Any]] = []
    for row in summaries:
        if row.get("capacity_skipped"):
            continue
        batch_size = row.get("max_batch_size")
        if batch_size is None or batch_size <= 0:
            continue
        runs.append(
            {
                "freeze_layers": int(row["freeze_layers"]),
                "batch_size": int(batch_size),
            }
        )
    return runs


def load_sweep_config_adjacent_to_summary(capacity_summary_path: Path) -> Dict[str, Any]:
    sweep_config_path = capacity_summary_path.parent / "sweep_config.json"
    if not sweep_config_path.is_file():
        raise FileNotFoundError(
            f"Expected sweep_config.json next to capacity summary: {sweep_config_path}"
        )
    with open(sweep_config_path, "r", encoding="utf-8") as f:
        return json.load(f)
