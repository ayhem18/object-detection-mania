"""
Learning-rate finder for each ``(freeze_layers, batch_size)`` pair.

Uses the full training split and log-linear LR spacing. After the sweep, analyzes
the loss curve to find ``lr_decrease_onset`` (where loss starts falling
consistently) and ``lr_increase_onset`` (where loss starts rising consistently).
Batch sizes come from a prior capacity sweep (``model_params_sweep.py``) or
explicit CLI flags.

Run from repo root::

    uv run python src/dl_lib/etalon_object_detection/scripts/anchors/register_anchor_configs.py

From capacity sweep summary::

    uv run python .../lr_sweep.py --capacity-summary path/to/sweep_summary.json

Single ``(freeze_layers, batch_size)`` run::

    uv run python .../lr_sweep.py --freeze-layers 2 --batch-size 8
    uv run python .../lr_sweep.py --freeze-layers 2 --batch-size 8 0fdeb8457bec592333f887d9f563a360

Artifact layout::

    artifacts/{dataset_hash}/{split_hash}/{anchor_config_hash}/sweeps/lr_find/frozen_layers_{n}/
        run_config.json
        lr_find.jsonl
        lr_find_summary.json   # lr_decrease_onset, lr_increase_onset, ...
        lr_find_plot.png
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dl_lib.etalon_object_detection.modules.anchors.anchor_config_registry import (
    DIMENSION_BASED_ANCHOR_CONFIG,
    load_registered_anchor_config,
    recipe_to_payload,
)
from dl_lib.etalon_object_detection.modules.ds_utils import (
    DL_LIB_ETALON_FIXED_SIZE,
    WeldingDetectionDataset,
    collate_fn,
    get_path_split_callables,
)
from dl_lib.etalon_object_detection.modules.path_layout import (
    anchor_artifacts_root,
    images_dir,
    labels_data_dir,
    master_labels_path,
    resolve_latest_dataset_hash,
    segmentation_weights_path,
    sweeps_lr_find_dir,
    sweeps_lr_find_run_dir,
)
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_anchors import (
    ensure_anchor_training_spec,
    get_anchor_config_hash,
)
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_detector import (
    build_dl_lab_etalon_retinanet_from_segmentation_weights,
)
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_train import _single_iteration
from dl_lib.etalon_object_detection.scripts.object_detection.parameter_sweeps.sweep_utils import (
    clear_cuda_cache,
    get_no_aug_transforms,
    load_capacity_sweep_runs,
    load_sweep_config_adjacent_to_summary,
)
from mypt.code_utils.pytorch_utils import seed_everything

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_ANCHOR_CONFIG_HASH = recipe_to_payload(DIMENSION_BASED_ANCHOR_CONFIG)["config_hash"]

_DEFAULT_LR_FIND_PARAMS = {
    "start_lr": 1e-7,
    "end_lr": 1.0,
    "num_iterations": 100,
    "min_consecutive_steps": 10,
    "smooth_window": 5,
}


def log_linear_lr_schedule(
    start_lr: float,
    end_lr: float,
    num_iterations: int,
) -> np.ndarray:
    """Log-linear (geometric) LR values from ``start_lr`` to ``end_lr``."""
    if start_lr <= 0 or end_lr <= 0:
        raise ValueError("start_lr and end_lr must be positive for log-linear spacing.")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive.")
    if num_iterations == 1:
        return np.array([start_lr], dtype=np.float64)
    return np.logspace(
        math.log10(start_lr),
        math.log10(end_lr),
        num=num_iterations,
        dtype=np.float64,
    )


def _smooth_losses(losses: np.ndarray, window: int) -> np.ndarray:
    if len(losses) == 0:
        return losses.copy()
    window = max(1, min(window, len(losses)))
    if window == 1:
        return losses.copy()

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(losses, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def _delta_threshold(smoothed: np.ndarray) -> float:
    """Noise floor estimated from the early, low-LR portion of the curve."""
    if len(smoothed) < 2:
        return 1e-4 * float(np.nanmean(smoothed))

    early_len = max(3, len(smoothed) // 10)
    early = smoothed[:early_len]
    if len(early) < 2:
        return 1e-4 * float(np.nanmean(smoothed))

    mad = float(np.median(np.abs(np.diff(early))))
    return max(mad * 2.0, 1e-4 * float(np.nanmean(smoothed)))


def _first_consistent_trend_index(
    deltas: np.ndarray,
    *,
    direction: str,
    threshold: float,
    min_consecutive: int,
    start: int = 0,
) -> int | None:
    if min_consecutive <= 0:
        raise ValueError("min_consecutive must be positive.")
    if len(deltas) < min_consecutive:
        return None

    start = max(0, start)
    for idx in range(start, len(deltas) - min_consecutive + 1):
        window = deltas[idx : idx + min_consecutive]
        if direction == "decrease":
            if np.all(window < -threshold):
                return idx
        elif direction == "increase":
            if np.all(window > threshold):
                return idx
        else:
            raise ValueError(f"Unknown direction: {direction}")
    return None


def analyze_lr_curve(
    lrs: List[float],
    losses: List[float],
    *,
    min_consecutive: int = 3,
    smooth_window: int = 5,
) -> Dict[str, Any]:
    """
    Find LR thresholds from a log-linear sweep curve.

    Returns the LR where loss begins decreasing consistently (lower bound of the
    useful range) and where it begins increasing consistently (upper bound).
    """
    if not lrs or not losses:
        return {
            "lr_decrease_onset": None,
            "lr_increase_onset": None,
            "lr_decrease_onset_step": None,
            "lr_increase_onset_step": None,
            "min_loss": None,
            "min_loss_lr": None,
        }

    lrs_arr = np.array(lrs, dtype=np.float64)
    losses_arr = np.array(losses, dtype=np.float64)

    finite_mask = np.isfinite(losses_arr)
    lrs_arr = lrs_arr[finite_mask]
    losses_arr = losses_arr[finite_mask]
    if len(losses_arr) == 0:
        return {
            "lr_decrease_onset": None,
            "lr_increase_onset": None,
            "lr_decrease_onset_step": None,
            "lr_increase_onset_step": None,
            "min_loss": None,
            "min_loss_lr": None,
        }

    smoothed = _smooth_losses(losses_arr, smooth_window)
    deltas = np.diff(smoothed)
    threshold = _delta_threshold(smoothed)
    min_loss_idx = int(np.argmin(smoothed))

    decrease_idx = _first_consistent_trend_index(
        deltas,
        direction="decrease",
        threshold=threshold,
        min_consecutive=min_consecutive,
        start=0,
    )
    increase_search_start = decrease_idx if decrease_idx is not None else 0
    increase_idx = _first_consistent_trend_index(
        deltas,
        direction="increase",
        threshold=threshold,
        min_consecutive=min_consecutive,
        start=max(increase_search_start, min_loss_idx + 1),
    )

    return {
        "lr_decrease_onset": float(lrs_arr[decrease_idx]) if decrease_idx is not None else None,
        "lr_increase_onset": float(lrs_arr[increase_idx]) if increase_idx is not None else None,
        "lr_decrease_onset_step": decrease_idx,
        "lr_increase_onset_step": increase_idx,
        "min_loss": float(np.min(losses_arr)),
        "min_loss_lr": float(lrs_arr[min_loss_idx]),
        "delta_threshold": threshold,
        "min_consecutive_steps": min_consecutive,
        "smooth_window": smooth_window,
    }


def run_lr_finder(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    start_lr: float,
    end_lr: float,
    num_iterations: int,
    min_consecutive: int = 3,
    smooth_window: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """LR sweep with log-linear spacing over ``num_iterations`` training steps."""
    if num_iterations <= 0:
        raise ValueError("lr_find_params.num_iterations must be positive.")

    learning_rates = log_linear_lr_schedule(start_lr, end_lr, num_iterations)
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr, weight_decay=1e-4)

    records: List[Dict[str, Any]] = []
    data_iter = iter(train_loader)

    model.train()
    for step, current_lr in enumerate(learning_rates):
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)

        current_lr = float(current_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        batch_metrics = _single_iteration(model, images, targets)
        loss = batch_metrics["loss"]

        record = {
            "step": step,
            "learning_rate": current_lr,
            "loss": loss,
            "loss_classifier": batch_metrics["loss_classifier"],
            "loss_box_reg": batch_metrics["loss_box_reg"],
        }
        records.append(record)

        if not math.isfinite(loss):
            logger.warning("LR finder stopped at step %d: non-finite loss.", step)
            break

        batch_metrics["loss_tensor"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step == 0 or (step + 1) % max(1, num_iterations // 10) == 0:
            print(
                f"LR find {step + 1:03d}/{num_iterations:03d} | "
                f"lr={current_lr:.2e} | loss={loss:.4f}"
            )

    lrs = [record["learning_rate"] for record in records]
    losses = [record["loss"] for record in records]
    analysis = analyze_lr_curve(
        lrs,
        losses,
        min_consecutive=min_consecutive,
        smooth_window=smooth_window,
    )

    summary = {
        "start_lr": start_lr,
        "end_lr": end_lr,
        "num_iterations": num_iterations,
        "lr_spacing": "log_linear",
        "completed_steps": len(records),
        **analysis,
    }
    return records, summary


def _save_lr_find_plot(
    records: List[Dict[str, Any]],
    plot_path: Path,
    *,
    lr_decrease_onset: float | None = None,
    lr_increase_onset: float | None = None,
) -> None:
    if not records:
        return
    lrs = [record["learning_rate"] for record in records]
    losses = [record["loss"] for record in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, losses)
    if lr_decrease_onset is not None:
        ax.axvline(
            lr_decrease_onset,
            color="green",
            linestyle="--",
            linewidth=1.2,
            label=f"decrease onset ({lr_decrease_onset:.2e})",
        )
    if lr_increase_onset is not None:
        ax.axvline(
            lr_increase_onset,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"increase onset ({lr_increase_onset:.2e})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Loss")
    ax.set_title("LR finder")
    ax.grid(True, alpha=0.3)
    if lr_decrease_onset is not None or lr_increase_onset is not None:
        ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def _resolve_train_split(
    *,
    dataset_hash: str,
    config: Dict[str, Any],
    split_hash: str | None = None,
) -> Tuple[str, Callable[[str], bool], int]:
    labels_dir = labels_data_dir(dataset_hash)
    master_json = master_labels_path(dataset_hash)

    split_params = config["split_params"]
    train_filter, _, resolved_split_hash = get_path_split_callables(
        master_json_path=str(master_json),
        val_ratio=split_params["val_ratio"],
        seed=split_params["seed"],
        save_dir=str(labels_dir.parent),
    )
    if split_hash is None:
        split_hash = resolved_split_hash
    elif split_hash != resolved_split_hash:
        raise ValueError(
            f"split_hash mismatch: config has {split_hash}, "
            f"resolved split is {resolved_split_hash}."
        )

    with open(master_json, "r", encoding="utf-8") as f:
        all_samples = json.load(f).get("samples", [])
    num_train_samples = sum(
        1 for sample in all_samples if train_filter(sample["img_path"])
    )
    return split_hash, train_filter, num_train_samples


def run_single_lr_find(
    *,
    freeze_layers: int,
    batch_size: int,
    config: Dict[str, Any],
    dataset_hash: str,
    split_hash: str,
    train_filter: Callable[[str], bool],
    anchor_spec: Any,
    device: torch.device,
) -> Dict[str, Any]:
    run_dir = sweeps_lr_find_run_dir(
        dataset_hash, split_hash, anchor_spec.config_hash, freeze_layers
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "freeze_layers": freeze_layers,
        "batch_size": batch_size,
        "anchor_config_hash": anchor_spec.config_hash,
        "split_params": config["split_params"],
        "lr_find_params": config["lr_find_params"],
        "model_params": config["model_params"],
        "initial_2_ds_cls_ids": config["initial_2_ds_cls_ids"],
        "seed": config["seed"],
        "artifact_dir": str(run_dir),
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4, ensure_ascii=False)

    seed_everything(config["seed"])
    labels_dir = labels_data_dir(dataset_hash)
    model_params = config["model_params"]

    train_dataset = WeldingDetectionDataset(
        images_dir=str(images_dir()),
        cache_dir=str(labels_dir),
        target_size=DL_LIB_ETALON_FIXED_SIZE,
        initial_2_ds_cls_ids=config["initial_2_ds_cls_ids"],
        transformations=get_no_aug_transforms(),
        path_filter=train_filter,
    )

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    logger.info(
        "freeze_layers=%d: batch_size=%d, train_samples=%d, running LR finder.",
        freeze_layers,
        batch_size,
        len(train_dataset),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    model = build_dl_lab_etalon_retinanet_from_segmentation_weights(
        anchor_spec=anchor_spec,
        segmentation_weights_path=str(segmentation_weights_path()),
        img_size=model_params["img_size"],
        device=device,
        freeze_layers=freeze_layers,
    )

    lr_params = config["lr_find_params"]
    records, lr_summary = run_lr_finder(
        model=model,
        train_loader=train_loader,
        device=device,
        start_lr=lr_params["start_lr"],
        end_lr=lr_params["end_lr"],
        num_iterations=lr_params["num_iterations"],
        min_consecutive=lr_params.get("min_consecutive_steps", 3),
        smooth_window=lr_params.get("smooth_window", 5),
    )

    with open(run_dir / "lr_find.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    lr_summary = {
        **lr_summary,
        "freeze_layers": freeze_layers,
        "batch_size": batch_size,
        "num_train_samples": len(train_dataset),
    }
    with open(run_dir / "lr_find_summary.json", "w", encoding="utf-8") as f:
        json.dump(lr_summary, f, indent=4, ensure_ascii=False)

    _save_lr_find_plot(
        records,
        run_dir / "lr_find_plot.png",
        lr_decrease_onset=lr_summary.get("lr_decrease_onset"),
        lr_increase_onset=lr_summary.get("lr_increase_onset"),
    )

    del model
    clear_cuda_cache()

    return {
        "freeze_layers": freeze_layers,
        "artifact_dir": str(run_dir),
        "batch_size": batch_size,
        "num_train_samples": len(train_dataset),
        "lr_decrease_onset": lr_summary.get("lr_decrease_onset"),
        "lr_increase_onset": lr_summary.get("lr_increase_onset"),
        "min_loss_lr": lr_summary.get("min_loss_lr"),
        "min_loss": lr_summary.get("min_loss"),
        "lr_find_skipped": False,
    }


def run_lr_sweep(
    config: Dict[str, Any],
    runs: List[Dict[str, int]],
) -> None:
    if not runs:
        raise ValueError("No LR finder runs to execute (empty runs list).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_hash = config["dataset_hash"]
    labels_dir = labels_data_dir(dataset_hash)

    split_hash, train_filter, num_train_samples = _resolve_train_split(
        dataset_hash=dataset_hash,
        config=config,
        split_hash=config.get("split_hash"),
    )

    anchor_config_hash = get_anchor_config_hash(config)
    anchor_artifacts_root(dataset_hash, split_hash, anchor_config_hash).mkdir(
        parents=True, exist_ok=True
    )

    anchor_dataset = WeldingDetectionDataset(
        images_dir=str(images_dir()),
        cache_dir=str(labels_dir),
        target_size=DL_LIB_ETALON_FIXED_SIZE,
        initial_2_ds_cls_ids=config["initial_2_ds_cls_ids"],
        transformations=None,
        path_filter=train_filter,
    )
    anchor_spec = ensure_anchor_training_spec(
        config,
        dataset=anchor_dataset,
        dataset_hash=dataset_hash,
        split_hash=split_hash,
        force_reoptimize=config["force_reoptimize_anchors"],
    )

    if anchor_spec.config_hash != anchor_config_hash:
        raise ValueError(
            f"anchor_config hash mismatch: config declares {anchor_config_hash}, "
            f"resolved manifest has {anchor_spec.config_hash}."
        )

    sweep_dir = sweeps_lr_find_dir(dataset_hash, split_hash, anchor_config_hash)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    sweep_config = {
        **config,
        "split_hash": split_hash,
        "anchor_config_hash": anchor_config_hash,
        "num_train_samples": num_train_samples,
        "sweep_dir": str(sweep_dir),
        "lr_runs": runs,
    }
    with open(sweep_dir / "sweep_config.json", "w", encoding="utf-8") as f:
        json.dump(sweep_config, f, indent=4, ensure_ascii=False)

    print(f"Device: {device}")
    print(f"Dataset hash: {dataset_hash}")
    print(f"Split hash: {split_hash}")
    print(f"Train samples: {num_train_samples} (full training split)")
    print(f"Anchor: {anchor_spec.method} ({anchor_spec.config_hash})")
    print(f"Sweep artifacts: {sweep_dir}")
    print(f"LR runs: {runs}")

    summaries: List[Dict[str, Any]] = []
    for run in runs:
        summary = run_single_lr_find(
            freeze_layers=run["freeze_layers"],
            batch_size=run["batch_size"],
            config=config,
            dataset_hash=dataset_hash,
            split_hash=split_hash,
            train_filter=train_filter,
            anchor_spec=anchor_spec,
            device=device,
        )
        summaries.append(summary)

    with open(sweep_dir / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=4, ensure_ascii=False)

    print("\nLR sweep complete:")
    for row in summaries:
        if row.get("lr_find_skipped"):
            print(
                f"  freeze_layers={row['freeze_layers']}: "
                f"skipped ({row.get('reason', 'unknown')})"
            )
            continue
        lr_dec = row.get("lr_decrease_onset")
        lr_inc = row.get("lr_increase_onset")
        dec_str = f"{lr_dec:.2e}" if lr_dec is not None else "n/a"
        inc_str = f"{lr_inc:.2e}" if lr_inc is not None else "n/a"
        print(
            f"  freeze_layers={row['freeze_layers']}: "
            f"batch_size={row['batch_size']} "
            f"lr_decrease_onset={dec_str} lr_increase_onset={inc_str} "
            f"({row['artifact_dir']})"
        )


def _default_config(anchor_config_hash: str) -> Dict[str, Any]:
    return {
        "dataset_hash": "latest",
        "split_params": {
            "val_ratio": 0.15,
            "seed": 42,
        },
        "lr_find_params": dict(_DEFAULT_LR_FIND_PARAMS),
        "model_params": {
            "img_size": DL_LIB_ETALON_FIXED_SIZE,
        },
        "anchor_config_hash": anchor_config_hash,
        "force_reoptimize_anchors": False,
        "initial_2_ds_cls_ids": {0: 1},
        "seed": 42,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LR finder on the full training split for each (freeze_layers, batch_size) pair. "
            "Batch sizes come from a capacity sweep summary or explicit CLI flags."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--capacity-summary",
        type=Path,
        metavar="PATH",
        help="Path to capacity sweep_summary.json; runs LR finder for all valid rows.",
    )
    mode.add_argument(
        "--freeze-layers",
        type=int,
        metavar="N",
        help="Single run: freeze_layers value (requires --batch-size).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        metavar="B",
        help="Single run: batch size from capacity probe (requires --freeze-layers).",
    )
    parser.add_argument(
        "anchor_config_hash",
        nargs="?",
        default=_DEFAULT_ANCHOR_CONFIG_HASH,
        help=(
            "Registered anchor config hash for single-run mode "
            f"(default: {_DEFAULT_ANCHOR_CONFIG_HASH}, dimension_based). "
            "Ignored when --capacity-summary is used."
        ),
    )
    args = parser.parse_args()

    if args.freeze_layers is not None and args.batch_size is None:
        parser.error("--freeze-layers requires --batch-size.")
    if args.batch_size is not None and args.freeze_layers is None:
        parser.error("--batch-size requires --freeze-layers.")
    if args.capacity_summary is None and args.freeze_layers is None:
        parser.error("Provide --capacity-summary or both --freeze-layers and --batch-size.")

    return args


def main() -> None:
    args = parse_args()

    if args.capacity_summary is not None:
        capacity_summary_path = args.capacity_summary.resolve()
        if not capacity_summary_path.is_file():
            raise FileNotFoundError(f"Capacity summary not found: {capacity_summary_path}")

        config = load_sweep_config_adjacent_to_summary(capacity_summary_path)
        config.setdefault("lr_find_params", dict(_DEFAULT_LR_FIND_PARAMS))

        anchor_config_hash = config.get("anchor_config_hash")
        if not anchor_config_hash:
            raise ValueError(
                "Capacity sweep_config.json is missing anchor_config_hash."
            )
        load_registered_anchor_config(anchor_config_hash)

        runs = load_capacity_sweep_runs(capacity_summary_path)
        if not runs:
            raise ValueError(
                f"No valid (freeze_layers, batch_size) rows in {capacity_summary_path}."
            )

        print(f"Capacity summary: {capacity_summary_path}")
        print(f"Anchor config hash: {anchor_config_hash}")
        print(f"Loaded {len(runs)} LR run(s) from capacity sweep.")
        run_lr_sweep(config, runs)
        return

    load_registered_anchor_config(args.anchor_config_hash)

    config = _default_config(args.anchor_config_hash)
    if config["dataset_hash"] == "latest":
        latest = resolve_latest_dataset_hash()
        if latest:
            config["dataset_hash"] = latest
            print(f"Using latest dataset hash: {latest}")

    runs = [{"freeze_layers": args.freeze_layers, "batch_size": args.batch_size}]
    print(f"Anchor config hash: {args.anchor_config_hash}")
    print(f"Single run: freeze_layers={args.freeze_layers}, batch_size={args.batch_size}")
    run_lr_sweep(config, runs)


if __name__ == "__main__":
    main()
