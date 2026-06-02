"""
Fast capacity sweeps over ``freeze_layers`` on a training subset.

Probes GPU batch size **once** on the full training split (using the lowest
``freeze_layers`` candidate — most memory-hungry), then runs each overfitting
comparison at that shared batch size on the fixed subset.

Run from repo root::

    uv run python src/dl_lib/etalon_object_detection/scripts/anchors/register_anchor_configs.py
    uv run python src/dl_lib/etalon_object_detection/scripts/object_detection/parameter_sweeps/model_params_sweep.py

Optional: pass a registered anchor config hash (default: dimension_based recipe)::

    uv run python .../model_params_sweep.py 0fdeb8457bec592333f887d9f563a360

Artifact layout::

    artifacts/{dataset_hash}/{split_hash}/{anchor_config_hash}/sweeps/capacity/{subset_hash}/
        batch_size_summary.json
        frozen_layers_{n}/
            run_config.json
            metrics.jsonl
            metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

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
    resolve_dataset_hash_by_version_name,
    segmentation_weights_path,
    sweeps_capacity_run_dir,
    sweeps_capacity_subset_dir,
)
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_anchors import (
    ensure_anchor_training_spec,
    get_anchor_config_hash,
)
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_detector import (
    build_dl_lab_etalon_retinanet_from_segmentation_weights,
)
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_train import train_single_epoch
from dl_lib.etalon_object_detection.scripts.object_detection.parameter_sweeps.sweep_utils import (
    compute_subset_hash,
    find_max_batch_size,
    get_no_aug_transforms,
    select_subset_paths_by_data_batch,
)
from mypt.code_utils.pytorch_utils import seed_everything

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_ANCHOR_CONFIG_HASH = recipe_to_payload(DIMENSION_BASED_ANCHOR_CONFIG)["config_hash"]

_TRAIN_LOSS_FIELDS = (
    ("epoch_loss", "train_loss"),
    ("loss_classifier", "train_cls_loss"),
    ("loss_box_reg", "train_reg_loss"),
)


def summarize_capacity_metrics(epoch_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Final and best (minimum) values for each training loss component."""
    if not epoch_records:
        return {}

    final = epoch_records[-1]["train"]
    summary: Dict[str, Any] = {
        "final_epoch": epoch_records[-1]["epoch"],
        "final_train_loss": final.get("epoch_loss"),
        "final_train_cls_loss": final.get("loss_classifier"),
        "final_train_reg_loss": final.get("loss_box_reg"),
    }

    for metric_key, prefix in _TRAIN_LOSS_FIELDS:
        best_record = min(
            epoch_records,
            key=lambda record: record["train"].get(metric_key, float("inf")),
        )
        summary[f"best_{prefix}"] = best_record["train"].get(metric_key)
        summary[f"best_{prefix}_epoch"] = best_record["epoch"]

    return summary


def run_capacity_training(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    metrics_path: Path,
) -> List[Dict[str, Any]]:
    """Train-only loop; append one JSON line per epoch to ``metrics_path``."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    epoch_records: List[Dict[str, Any]] = []
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        for epoch in range(epochs):
            train_metrics = train_single_epoch(
                model,
                train_loader,
                optimizer,
                device,
                scheduler=None,
            )
            record = {"epoch": epoch, "train": train_metrics}
            epoch_records.append(record)
            metrics_file.write(json.dumps(record) + "\n")

            print(
                f"Epoch {epoch + 1:03d}/{epochs:03d} | "
                f"T-Loss: {train_metrics['epoch_loss']:.4f} | "
                f"T-Cls: {train_metrics['loss_classifier']:.4f} | "
                f"T-Reg: {train_metrics['loss_box_reg']:.4f}"
            )

    return epoch_records


def run_single_freeze_layers(
    *,
    freeze_layers: int,
    batch_size: int,
    config: Dict[str, Any],
    dataset_hash: str,
    split_hash: str,
    subset_hash: str,
    subset_filter: Callable[[str], bool],
    anchor_spec: Any,
    device: torch.device,
) -> Dict[str, Any]:
    run_dir = sweeps_capacity_run_dir(
        dataset_hash,
        split_hash,
        anchor_spec.config_hash,
        subset_hash,
        freeze_layers,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "subset_hash": subset_hash,
        "freeze_layers": freeze_layers,
        "batch_size": batch_size,
        "anchor_config_hash": anchor_spec.config_hash,
        "subset_params": config["subset_params"],
        "train_params": config["train_params"],
        "model_params": config["model_params"],
        "initial_2_ds_cls_ids": config["initial_2_ds_cls_ids"],
        "seed": config["seed"],
        "artifact_dir": str(run_dir),
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4, ensure_ascii=False)

    seed_everything(config["seed"])
    labels_dir = labels_data_dir(dataset_hash)

    train_dataset = WeldingDetectionDataset(
        images_dir=str(images_dir()),
        cache_dir=str(labels_dir),
        target_size=DL_LIB_ETALON_FIXED_SIZE,
        initial_2_ds_cls_ids=config["initial_2_ds_cls_ids"],
        transformations=get_no_aug_transforms(),
        path_filter=subset_filter,
    )

    model_params = config["model_params"]
    model = build_dl_lab_etalon_retinanet_from_segmentation_weights(
        anchor_spec=anchor_spec,
        segmentation_weights_path=str(segmentation_weights_path()),
        img_size=model_params["img_size"],
        device=device,
        freeze_layers=freeze_layers,
    )

    train_params = config["train_params"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    logger.info(
        "Capacity run: freeze_layers=%d, batch_size=%d, subset=%d samples, artifacts=%s",
        freeze_layers,
        batch_size,
        len(train_dataset),
        run_dir,
    )

    epoch_records = run_capacity_training(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=train_params["epochs"],
        learning_rate=train_params["learning_rate"],
        metrics_path=run_dir / "metrics.jsonl",
    )

    metrics_summary = summarize_capacity_metrics(epoch_records)
    with open(run_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4, ensure_ascii=False)

    return {
        "freeze_layers": freeze_layers,
        "artifact_dir": str(run_dir),
        "num_train_samples": len(train_dataset),
        "batch_size": batch_size,
        "max_batch_size": batch_size,
        "capacity_skipped": False,
        **metrics_summary,
    }


def probe_shared_batch_size(
    *,
    config: Dict[str, Any],
    train_filter: Callable[[str], bool],
    anchor_spec: Any,
    device: torch.device,
    dataset_hash: str,
) -> tuple[int, Dict[str, Any]]:
    """
    Probe the largest GPU batch size once on the full training split.

    Uses ``min(freeze_layers_candidates)`` (most trainable layers / highest memory).
    """
    labels_dir = labels_data_dir(dataset_hash)
    probe_freeze_layers = min(config["freeze_layers_candidates"])
    model_params = config["model_params"]
    bs_params = config["batch_size_finder_params"]

    probe_dataset = WeldingDetectionDataset(
        images_dir=str(images_dir()),
        cache_dir=str(labels_dir),
        target_size=DL_LIB_ETALON_FIXED_SIZE,
        initial_2_ds_cls_ids=config["initial_2_ds_cls_ids"],
        transformations=get_no_aug_transforms(),
        path_filter=train_filter,
    )

    def build_probe_model() -> torch.nn.Module:
        return build_dl_lab_etalon_retinanet_from_segmentation_weights(
            anchor_spec=anchor_spec,
            segmentation_weights_path=str(segmentation_weights_path()),
            img_size=model_params["img_size"],
            device=device,
            freeze_layers=probe_freeze_layers,
        )

    seed_everything(config["seed"])
    max_batch_size, probe_log = find_max_batch_size(
        build_model=build_probe_model,
        dataset=probe_dataset,
        device=device,
        min_batch_size=bs_params["min_batch_size"],
        max_batch_size=bs_params["max_batch_size"],
    )

    summary: Dict[str, Any] = {
        "shared_across_freeze_layers": True,
        "probe_freeze_layers": probe_freeze_layers,
        "probe_num_train_samples": len(probe_dataset),
        "max_batch_size": max_batch_size,
        "probe_log": probe_log,
        "min_batch_size": bs_params["min_batch_size"],
        "max_batch_size_limit": bs_params["max_batch_size"],
    }
    return max_batch_size, summary


def run_model_params_sweep(config: Dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_hash = config["dataset_hash"]
    labels_dir = labels_data_dir(dataset_hash)
    master_json = master_labels_path(dataset_hash)

    split_params = config["split_params"]
    train_filter, _, split_hash = get_path_split_callables(
        master_json_path=str(master_json),
        val_ratio=split_params["val_ratio"],
        seed=split_params["seed"],
        save_dir=str(labels_dir.parent),
    )

    with open(master_json, "r", encoding="utf-8") as f:
        all_samples = json.load(f).get("samples", [])
    train_paths = sorted(
        s["img_path"] for s in all_samples if train_filter(s["img_path"])
    )

    subset_params = config["subset_params"]
    subset_hash = compute_subset_hash(subset_params, split_hash)
    subset_paths, subset_metadata = select_subset_paths_by_data_batch(
        train_paths,
        max_samples_per_batch=subset_params["max_samples"],
        seed=subset_params["seed"],
    )
    subset_set = set(subset_paths)
    subset_filter: Callable[[str], bool] = lambda p: p in subset_set

    anchor_config_hash = get_anchor_config_hash(config)
    anchor_artifacts_root(dataset_hash, split_hash, anchor_config_hash).mkdir(
        parents=True, exist_ok=True
    )

    subset_dir = sweeps_capacity_subset_dir(
        dataset_hash, split_hash, anchor_config_hash, subset_hash
    )
    subset_dir.mkdir(parents=True, exist_ok=True)

    with open(subset_dir / "subset_files.json", "w", encoding="utf-8") as f:
        json.dump(subset_metadata, f, indent=4, ensure_ascii=False)

    sweep_config = {
        **config,
        "split_hash": split_hash,
        "anchor_config_hash": anchor_config_hash,
        "subset_hash": subset_hash,
        "subset_sampling_strategy": subset_metadata["sampling_strategy"],
        "max_samples_per_batch": subset_params["max_samples"],
        "num_subset_samples": len(subset_paths),
        "num_data_batches": subset_metadata["num_data_batches"],
        "subset_dir": str(subset_dir),
    }
    with open(subset_dir / "sweep_config.json", "w", encoding="utf-8") as f:
        json.dump(sweep_config, f, indent=4, ensure_ascii=False)

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

    print(f"Device: {device}")
    print(f"Dataset hash: {dataset_hash}")
    print(f"Split hash: {split_hash}")
    print(
        f"Subset hash: {subset_hash} "
        f"({len(subset_paths)} samples = "
        f"{subset_params['max_samples']} per data batch × "
        f"{subset_metadata['num_data_batches']} batches)"
    )
    for batch_name, details in subset_metadata["data_batches"].items():
        print(
            f"  {batch_name}: selected {details['num_selected']}/{details['num_train_frames']} "
            f"frames from {details['num_dcm_stems']} DCM stems"
        )
    print(f"Anchor: {anchor_spec.method} ({anchor_spec.config_hash})")
    print(f"Sweep artifacts: {subset_dir}")

    shared_batch_size, batch_size_summary = probe_shared_batch_size(
        config=config,
        train_filter=train_filter,
        anchor_spec=anchor_spec,
        device=device,
        dataset_hash=dataset_hash,
    )
    with open(subset_dir / "batch_size_summary.json", "w", encoding="utf-8") as f:
        json.dump(batch_size_summary, f, indent=4, ensure_ascii=False)

    print(
        f"Shared batch size: {shared_batch_size} "
        f"(probed with freeze_layers={batch_size_summary['probe_freeze_layers']}, "
        f"train_samples={batch_size_summary['probe_num_train_samples']})"
    )

    summaries: List[Dict[str, Any]] = []
    if shared_batch_size <= 0:
        logger.error(
            "Batch size probe failed in [%d, %d]; skipping all freeze_layers runs.",
            batch_size_summary["min_batch_size"],
            batch_size_summary["max_batch_size_limit"],
        )
        for freeze_layers in config["freeze_layers_candidates"]:
            summaries.append(
                {
                    "freeze_layers": freeze_layers,
                    "batch_size": shared_batch_size,
                    "max_batch_size": shared_batch_size,
                    "capacity_skipped": True,
                    "reason": "batch_size_probe_failed",
                }
            )
    else:
        for freeze_layers in config["freeze_layers_candidates"]:
            summary = run_single_freeze_layers(
                freeze_layers=freeze_layers,
                batch_size=shared_batch_size,
                config=config,
                dataset_hash=dataset_hash,
                split_hash=split_hash,
                subset_hash=subset_hash,
                subset_filter=subset_filter,
                anchor_spec=anchor_spec,
                device=device,
            )
            summaries.append(summary)

    with open(subset_dir / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=4, ensure_ascii=False)

    print("\nCapacity sweep complete:")
    for row in summaries:
        if row.get("capacity_skipped"):
            print(
                f"  freeze_layers={row['freeze_layers']}: "
                f"skipped ({row.get('reason', 'unknown')})"
            )
            continue
        final_loss = row.get("final_train_loss")
        best_loss = row.get("best_train_loss")
        final_str = f"{final_loss:.4f}" if final_loss is not None else "n/a"
        best_str = f"{best_loss:.4f}" if best_loss is not None else "n/a"
        print(
            f"  freeze_layers={row['freeze_layers']}: "
            f"batch_size={row.get('max_batch_size')} "
            f"T-Loss final={final_str} best={best_str} "
            f"({row['artifact_dir']})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capacity sweep over freeze_layers on a training subset.",
    )
    parser.add_argument(
        "anchor_config_hash",
        nargs="?",
        default=_DEFAULT_ANCHOR_CONFIG_HASH,
        help=(
            "Registered anchor config hash "
            f"(default: {_DEFAULT_ANCHOR_CONFIG_HASH}, dimension_based)."
        ),
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help=(
            "Dataset version name from labeling/cache/*/dataset_config.json "
            "(e.g. full_dataset, batch_large_diameters, small_diameters)."
        ),
    )
    parser.add_argument(
        "--dataset-hash",
        type=str,
        default=None,
        help="Explicit dataset hash (overrides --dataset-version and 'latest').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_registered_anchor_config(args.anchor_config_hash)

    config: Dict[str, Any] = {
        "dataset_hash": "latest",
        "split_params": {
            "val_ratio": 0.15,
            "seed": 42,
        },
        "subset_params": {
            # Quota per data batch folder (batch_large_diameters, etc.), not global.
            "max_samples": 10,
            "seed": 42,
        },
        "train_params": {
            "learning_rate": 1e-3,
            "epochs": 200,
        },
        "batch_size_finder_params": {
            "min_batch_size": 1,
            "max_batch_size": 64,
        },
        "freeze_layers_candidates": [0, 1, 2, 3, 4],
        "model_params": {
            "img_size": DL_LIB_ETALON_FIXED_SIZE,
        },
        "anchor_config_hash": args.anchor_config_hash,
        "force_reoptimize_anchors": True,
        "initial_2_ds_cls_ids": {0: 1},
        "seed": 42,
    }

    # args.dataset_version = "small_diameteres"

    if args.dataset_hash:
        config["dataset_hash"] = args.dataset_hash
        print(f"Using dataset hash: {args.dataset_hash}")
    elif args.dataset_version:
        config["dataset_hash"] = resolve_dataset_hash_by_version_name(args.dataset_version)
        print(f"Using dataset version {args.dataset_version!r} -> {config['dataset_hash']}")
    elif config["dataset_hash"] == "latest":
        latest = resolve_latest_dataset_hash()
        if latest:
            config["dataset_hash"] = latest
            print(f"Using latest dataset hash: {latest}")

    print(f"Anchor config hash: {args.anchor_config_hash}")
    run_model_params_sweep(config)


if __name__ == "__main__":
    main()
