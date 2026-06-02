"""
RetinaNet training with a single anchor configuration per run.

Run from repo root::

    uv run python src/dl_lib/etalon_object_detection/scripts/object_detection/retinanet_train.py

Prerequisites::

    uv run python src/dl_lib/etalon_object_detection/scripts/anchors/register_anchor_configs.py

Artifact layout::

    artifacts/{dataset_hash}/{split_hash}/{anchor_config_hash}/full_training/{experiment_hash}/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from home_made_od.anchors.anchor_config_registry import (
    DIMENSION_BASED_ANCHOR_CONFIG,
    recipe_to_payload,
)
from dl_lib.etalon_object_detection.modules.ds_utils import (
    DL_LIB_ETALON_FIXED_SIZE,
    WeldingDetectionDataset,
    collate_fn,
    get_path_split_callables,
)
from home_made_od.general.path_layout import (
    anchor_artifacts_root,
    full_training_run_dir,
    images_dir,
    labels_data_dir,
    master_labels_path,
    resolve_dataset_hash_by_version_name,
    resolve_latest_dataset_hash,
    segmentation_weights_path,
)
from home_made_od.retinanet.retinanet_anchors import (
    ensure_anchor_training_spec,
    get_anchor_config_hash,
)
from home_made_od.retinanet.retinanet_detector import (
    build_dl_lab_etalon_retinanet_from_segmentation_weights,
)
from home_made_od.retinanet.retinanet_train import train_retinanet_model

from mypt.code_utils.pytorch_utils import seed_everything

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_ANCHOR_CONFIG_HASH = recipe_to_payload(DIMENSION_BASED_ANCHOR_CONFIG)["config_hash"]


def get_transforms(aug_config: Dict[str, Any], train: bool = True):
    transforms: List[Any] = [v2.ToImage()]
    if train:
        transforms.extend([
            v2.RandomHorizontalFlip(p=aug_config.get("horizontal_flip_p", 0.5)),
            v2.RandomVerticalFlip(p=aug_config.get("vertical_flip_p", 0.5)),
        ])
        rotation_degrees = aug_config.get("rotation_degrees")
        if rotation_degrees is not None:
            transforms.append(
                v2.RandomApply(
                    [v2.RandomRotation(degrees=rotation_degrees)],
                    p=aug_config.get("rotation_p", 0.5),
                )
            )
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    return v2.Compose(transforms)


def get_experiment_hash(config: Dict[str, Any]) -> str:
    """MD5 of all parameters that define this training run."""
    payload = {
        "anchor_config_hash": get_anchor_config_hash(config),
        "dataset_hash": config["dataset_hash"],
        "dataset_version": config.get("dataset_version"),
        "split_params": config["split_params"],
        "train_params": config["train_params"],
        "model_params": config["model_params"],
        "augmentation": config["augmentation"],
        "initial_2_ds_cls_ids": config["initial_2_ds_cls_ids"],
        "seed": config["seed"],
        "force_reoptimize_anchors": config["force_reoptimize_anchors"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def run_experiment(config: Dict[str, Any]) -> None:
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_hash = config["dataset_hash"]
    anchor_config_hash = get_anchor_config_hash(config)
    labels_dir = labels_data_dir(dataset_hash)
    master_json = master_labels_path(dataset_hash)

    split_params = config["split_params"]
    train_filter, val_filter, split_hash = get_path_split_callables(
        master_json_path=str(master_json),
        val_ratio=split_params["val_ratio"],
        seed=split_params["seed"],
        save_dir=str(labels_dir.parent),
    )

    anchor_artifacts_root(dataset_hash, split_hash, anchor_config_hash).mkdir(
        parents=True, exist_ok=True
    )
    experiment_hash = get_experiment_hash(config)
    artifact_dir = full_training_run_dir(
        dataset_hash, split_hash, anchor_config_hash, experiment_hash
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_dataset_for_anchors = WeldingDetectionDataset(
        images_dir=str(images_dir()),
        cache_dir=str(labels_dir),
        target_size=DL_LIB_ETALON_FIXED_SIZE,
        initial_2_ds_cls_ids=config["initial_2_ds_cls_ids"],
        transformations=None,
        path_filter=train_filter,
    )

    anchor_spec = ensure_anchor_training_spec(
        config,
        dataset=train_dataset_for_anchors,
        dataset_hash=dataset_hash,
        split_hash=split_hash,
        force_reoptimize=config["force_reoptimize_anchors"],
    )

    if anchor_spec.config_hash != anchor_config_hash:
        raise ValueError(
            f"anchor_config hash mismatch: config declares {anchor_config_hash}, "
            f"resolved manifest has {anchor_spec.config_hash}."
        )

    run_config = {
        **config,
        "split_hash": split_hash,
        "anchor_config_hash": anchor_spec.config_hash,
        "experiment_hash": experiment_hash,
        "artifact_dir": str(artifact_dir),
        "anchor_training_spec": anchor_spec.to_dict(),
    }
    with open(artifact_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4, ensure_ascii=False)

    print(f"Device: {device}")
    print(f"Dataset hash: {dataset_hash}")
    if config.get("dataset_version"):
        print(f"Dataset version: {config['dataset_version']}")
    print(f"Split hash: {split_hash}")
    print(f"Anchor: {anchor_spec.method} ({anchor_spec.config_hash})")
    print(f"Experiment hash: {experiment_hash}")
    print(f"Artifacts: {artifact_dir}")

    train_dataset = WeldingDetectionDataset(
        images_dir=str(images_dir()),
        cache_dir=str(labels_dir),
        target_size=DL_LIB_ETALON_FIXED_SIZE,
        initial_2_ds_cls_ids=config["initial_2_ds_cls_ids"],
        transformations=get_transforms(config["augmentation"], train=True),
        path_filter=train_filter,
    )
    val_dataset = WeldingDetectionDataset(
        images_dir=str(images_dir()),
        cache_dir=str(labels_dir),
        target_size=DL_LIB_ETALON_FIXED_SIZE,
        initial_2_ds_cls_ids=config["initial_2_ds_cls_ids"],
        transformations=get_transforms(config["augmentation"], train=False),
        path_filter=val_filter,
    )

    train_params = config["train_params"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_params["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_params["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    model_params = config["model_params"]
    model = build_dl_lab_etalon_retinanet_from_segmentation_weights(
        anchor_spec=anchor_spec,
        segmentation_weights_path=str(segmentation_weights_path()),
        img_size=model_params["img_size"],
        device=device,
        freeze_layers=model_params["freeze_layers"],
    )

    train_retinanet_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_params["epochs"],
        learning_rate=train_params["learning_rate"],
        artifact_dir=str(artifact_dir),
        device=device,
        patience=train_params["patience"],
        cls_id_2_cls_name=config["cls_id_2_cls_name"],
        anchor_spec=anchor_spec,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Etalon RetinaNet on one anchor config.")
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


def resolve_dataset_hash(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Resolve ``config['dataset_hash']`` from CLI flags (mutates *config* in place)."""
    if args.dataset_hash:
        config["dataset_hash"] = args.dataset_hash
        config.pop("dataset_version", None)
        print(f"Using dataset hash: {args.dataset_hash}")
        return

    if args.dataset_version:
        config["dataset_version"] = args.dataset_version
        config["dataset_hash"] = resolve_dataset_hash_by_version_name(args.dataset_version)
        print(f"Using dataset version {args.dataset_version!r} -> {config['dataset_hash']}")
        return

    if config.get("dataset_version"):
        version_name = config["dataset_version"]
        config["dataset_hash"] = resolve_dataset_hash_by_version_name(version_name)
        print(f"Using dataset version {version_name!r} -> {config['dataset_hash']}")
        return

    if config["dataset_hash"] == "latest":
        latest = resolve_latest_dataset_hash()
        if latest:
            config["dataset_hash"] = latest
            print(f"Using latest dataset hash: {latest}")


def main() -> None:
    args = parse_args()
    config: Dict[str, Any] = {
        "dataset_hash": "latest",
        "dataset_version": "full_dataset",
        "split_params": {
            "val_ratio": 0.15,
            "seed": 42,
        },
        "train_params": {
            "learning_rate": 1e-3,
            "epochs": 200,
            "batch_size": 4,
            "patience": 30,
        },
        "model_params": {
            "freeze_layers": 2,
            "img_size": DL_LIB_ETALON_FIXED_SIZE,
        },
        "anchor_config_hash": _DEFAULT_ANCHOR_CONFIG_HASH,
        "force_reoptimize_anchors": True,
        "augmentation": {
            "horizontal_flip_p": 0.5,
            "vertical_flip_p": 0.5,
            "rotation_degrees": 30,
            "rotation_p": 0.5,
        },
        "initial_2_ds_cls_ids": {0: 1},
        "cls_id_2_cls_name": {1: "wired", 0: "background"},
        "seed": 42,
    }

    resolve_dataset_hash(config, args)
    run_experiment(config)


if __name__ == "__main__":
    main()
