"""
Shared RetinaNet training pipeline for road-sign datasets (resized / patch).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from torchvision.transforms import v2

from home_made_od.anchors.anchor_config_registry import (
    DIMENSION_BASED_ANCHOR_CONFIG,
    recipe_to_payload,
)
from home_made_od.general.path_utils import (
    DATASET_VERSION_PATCH,
    EXPERIMENT_CONFIG_FILENAME,
    RoadSignDatasetVersion,
    canonical_json_hash,
    load_dataset_config,
    model_experiment_dir,
    model_retinanet_anchor_config_path,
)
from home_made_od.retinanet.retinanet_anchors import compute_and_save_retinanet_anchors
from home_made_od.retinanet.retinanet_train import train_retinanet_model
from mypt.code_utils.pytorch_utils import seed_everything
from road_sign.utils.data_utils import (
    build_retinanet_class_id_map,
    build_retinanet_dataloaders,
    collect_train_box_dimensions_for_anchors,
    load_road_sign_class_mapping,
    retinanet_cls_id_to_name,
    retinanet_num_classes,
    validate_retinanet_class_id_map,
)

from script_utils import (
    RETINANET_MODEL_NAME,
    anchor_config_summary_payload,
    build_retinanet_from_spec,
    load_optional_patch_config,
    read_anchor_level_metadata,
    resolve_dataset_and_split,
    resolve_target_size,
)

logger = logging.getLogger(__name__)

_ANCHOR_RECIPE = recipe_to_payload(DIMENSION_BASED_ANCHOR_CONFIG)
DEFAULT_ANCHOR_CONFIG_HASH = _ANCHOR_RECIPE["config_hash"]
DEFAULT_ANCHOR_METHOD = _ANCHOR_RECIPE["method"]
DEFAULT_ANCHOR_METHOD_PARAMETERS = _ANCHOR_RECIPE["method_parameters"]

DEFAULT_TRAIN_AUGMENTATION: Dict[str, Any] = {
    "horizontal_flip_p": 0.5,
    "vertical_flip_p": 0.5,
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
    },
    "grayscale_p": 0.1,
    "rotation_degrees": 30,
    "rotation_p": 0.5,
}

DEFAULT_TRAIN_PARAMS: Dict[str, Any] = {
    "target_size": (512, 512),
    "batch_size": 32,
    "epochs": 200,
    "learning_rate": 1e-4,
    "early_stop_patience": 25,
    "num_workers": 2,
}


def build_retinanet_transforms(aug_config: Dict[str, Any], *, train: bool) -> v2.Compose:
    transforms: list[Any] = [v2.ToImage()]
    if train:
        horizontal_flip_p = aug_config.get("horizontal_flip_p")
        if horizontal_flip_p is not None:
            transforms.append(v2.RandomHorizontalFlip(p=float(horizontal_flip_p)))

        vertical_flip_p = aug_config.get("vertical_flip_p")
        if vertical_flip_p is not None:
            transforms.append(v2.RandomVerticalFlip(p=float(vertical_flip_p)))

        color_jitter = aug_config.get("color_jitter")
        if color_jitter:
            transforms.append(
                v2.ColorJitter(
                    brightness=color_jitter.get("brightness", 0.0),
                    contrast=color_jitter.get("contrast", 0.0),
                )
            )

        grayscale_p = aug_config.get("grayscale_p")
        if grayscale_p is not None:
            transforms.append(v2.RandomGrayscale(p=float(grayscale_p)))

        rotation_degrees = aug_config.get("rotation_degrees")
        if rotation_degrees is not None:
            transforms.append(
                v2.RandomApply(
                    [v2.RandomRotation(degrees=float(rotation_degrees))],
                    p=aug_config.get("rotation_p", 0.5),
                )
            )
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    return v2.Compose(transforms)


def normalize_class_id_map_config(raw: Dict[Any, Any]) -> Dict[int, int]:
    """Parse ``class_id_map`` from a JSON/YAML config (keys may be strings)."""
    return validate_retinanet_class_id_map({int(k): int(v) for k, v in raw.items()})


def resolve_class_id_map(
    config: Dict[str, Any],
    class_mapping: Dict[int, str],
) -> Dict[int, int]:
    """
    Resolve dataset → model label mapping from the training config.

    If ``class_id_map`` is ``None``, builds a contiguous map from ``class_mapping``
    using ``retinanet_label_start`` (default 1).
    """
    raw = config.get("class_id_map")
    if raw is None:
        start_index = int(config.get("retinanet_label_start", 1))
        return build_retinanet_class_id_map(class_mapping, start_index=start_index)
    return normalize_class_id_map_config(raw)


def compute_experiment_hash(config: Dict[str, Any]) -> str:
    payload = {
        "model_name": RETINANET_MODEL_NAME,
        "dataset_version": config["dataset_version"],
        "dataset_hash": config["dataset_hash"],
        "split_hash": config["split_hash"],
        "anchor_config_hash": config["anchor_config_hash"],
        "split_params": config["split_params"],
        "train_params": config["train_params"],
        "model_params": config["model_params"],
        "augmentation": config["augmentation"],
        "class_id_map": config.get("class_id_map"),
        "retinanet_label_start": config.get("retinanet_label_start", 1),
        "seed": config["seed"],
        "force_recompute_anchors": config.get("force_recompute_anchors", True),
    }
    if config.get("patch_config") is not None:
        payload["patch_config"] = config["patch_config"]
    return canonical_json_hash(payload)


def run_retinanet_training(
    version: RoadSignDatasetVersion,
    config: Dict[str, Any],
) -> Path:
    """
    End-to-end training: split → anchors → dataloaders → model → ``train_retinanet_model``.
    """
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_hash, split_hash = resolve_dataset_and_split(version, config)
    config["dataset_hash"] = dataset_hash
    config["split_hash"] = split_hash
    config.setdefault("anchor_config_hash", DEFAULT_ANCHOR_CONFIG_HASH)

    if version == DATASET_VERSION_PATCH:
        config["patch_config"] = load_optional_patch_config(dataset_hash)

    target_size = resolve_target_size(version, dataset_hash, train_config=config)
    config["train_params"]["target_size"] = list(target_size)

    box_dimensions = collect_train_box_dimensions_for_anchors(
        version, dataset_hash, split_hash, target_size
    )
    if not box_dimensions:
        raise ValueError(
            f"No training boxes found for {version}/{dataset_hash}/{split_hash}. "
            "Check labels and split files."
        )

    anchor_config_path = model_retinanet_anchor_config_path(
        RETINANET_MODEL_NAME, version, dataset_hash, split_hash
    )
    anchor_spec = compute_and_save_retinanet_anchors(
        box_dimensions,
        anchor_config_path,
        method=config.get("anchor_method", DEFAULT_ANCHOR_METHOD),
        method_parameters=config.get(
            "anchor_method_parameters", DEFAULT_ANCHOR_METHOD_PARAMETERS
        ),
        force=config.get("force_recompute_anchors", True),
    )

    class_mapping = load_road_sign_class_mapping(version, dataset_hash)
    class_id_map = resolve_class_id_map(config, class_mapping)
    config["class_id_map"] = class_id_map

    train_params = config["train_params"]
    aug_config = config["augmentation"]

    train_loader, val_loader, meta = build_retinanet_dataloaders(
        version=version,
        dataset_hash=dataset_hash,
        split_hash=split_hash,
        target_size=target_size,
        class_id_map=class_id_map,
        batch_size=train_params["batch_size"],
        train_transforms=build_retinanet_transforms(aug_config, train=True),
        val_transforms=build_retinanet_transforms(aug_config, train=False),
        num_workers=train_params.get("num_workers", 0),
    )

    num_classes = retinanet_num_classes(class_id_map)
    cls_id_2_cls_name = retinanet_cls_id_to_name(class_mapping, class_id_map)

    experiment_hash = compute_experiment_hash(config)
    artifact_dir = model_experiment_dir(
        RETINANET_MODEL_NAME,
        version,
        dataset_hash,
        split_hash,
        experiment_hash,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run_record = {
        **config,
        "experiment_hash": experiment_hash,
        "artifact_dir": str(artifact_dir),
        "anchor_config_path": str(anchor_spec.config_path),
        "anchor_training_spec": anchor_spec.to_dict(),
        "anchor_config": anchor_config_summary_payload(anchor_spec),
        "class_id_map": class_id_map,
        "num_foreground_classes": meta["num_foreground_classes"],
        "num_classes": num_classes,
        "train_size": meta["train_size"],
        "val_size": meta["val_size"],
    }
    with open(artifact_dir / EXPERIMENT_CONFIG_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(run_record, handle, indent=4, ensure_ascii=False)

    model_params = config["model_params"]
    model = build_retinanet_from_spec(
        anchor_spec,
        num_classes=num_classes,
        img_size=target_size,
        device=device,
        checkpoint_path=config.get("checkpoint_path"),
        freeze_backbone_layers=model_params.get("freeze_backbone_layers", 2),
    )

    anchor_meta = read_anchor_level_metadata(anchor_spec)
    logger.info("Device: %s", device)
    logger.info("Dataset: %s / %s", version, dataset_hash)
    logger.info("Split: %s | train=%d val=%d", split_hash, meta["train_size"], meta["val_size"])
    logger.info("Anchors: %s (%s)", anchor_spec.config_path, anchor_spec.config_hash)
    logger.info(
        "FPN levels: requested=%s used=%s added=%s",
        anchor_meta["requested_fpn_levels"],
        anchor_meta["used_fpn_levels"],
        anchor_meta["added_fpn_levels"],
    )
    logger.info("Experiment: %s -> %s", experiment_hash, artifact_dir)

    train_retinanet_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_params["epochs"],
        learning_rate=train_params["learning_rate"],
        artifact_dir=str(artifact_dir),
        device=device,
        cls_id_2_cls_name=cls_id_2_cls_name,
        patience=train_params.get(
            "early_stop_patience", DEFAULT_TRAIN_PARAMS["early_stop_patience"]
        ),
        anchor_spec=anchor_spec,
    )
    return artifact_dir

