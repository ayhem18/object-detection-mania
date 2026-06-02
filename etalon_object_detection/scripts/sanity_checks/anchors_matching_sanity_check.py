"""
Pre-training anchor evaluation (Tests 1 + 2).

Run from the repo root::

    uv run python src/dl_lib/etalon_object_detection/scripts/anchors/register_anchor_configs.py
    uv run python src/dl_lib/etalon_object_detection/scripts/sanity_checks/anchors_matching_sanity_check.py

Writes split-dependent anchors under::

    artifacts/{dataset_hash}/{split_hash}/anchors_data/{config_hash}/
        anchor_config.json
        master_labels_enriched.json
        summary.json
        cell_recall/  (optional viz)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from mypt.code_utils.pytorch_utils import seed_everything

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

from dl_lib.etalon_object_detection.modules.path_layout import (
    images_dir,
    labels_data_dir,
    master_labels_path,
    resolve_latest_dataset_hash,
    segmentation_weights_path,
    split_artifacts_root,
)
from dl_lib.etalon_object_detection.modules.anchors.anchor_computation_strategies import (
    anchor_config_output_dir,
)
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_anchors import (
    optimize_retinanet_anchors,
)
from dl_lib.etalon_object_detection.modules.anchors.anchor_config_registry import (
    REGISTERED_ANCHOR_CONFIGS,
    load_registered_anchor_config,
    recipe_to_payload,
)
from dl_lib.etalon_object_detection.modules.anchors.anchor_eval_utils import (
    FlatCandidateBatch,
    flatten_candidate_boxes,
)
from dl_lib.etalon_object_detection.modules.anchors.anchor_evaluation import (
    evaluate_anchors,
    format_anchor_evaluation_report,
)
from dl_lib.etalon_object_detection.modules.ds_utils import (
    DL_LIB_ETALON_FIXED_SIZE,
    WeldingDetectionDataset,
    get_path_split_callables,
)
from dl_lib.etalon_object_detection.modules.retinanet.error_analysis.retinanet_grid_index import (
    build_retinanet_grid_layout,
)
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_anchors import (
    anchor_spec_from_manifest_path,
    build_anchor_generator_from_manifest,
    get_anchor_generator_config,
)
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_detector import (
    build_dl_lab_etalon_retinanet_from_segmentation_weights,
)

DEFAULT_ANCHOR_CONFIG_HASHES: List[str] = [
    recipe_to_payload(recipe)["config_hash"]
    for recipe in REGISTERED_ANCHOR_CONFIGS.values()
]


def extract_model_anchors_and_grid(
    anchor_config_path: str,
    weights_path: str,
    img_size: Tuple[int, int],
) -> Tuple[torch.nn.Module, torch.Tensor, list, list]:
    """Build RetinaNet; return model, flat anchors, prior_to_cell, level_grids."""
    metadata = get_anchor_generator_config(anchor_config_path)
    used_levels = list(metadata["used_fpn_levels"])
    assignment_method = metadata["assignment_method"]

    logger.info(
        "Building RetinaNet (%s): levels=%s from %s",
        assignment_method,
        used_levels,
        anchor_config_path,
    )
    anchor_spec = anchor_spec_from_manifest_path(anchor_config_path)
    anchor_generator = build_anchor_generator_from_manifest(anchor_config_path)

    device = torch.device("cpu")
    model = build_dl_lab_etalon_retinanet_from_segmentation_weights(
        anchor_spec=anchor_spec,
        segmentation_weights_path=weights_path,
        img_size=img_size,
        device=device,
        anchor_generator=anchor_generator,
    )
    model.eval()

    raw_image = torch.zeros((3, img_size[0], img_size[1]), dtype=torch.float32)
    level_ids = used_levels

    with torch.no_grad():
        images, _ = model.transform([raw_image], None)
        if images.tensors.shape[-2:] != img_size:
            raise ValueError(
                f"Image size mismatch: {images.tensors.shape[-2:]} != {img_size}"
            )
        img_h, img_w = images.tensors.shape[-2:]
        features = list(model.backbone(images.tensors).values())
        if len(features) != len(used_levels):
            if len(features) < len(used_levels):
                raise ValueError(
                    f"Backbone produced {len(features)} feature maps but "
                    f"anchor config has {len(used_levels)} levels {used_levels}."
                )
            logger.warning(
                "Truncating %d backbone feature maps to %d manifest levels %s",
                len(features),
                len(used_levels),
                used_levels,
            )
            features = features[: len(used_levels)]
        anchors_list = model.anchor_generator(images, features)
        prior_to_cell, level_grids = build_retinanet_grid_layout(
            model, img_h, img_w, features, level_ids=level_ids
        )

    n_anchors = anchors_list[0].shape[0]
    logger.info(
        "Model ready: %d priors, %d FPN grid levels",
        n_anchors,
        len(level_grids),
    )
    return model, anchors_list[0], prior_to_cell, level_grids


def load_flat_candidates(
    samples: List[Dict[str, Any]],
    cache_dir: Path,
) -> Tuple[FlatCandidateBatch, List[str]]:
    """Load all GT boxes into one flat batch for batched anchor evaluation."""
    boxes_per_image: List[torch.Tensor] = []
    image_stems: List[str] = []
    assigned_levels: List[str] = []

    for sample in tqdm(samples, desc="Loading GT boxes", unit="sample"):
        lbl_path = cache_dir / sample["lbl_path"]
        if not lbl_path.exists():
            continue

        with open(lbl_path, "r", encoding="utf-8") as f:
            lbl_data = json.load(f)

        boxes = torch.tensor(lbl_data["boxes"], dtype=torch.float32)
        if boxes.numel() == 0:
            continue

        boxes_per_image.append(boxes)
        image_stems.append(Path(sample["img_path"]).stem)
        assigned_levels.extend(
            e["assigned_level"] for e in sample["enriched_etalons"]
        )

    candidates = flatten_candidate_boxes(boxes_per_image, image_stems=image_stems)
    if len(assigned_levels) != candidates.num_boxes:
        raise ValueError(
            f"assigned_levels ({len(assigned_levels)}) != "
            f"candidate boxes ({candidates.num_boxes})"
        )

    logger.info(
        "Loaded %d images, %d GT boxes (flat batch)",
        candidates.num_images,
        candidates.num_boxes,
    )
    return candidates, assigned_levels


def run_single_anchor_config(
    *,
    config_hash: str,
    train_dataset: WeldingDetectionDataset,
    dataset_hash: str,
    labels_dir: Path,
    config: Dict[str, Any],
    split_hash: str,
) -> None:
    """Optimize anchors + run Tests 1 & 2 for one registered anchor config."""
    recipe = load_registered_anchor_config(config_hash)
    method = recipe["method"]
    method_parameters = recipe["method_parameters"]
    img_size = config["img_size"]
    weights_path = str(segmentation_weights_path())

    fg_iou_thresh = config.get("fg_iou_thresh", 0.5)
    bbox_level_iou_threshold = config.get(
        "bbox_level_iou_threshold", config.get("grid_iou_threshold", fg_iou_thresh)
    )
    background_iou_threshold = config.get("background_iou_threshold", 0.4)
    metric_threshold = config.get("metric_threshold", 0.5)
    min_gt_threshold_ratio = config.get("min_gt_threshold_ratio", 0.95)
    run_test1 = config.get("run_detector_matching", True)
    run_test2 = config.get("run_cell_recall", True)

    config_dir = anchor_config_output_dir(dataset_hash, split_hash, config_hash)
    config_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Optimizing anchors: method=%s, config_hash=%s, output=%s",
        method,
        config_hash,
        config_dir,
    )
    enriched_json_path, anchor_config_path = optimize_retinanet_anchors(
        dataset=train_dataset,
        output_dir=config_dir,
        dataset_hash=dataset_hash,
        method=method,
        method_parameters=method_parameters,
    )

    model, all_anchors, prior_to_cell, level_grids = extract_model_anchors_and_grid(
        str(anchor_config_path), weights_path, img_size
    )

    with open(enriched_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    if not samples:
        logger.warning("No training samples for config %s", config_hash)
        return

    logger.info("Running Tests 1 & 2 for %s", method)
    candidates, assigned_levels = load_flat_candidates(samples, labels_dir)

    print("\n" + "=" * 60)
    print(f"Anchor config: {method} ({config_hash})")
    print("=" * 60)
    print(f"Dataset Hash:  {dataset_hash}")
    print(f"Split Hash:    {split_hash}")
    print(f"Train Samples: {len(samples)}")
    print(f"Config Dir:    {config_dir}")
    print(f"Image Size:    {img_size} | Anchors: {all_anchors.shape[0]}")
    print(
        f"Test 2 thresholds: recall>={metric_threshold}, "
        f"gt_pass_fraction>={min_gt_threshold_ratio}, "
        f"fg_cell>={bbox_level_iou_threshold}, bg_cell<{background_iou_threshold}"
    )

    eval_report = evaluate_anchors(
        reference_bboxes=all_anchors,
        reference_assignments=prior_to_cell,
        grids=level_grids,
        candidates=candidates,
        assigned_levels=assigned_levels,
        model=model,
        fg_iou_threshold=fg_iou_thresh,
        bbox_level_iou_threshold=bbox_level_iou_threshold,
        background_iou_threshold=background_iou_threshold,
        metric_threshold=metric_threshold,
        min_gt_threshold_ratio=min_gt_threshold_ratio,
        run_detector_matching=run_test1,
        run_cell_recall=run_test2,
        output_dir=config_dir if config.get("save_visualizations", True) else None,
    )

    print("\n" + format_anchor_evaluation_report(eval_report))

    if eval_report.output_dir is not None:
        summary_path = config_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(eval_report.to_dict(), f, indent=2)
        logger.info("Wrote summary: %s", summary_path)


def run_anchor_evaluation(config: Dict[str, Any]) -> None:
    """Run all registered anchor configs on the train split."""
    seed_everything(config.get("seed", 42))

    dataset_hash = config["dataset_hash"]
    master_json = master_labels_path(dataset_hash)
    if not master_json.is_file():
        logger.error("Master labels not found: %s", master_json)
        return

    labels_dir = labels_data_dir(dataset_hash)
    split_params = config["split_params"]
    anchor_config_hashes = config.get("anchor_config_hashes", DEFAULT_ANCHOR_CONFIG_HASHES)

    train_filter, _, split_hash = get_path_split_callables(
        master_json_path=str(master_json),
        val_ratio=split_params["val_ratio"],
        seed=split_params["seed"],
        save_dir=str(labels_dir.parent),
    )

    split_artifacts_root(dataset_hash, split_hash).mkdir(parents=True, exist_ok=True)

    train_dataset = WeldingDetectionDataset(
        images_dir=str(images_dir()),
        cache_dir=str(labels_dir),
        target_size=DL_LIB_ETALON_FIXED_SIZE,
        transformations=None,
        path_filter=train_filter,
    )

    logger.info(
        "Split %s: %d anchor config(s), %d train samples",
        split_hash,
        len(anchor_config_hashes),
        len(train_dataset),
    )

    for config_hash in tqdm(anchor_config_hashes, desc="Anchor configs", unit="config"):
        run_single_anchor_config(
            config_hash=config_hash,
            train_dataset=train_dataset,
            dataset_hash=dataset_hash,
            labels_dir=labels_dir,
            config=config,
            split_hash=split_hash,
        )


def main() -> None:
    config: Dict[str, Any] = {
        "dataset_hash": "latest",
        "fg_iou_thresh": 0.5,
        "bbox_level_iou_threshold": 0.5,
        "background_iou_threshold": 0.4,
        "metric_threshold": 0.5,
        "min_gt_threshold_ratio": 0.95,
        "img_size": DL_LIB_ETALON_FIXED_SIZE,
        "save_visualizations": True,
        "run_detector_matching": True,
        "run_cell_recall": True,
        "split_params": {
            "val_ratio": 0.15,
            "seed": 42,
        },
        "anchor_config_hashes": DEFAULT_ANCHOR_CONFIG_HASHES,
        "seed": 42,
    }

    if config["dataset_hash"] == "latest":
        latest = resolve_latest_dataset_hash()
        if latest:
            config["dataset_hash"] = latest
            logger.info("Using latest dataset hash: %s", latest)

    run_anchor_evaluation(config)
    logger.info("All anchor configs finished.")


if __name__ == "__main__":
    main()
