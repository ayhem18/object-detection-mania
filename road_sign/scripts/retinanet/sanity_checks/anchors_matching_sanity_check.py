"""
Pre-training anchor evaluation (Tests 1 + 2) for road-sign RetinaNet datasets.

Run from monorepo root::

    uv run python road_sign/scripts/anchors/register_anchor_configs.py
    uv run python road_sign/scripts/retinanet/sanity_checks/anchors_matching_sanity_check.py

Outputs (per dataset version, split, and registered anchor recipe)::

    road_sign/artifacts/retinanet/anchor_matching_sa/{version}/{dataset_hash}/{split_hash}/{config_hash}/
        anchor_config.json
        summary.json
        cell_recall/   (optional viz)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from tqdm import tqdm

_current = Path(__file__).resolve().parent
while _current != _current.parent:
    if (_current / "road_sign").is_dir() and (_current / "home_made_od").is_dir():
        if str(_current) not in sys.path:
            sys.path.insert(0, str(_current))
        break
    _current = _current.parent
else:
    raise RuntimeError("Could not find monorepo root (road_sign + home_made_od).")

from home_made_od.anchors.anchor_computation_strategies import (  # noqa: E402
    assign_to_level_by_area,
    assign_to_level_by_dimension,
    calculate_fpn_specs,
)
from home_made_od.anchors.anchor_config_registry import (  # noqa: E402
    REGISTERED_ANCHOR_CONFIGS,
    load_registered_anchor_config,
    recipe_to_payload,
)
from home_made_od.anchors.anchor_eval_utils import (  # noqa: E402
    FlatCandidateBatch,
    flatten_candidate_boxes,
)
from home_made_od.anchors.anchor_evaluation import (  # noqa: E402
    evaluate_anchors,
    format_anchor_evaluation_report,
)
from home_made_od.general.path_utils import (  # noqa: E402
    DATASET_VERSION_PATCH,
    DATASET_VERSION_RESIZED,
    RoadSignDatasetVersion,
    create_and_cache_split,
    load_dataset_config,
    patch_config_path,
    resolve_latest_dataset_hash,
    resolve_latest_split_hash,
    road_sign_artifacts_root,
)
from home_made_od.retinanet.error_analysis import build_retinanet_grid_layout  # noqa: E402
from home_made_od.retinanet.retinanet_anchors import (  # noqa: E402
    AnchorTrainingSpec,
    compute_and_save_retinanet_anchors,
)
from home_made_od.retinanet.retinanet_detector import build_retinanet  # noqa: E402
from mypt.code_utils.pytorch_utils import seed_everything  # noqa: E402
from road_sign.utils.data_utils import (  # noqa: E402
    RoadSignRetinaNetDataset,
    build_retinanet_class_id_map,
    collect_train_box_dimensions_for_anchors,
    load_road_sign_class_mapping,
    retinanet_num_classes,
)

logger = logging.getLogger(__name__)

RETINANET_MODEL_NAME = "retinanet"
VERSION_CHOICES: List[RoadSignDatasetVersion] = [
    DATASET_VERSION_RESIZED,
    DATASET_VERSION_PATCH,
]
DEFAULT_DATASET_VERSIONS = list(VERSION_CHOICES)

DEFAULT_ANCHOR_CONFIG_HASHES: List[str] = [
    recipe_to_payload(recipe)["config_hash"]
    for recipe in REGISTERED_ANCHOR_CONFIGS.values()
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RetinaNet anchor matching sanity check.")
    parser.add_argument(
        "--dataset-version",
        choices=VERSION_CHOICES,
        action="append",
        dest="dataset_versions",
        help="Run one or more versions (default: resized + patch_based).",
    )
    parser.add_argument("--dataset-hash", default=None)
    parser.add_argument("--split-hash", default=None)
    parser.add_argument(
        "--anchor-config-hash",
        action="append",
        dest="anchor_config_hashes",
        help="Registered recipe hash(es); default: all registered recipes.",
    )
    parser.add_argument("--output-dir", default=None, help="Override artifact root.")
    parser.add_argument("--force-recompute-anchors", action="store_true")
    parser.add_argument("--fg-iou-thresh", type=float, default=0.5)
    parser.add_argument("--bbox-level-iou-threshold", type=float, default=0.5)
    parser.add_argument("--background-iou-threshold", type=float, default=0.4)
    parser.add_argument("--metric-threshold", type=float, default=0.5)
    parser.add_argument("--min-gt-threshold-ratio", type=float, default=0.95)
    parser.add_argument("--no-visualizations", default=False)
    parser.add_argument("--skip-test1", action="store_true", help="Skip detector matching.")
    parser.add_argument("--skip-test2", action="store_true", help="Skip cell recall.")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--retinanet-label-start", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_dataset_versions(args: argparse.Namespace) -> List[RoadSignDatasetVersion]:
    if args.dataset_versions:
        return list(args.dataset_versions)
    return list(DEFAULT_DATASET_VERSIONS)


def load_optional_patch_config(dataset_hash: str) -> Optional[Dict[str, Any]]:
    path = patch_config_path(dataset_hash)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_target_size(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
) -> Tuple[int, int]:
    ds_cfg = load_dataset_config(version, dataset_hash)
    if "target_size" in ds_cfg:
        ts = ds_cfg["target_size"]
        return int(ts[0]), int(ts[1])

    if version == DATASET_VERSION_PATCH:
        patch_cfg = load_optional_patch_config(dataset_hash)
        if patch_cfg and "target_size" in patch_cfg:
            ts = patch_cfg["target_size"]
            return int(ts[0]), int(ts[1])

    return 512, 512


def resolve_split_hash(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: Optional[str],
    *,
    train_ratio: float,
    seed: int,
) -> str:
    resolved = split_hash or resolve_latest_split_hash(version, dataset_hash)
    if resolved is not None:
        return resolved
    logger.info(
        "No split found; creating split train_ratio=%s seed=%s",
        train_ratio,
        seed,
    )
    _, _, created = create_and_cache_split(
        version=version,
        dataset_hash=dataset_hash,
        train_ratio=train_ratio,
        seed=seed,
    )
    return created


def anchor_matching_output_dir(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    config_hash: str,
    output_dir_override: Optional[str] = None,
) -> Path:
    if output_dir_override:
        return Path(output_dir_override) / version / dataset_hash / split_hash / config_hash
    return (
        road_sign_artifacts_root()
        / RETINANET_MODEL_NAME
        / "anchor_matching_sa"
        / version
        / dataset_hash
        / split_hash
        / config_hash
    )


def _boxes_tensor(boxes: Any) -> torch.Tensor:
    if hasattr(boxes, "data"):
        return boxes.data.detach().cpu().float()
    if hasattr(boxes, "detach"):
        return boxes.detach().cpu().float()
    return torch.as_tensor(boxes, dtype=torch.float32)


def assign_level_for_box(
    box: torch.Tensor,
    *,
    method: str,
    method_parameters: Dict[str, Any],
) -> str:
    """FPN level assignment for one GT box (same rules as anchor clustering)."""
    y1, x1, y2, x2 = box.tolist()
    dims = (float(y2 - y1), float(x2 - x1))
    fpn_coefficient = int(method_parameters["fpn_coefficient"])
    fpn_specs = calculate_fpn_specs(fpn_coefficient)

    if method == "dimension_based":
        fallback = method_parameters.get("fallback_level", "P3")
        return assign_to_level_by_dimension(
            dims, fpn_specs, fpn_coefficient, fallback_level=fallback
        )["assigned_level"]
    return assign_to_level_by_area(dims, fpn_specs, fpn_coefficient)["assigned_level"]


def load_flat_candidates_from_dataset(
    train_dataset: RoadSignRetinaNetDataset,
    *,
    method: str,
    method_parameters: Dict[str, Any],
) -> Tuple[FlatCandidateBatch, List[str]]:
    """Load train GT boxes and per-box FPN assignments for Test 1 stratification."""
    boxes_per_image: List[torch.Tensor] = []
    image_stems: List[str] = []
    assigned_levels: List[str] = []

    for idx in tqdm(range(len(train_dataset)), desc="Loading GT boxes", unit="sample"):
        _image, target = train_dataset[idx]
        box_tensor = _boxes_tensor(target["boxes"])
        if box_tensor.numel() == 0:
            continue

        boxes_per_image.append(box_tensor)
        image_stems.append(train_dataset.sample_ids[idx])
        for box in box_tensor:
            assigned_levels.append(
                assign_level_for_box(box, method=method, method_parameters=method_parameters)
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


def extract_model_anchors_and_grid(
    anchor_spec: AnchorTrainingSpec,
    num_classes: int,
    img_size: Tuple[int, int],
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.Tensor, list, list]:
    """Build RetinaNet; return model, flat anchors, prior_to_cell, level_grids."""
    used_levels = list(anchor_spec.used_fpn_levels)
    anchor_config_path = anchor_spec.config_path

    logger.info(
        "Building RetinaNet (%s): levels=%s from %s",
        anchor_spec.method,
        used_levels,
        anchor_config_path,
    )

    model = build_retinanet(
        anchor_spec=anchor_spec,
        num_classes=num_classes,
        img_size=img_size,
        device=device,
    )
    model.eval()

    raw_image = torch.zeros((3, img_size[0], img_size[1]), dtype=torch.float32)
    level_ids = used_levels

    with torch.no_grad():
        images, _ = model.transform([raw_image], None)
        if tuple(images.tensors.shape[-2:]) != img_size:
            raise ValueError(
                f"Image size mismatch: {tuple(images.tensors.shape[-2:])} != {img_size}"
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


def run_single_anchor_config(
    *,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    config_hash: str,
    target_size: Tuple[int, int],
    class_id_map: Dict[int, int],
    eval_kwargs: Dict[str, Any],
    force_recompute_anchors: bool,
    output_dir_override: Optional[str],
) -> None:
    """Optimize anchors + run Tests 1 & 2 for one registered anchor config."""
    recipe = load_registered_anchor_config(config_hash)
    method = recipe["method"]
    method_parameters = recipe["method_parameters"]

    config_dir = anchor_matching_output_dir(
        version, dataset_hash, split_hash, config_hash, output_dir_override
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    anchor_config_path = config_dir / "anchor_config.json"

    box_dimensions = collect_train_box_dimensions_for_anchors(
        version, dataset_hash, split_hash, target_size
    )
    if not box_dimensions:
        raise ValueError(
            f"No training boxes for {version}/{dataset_hash}/{split_hash}. "
            "Check labels and split files."
        )

    logger.info(
        "Computing anchors: method=%s, config_hash=%s, %d boxes -> %s",
        method,
        config_hash,
        len(box_dimensions),
        anchor_config_path,
    )
    anchor_spec = compute_and_save_retinanet_anchors(
        box_dimensions,
        anchor_config_path,
        method=method,
        method_parameters=method_parameters,
        force=force_recompute_anchors,
    )

    num_classes = retinanet_num_classes(class_id_map)
    device = torch.device("cpu")
    model, all_anchors, prior_to_cell, level_grids = extract_model_anchors_and_grid(
        anchor_spec, num_classes, target_size, device
    )

    train_dataset = RoadSignRetinaNetDataset.from_split(
        version,
        dataset_hash,
        split_hash,
        split="train",
        target_size=target_size,
        class_id_map=class_id_map,
        transformations=None,
    )
    if len(train_dataset) == 0:
        logger.warning("No training samples for config %s", config_hash)
        return

    logger.info("Running Tests 1 & 2 for %s", method)
    candidates, assigned_levels = load_flat_candidates_from_dataset(
        train_dataset,
        method=method,
        method_parameters=method_parameters,
    )
    if candidates.num_boxes == 0:
        logger.warning("No GT boxes with objects for config %s", config_hash)
        return

    print("\n" + "=" * 60)
    print(f"Anchor config: {method} ({config_hash})")
    print("=" * 60)
    print(f"Version:       {version}")
    print(f"Dataset Hash:  {dataset_hash}")
    print(f"Split Hash:    {split_hash}")
    print(f"Train Images:  {len(train_dataset)}")
    print(f"GT Boxes:      {candidates.num_boxes}")
    print(f"Config Dir:    {config_dir}")
    print(f"Image Size:    {target_size} | Anchors: {all_anchors.shape[0]}")
    print(
        f"Test 2 thresholds: recall>={eval_kwargs['metric_threshold']}, "
        f"gt_pass_fraction>={eval_kwargs['min_gt_threshold_ratio']}, "
        f"fg_cell>={eval_kwargs['bbox_level_iou_threshold']}, "
        f"bg_cell<{eval_kwargs['background_iou_threshold']}"
    )

    eval_report = evaluate_anchors(
        reference_bboxes=all_anchors,
        reference_assignments=prior_to_cell,
        grids=level_grids,
        candidates=candidates,
        assigned_levels=assigned_levels,
        model=model,
        output_dir=config_dir if eval_kwargs["save_visualizations"] else None,
        **{k: v for k, v in eval_kwargs.items() if k != "save_visualizations"},
    )

    print("\n" + format_anchor_evaluation_report(eval_report))

    summary_path = config_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(eval_report.to_dict(), handle, indent=2)
    logger.info("Wrote summary: %s", summary_path)


def run_anchor_evaluation_for_version(
    *,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    target_size: Tuple[int, int],
    class_id_map: Dict[int, int],
    anchor_config_hashes: Sequence[str],
    eval_kwargs: Dict[str, Any],
    force_recompute_anchors: bool,
    output_dir_override: Optional[str],
) -> None:
    for config_hash in tqdm(anchor_config_hashes, desc="Anchor configs", unit="config"):
        run_single_anchor_config(
            version=version,
            dataset_hash=dataset_hash,
            split_hash=split_hash,
            config_hash=config_hash,
            target_size=target_size,
            class_id_map=class_id_map,
            eval_kwargs=eval_kwargs,
            force_recompute_anchors=force_recompute_anchors,
            output_dir_override=output_dir_override,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s")
    args = parse_args()
    seed_everything(args.seed)

    versions = resolve_dataset_versions(args)
    anchor_config_hashes = args.anchor_config_hashes or DEFAULT_ANCHOR_CONFIG_HASHES
    eval_kwargs = {
        "fg_iou_threshold": args.fg_iou_thresh,
        "bbox_level_iou_threshold": args.bbox_level_iou_threshold,
        "background_iou_threshold": args.background_iou_threshold,
        "metric_threshold": args.metric_threshold,
        "min_gt_threshold_ratio": args.min_gt_threshold_ratio,
        "run_detector_matching": not args.skip_test1,
        "run_cell_recall": not args.skip_test2,
        "save_visualizations": not args.no_visualizations,
    }

    failures: List[str] = []

    print("=" * 60)
    print("RETINANET ANCHOR MATCHING SANITY CHECK")
    print("=" * 60)
    print(f"  versions: {versions}")
    print(f"  anchor recipes: {len(anchor_config_hashes)}")

    for version in versions:
        dataset_hash = args.dataset_hash or resolve_latest_dataset_hash(version)
        if dataset_hash is None:
            failures.append(f"No dataset for {version!r}")
            continue

        try:
            split_hash = resolve_split_hash(
                version,
                dataset_hash,
                args.split_hash,
                train_ratio=args.train_ratio,
                seed=args.split_seed,
            )
            target_size = resolve_target_size(version, dataset_hash)
            class_mapping = load_road_sign_class_mapping(version, dataset_hash)
            class_id_map = build_retinanet_class_id_map(
                class_mapping, start_index=args.retinanet_label_start
            )

            run_anchor_evaluation_for_version(
                version=version,
                dataset_hash=dataset_hash,
                split_hash=split_hash,
                target_size=target_size,
                class_id_map=class_id_map,
                anchor_config_hashes=anchor_config_hashes,
                eval_kwargs=eval_kwargs,
                force_recompute_anchors=args.force_recompute_anchors,
                output_dir_override=args.output_dir,
            )
        except Exception as exc:
            failures.append(f"{version}: {exc}")
            logger.exception("Anchor matching sanity failed for %s", version)

    if failures:
        print("\nFAILURES:")
        for msg in failures:
            print(f"  - {msg}")
        raise SystemExit(1)

    print(f"\nAll anchor configs finished ({len(versions)} version(s)).")


if __name__ == "__main__":
    main()
