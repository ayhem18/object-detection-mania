import json
from pathlib import Path

import numpy as np

from dl_lib.etalon_object_detection.modules.anchors.anchor_computation_strategies import (
    anchor_config_output_dir,
    generate_anchors,
)
from dl_lib.etalon_object_detection.modules.anchors.anchor_config_registry import (
    DIMENSION_BASED_ANCHOR_CONFIG,
    load_registered_anchor_config,
    recipe_to_payload,
)
from dl_lib.etalon_object_detection.modules.ds_utils import (
    DL_LIB_ETALON_FIXED_SIZE,
    WeldingDetectionDataset,
    get_path_split_callables,
)
from dl_lib.etalon_object_detection.modules.path_layout import (
    images_dir,
    labels_data_dir,
    master_labels_path,
    resolve_latest_dataset_hash,
)
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_anchors import (
    optimize_retinanet_anchors,
    resolve_split_anchor_config,
)
from mypt.code_utils.pytorch_utils import seed_everything

_DEFAULT_CONFIG_HASH = recipe_to_payload(DIMENSION_BASED_ANCHOR_CONFIG)["config_hash"]


def print_aspect_ratio_report(
    anchor_config_path: Path,
    enriched_json_path: Path | None,
    num_centroids: int,
    max_iters: int,
    seed: int,
) -> None:
    """Prints torchvision-compatible aspect ratios (y_dim / x_dim) per FPN level."""
    with open(anchor_config_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    used_fpn_levels = metadata["used_fpn_levels"]
    stored_ratios = metadata["level_aspect_ratios"]

    print(f"Anchor config: {anchor_config_path}")
    print(f"Active FPN levels: {used_fpn_levels}\n")

    for level in used_fpn_levels:
        ratios = stored_ratios[level]
        print(f"--- {level} (stored ratios, y_dim/x_dim) ---")
        for ratio in ratios:
            if ratio > 1.2:
                shape = "Tall"
            elif ratio < 0.8:
                shape = "Wide"
            else:
                shape = "Square-ish"
            print(f"  {ratio:.3f} ({shape})")
        print("-" * 40)

    if enriched_json_path is None or not enriched_json_path.exists():
        return

    with open(enriched_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    level_yx_buckets = {level: [] for level in used_fpn_levels}
    for sample in data["samples"]:
        for etalon in sample["enriched_etalons"]:
            assigned_level = etalon["assigned_level"]
            if assigned_level not in used_fpn_levels:
                max_vis = etalon.get("max_visible_level")
                if max_vis in used_fpn_levels:
                    assigned_level = max_vis
                else:
                    assigned_level = used_fpn_levels[0]

            level_yx_buckets[assigned_level].append((etalon["y_dim"], etalon["x_dim"]))

    print("\nRecomputed ratios from enriched manifest (train split):")
    for level, yx_dims in level_yx_buckets.items():
        count = len(yx_dims)
        if count < num_centroids:
            print(f"{level}: Not enough samples ({count}) to recompute {num_centroids} centroids.")
            continue

        centroids = generate_anchors(
            yx_dims, num_anchors=num_centroids, max_iters=max_iters, seed=seed
        )
        ratios = np.sort(centroids[:, 0] / centroids[:, 1])
        gt_median = np.median([y / x for y, x in yx_dims])
        print(f"--- {level} ({count} etalons) | GT median y/x={gt_median:.3f} ---")
        for ratio in ratios:
            print(f"  {ratio:.3f}")
        print("-" * 40)


def run_analysis(
    dataset_hash: str,
    split_params: dict,
    anchor_config_hash: str,
    seed: int,
    recompute_anchors: bool,
) -> None:
    recipe = load_registered_anchor_config(anchor_config_hash)
    method = recipe["method"]
    method_parameters = recipe["method_parameters"]
    seed_everything(seed)

    labels_dir = labels_data_dir(dataset_hash)
    master_json = master_labels_path(dataset_hash)
    if not master_json.is_file():
        print(f"Error: {master_json} not found.")
        return

    train_filter, _, split_hash = get_path_split_callables(
        master_json_path=str(master_json),
        val_ratio=split_params["val_ratio"],
        seed=split_params["seed"],
        save_dir=str(labels_dir.parent),
    )

    if recompute_anchors:
        config_dir = anchor_config_output_dir(dataset_hash, split_hash, anchor_config_hash)
        train_dataset = WeldingDetectionDataset(
            images_dir=str(images_dir()),
            cache_dir=str(labels_dir),
            target_size=DL_LIB_ETALON_FIXED_SIZE,
            transformations=None,
            path_filter=train_filter,
        )
        _, anchor_config_path = optimize_retinanet_anchors(
            dataset=train_dataset,
            output_dir=config_dir,
            dataset_hash=dataset_hash,
            method=method,
            method_parameters=method_parameters,
        )
        config_dir = anchor_config_path.parent
    else:
        anchor_config_path, config_dir = resolve_split_anchor_config(
            dataset_hash, config_hash=anchor_config_hash, split_hash=split_hash
        )

    enriched_json_path = config_dir / "master_labels_enriched.json"
    print_aspect_ratio_report(
        anchor_config_path=anchor_config_path,
        enriched_json_path=enriched_json_path if enriched_json_path.exists() else None,
        num_centroids=method_parameters["num_aspect_ratios"],
        max_iters=method_parameters["max_iters"],
        seed=seed,
    )


if __name__ == "__main__":
    dataset_hash = "latest"
    if dataset_hash == "latest":
        latest = resolve_latest_dataset_hash()
        if latest:
            dataset_hash = latest
            print(f"Using latest dataset hash: {dataset_hash}")

    run_analysis(
        dataset_hash=dataset_hash,
        split_params={"val_ratio": 0.15, "seed": 42},
        anchor_config_hash=_DEFAULT_CONFIG_HASH,
        seed=42,
        recompute_anchors=False,
    )
