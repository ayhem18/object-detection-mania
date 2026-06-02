"""
Create a train/val split under ``data/{dataset_type}/{dataset_hash}/splits/{split_hash}/``
and optionally compute YOLO anchors under::

    road_sign/artifacts/{model_name}/{dataset_type}/{dataset_hash}/{split_hash}/anchors/
"""

from __future__ import annotations

import argparse

from home_made_od.general.path_utils import (
    DATASET_VERSION_ORIGINAL,
    DATASET_VERSION_PATCH,
    DATASET_VERSION_RESIZED,
    RoadSignDatasetVersion,
    compute_and_cache_anchors,
    create_and_cache_split,
    load_dataset_config,
    model_anchors_json_path,
    model_artifact_root,
    resolve_latest_dataset_hash,
    split_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create road-sign dataset split and artifacts.")
    parser.add_argument(
        "--dataset-version",
        choices=[DATASET_VERSION_ORIGINAL, DATASET_VERSION_RESIZED, DATASET_VERSION_PATCH],
        default=DATASET_VERSION_RESIZED,
    )
    parser.add_argument("--dataset-hash", default=None, help="Defaults to latest for the version.")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-name",
        default="yolov2",
        help="Model namespace under road_sign/artifacts/ (e.g. yolov2, retinanet).",
    )
    parser.add_argument("--num-anchors", type=int, default=5)
    parser.add_argument(
        "--anchors",
        action="store_true",
        help="Compute YOLO KMeans anchors (yolov2 only; RetinaNet uses its own anchor logic in training).",
    )
    parser.add_argument(
        "--force-anchors",
        action="store_true",
        help="Recompute anchors even if anchors.json already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version: RoadSignDatasetVersion = args.dataset_version

    dataset_hash = args.dataset_hash or resolve_latest_dataset_hash(version)
    if dataset_hash is None:
        raise FileNotFoundError(
            f"No dataset found for version {version!r}. Run the data prep script first."
        )

    config = load_dataset_config(version, dataset_hash)
    print(f"Dataset: {version}/{dataset_hash} ({config.get('image_count', '?')} images)")

    train_ids, val_ids, split_hash = create_and_cache_split(
        version=version,
        dataset_hash=dataset_hash,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(f"Split directory: {split_dir(version, dataset_hash, split_hash)}")

    artifact_root = model_artifact_root(
        args.model_name, version, dataset_hash, split_hash
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    print(f"Artifact root: {artifact_root}")

    if args.anchors:
        if args.model_name != "yolov2":
            print(
                f"Warning: --anchors is intended for yolov2; {args.model_name!r} "
                "typically computes anchors in its training script."
            )
        compute_and_cache_anchors(
            args.model_name,
            version,
            dataset_hash,
            split_hash,
            num_anchors=args.num_anchors,
            seed=args.seed,
            force=args.force_anchors,
        )
        print(f"Anchors: {model_anchors_json_path(args.model_name, version, dataset_hash, split_hash)}")

    print(f"Done. train={len(train_ids)} val={len(val_ids)} split_hash={split_hash}")


if __name__ == "__main__":
    main()
