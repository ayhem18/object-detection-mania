"""
Visual sanity check for RetinaNet training augmentations (road-sign datasets).

Each augmentation is forced (p=1.0) so every saved sample shows the effect.
Resize is applied by :class:`RoadSignRetinaNetDataset` (same as training).

Run from monorepo root (default: resized + patch_based)::

    uv run python road_sign/scripts/retinanet/sanity_checks/augmentation_sa.py

Outputs::

    road_sign/artifacts/retinanet/augmentation_sa/{version}/{dataset_hash}/{split_hash}/
        HorizontalFlip/
        VerticalFlip/
        ColorJitter/
        Rotation30/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from script_utils import (  # noqa: E402
    VERSION_CHOICES,
    add_monorepo_to_sys_path,
    resolve_dataset_versions,
    resolve_split_hash,
    resolve_target_size,
    retinanet_artifact_dir,
)

add_monorepo_to_sys_path()

from home_made_od.general.path_utils import (  # noqa: E402
    RoadSignDatasetVersion,
    resolve_latest_dataset_hash,
)
from mypt.code_utils.pytorch_utils import seed_everything  # noqa: E402
from road_sign.utils.data_utils import (  # noqa: E402
    RoadSignRetinaNetDataset,
    build_retinanet_class_id_map,
    load_road_sign_class_mapping,
    retinanet_cls_id_to_name,
)

logger = logging.getLogger(__name__)

# Match patch training defaults in train_patch.py
DEFAULT_COLOR_JITTER = {"brightness": 0.2, "contrast": 0.2}
DEFAULT_ROTATION_DEGREES = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize RetinaNet augmentations on road-sign data."
    )
    parser.add_argument(
        "--dataset-version",
        choices=VERSION_CHOICES,
        default=None,
        help="Single version (default: resized and patch_based).",
    )
    parser.add_argument("--dataset-hash", default=None)
    parser.add_argument("--split-hash", default=None)
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Split to sample images from.",
    )
    parser.add_argument("--target-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--retinanet-label-start", type=int, default=1)
    return parser.parse_args()


def resolve_output_dir(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    override: Optional[str],
) -> Path:
    return retinanet_artifact_dir(
        "augmentation_sa",
        version,
        dataset_hash,
        split_hash,
        output_dir_override=override,
        mkdir=True,
    )


def build_augmentation_presets(
    *,
    color_jitter: Dict[str, float],
    rotation_degrees: float,
) -> Dict[str, v2.Compose]:
    """
    Augmentations to visualize.

    ``p=1.0`` on random ops forces the effect on every sample. ``ToDtype`` last;
    ``RoadSignRetinaNetDataset`` injects ``Resize`` when missing.
    """
    def _base(*ops: Any) -> v2.Compose:
        return v2.Compose([v2.ToImage(), *ops, v2.ToDtype(torch.float32, scale=True)])

    cj = color_jitter
    return {
        "HorizontalFlip": _base(v2.RandomHorizontalFlip(p=1.0)),
        "VerticalFlip": _base(v2.RandomVerticalFlip(p=1.0)),
        "ColorJitter": _base(
            v2.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
            )
        ),
        "Rotation30": _base(v2.RandomRotation(degrees=rotation_degrees)),
    }


def _tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    img = image.permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _boxes_numpy(boxes: Any) -> np.ndarray:
    if hasattr(boxes, "data"):
        return boxes.data.detach().cpu().numpy()
    if hasattr(boxes, "detach"):
        return boxes.detach().cpu().numpy()
    return np.asarray(boxes)


def visualize_augmentation(
    aug_name: str,
    transform: v2.Compose,
    base_dataset: RoadSignRetinaNetDataset,
    label_names: Dict[int, str],
    output_dir: Path,
    indices: Sequence[int],
) -> None:
    print(f"\n  [{aug_name}]")
    aug_dir = output_dir / aug_name
    aug_dir.mkdir(parents=True, exist_ok=True)

    dataset = RoadSignRetinaNetDataset(
        data_pairs=base_dataset.data_pairs,
        target_size=base_dataset.target_size,
        class_id_map=base_dataset.class_id_map,
        transformations=transform,
        sample_ids=base_dataset.sample_ids,
    )

    for plot_i, idx in enumerate(indices):
        image, target = dataset[idx]
        img_np = _tensor_to_uint8(image)
        boxes = _boxes_numpy(target["boxes"])
        labels = target["labels"].cpu().numpy()

        plt.figure(figsize=(10, 10))
        plt.imshow(img_np)
        ax = plt.gca()
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor="lime", linewidth=2,
                )
            )
            name = label_names.get(int(label), str(int(label)))
            ax.text(
                x1, max(y1 - 5, 0), f"{int(label)}:{name}",
                color="lime", fontsize=9, backgroundcolor="black",
            )
        sid = dataset.sample_ids[idx].replace("/", "_")
        plt.title(f"{aug_name} | sample {plot_i} | idx={idx} | {sid}")
        plt.axis("off")
        save_path = aug_dir / f"sample_{plot_i:03d}_{sid}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"    saved {save_path.name}")


def run_augmentation_sanity_check(
    *,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    split: Literal["train", "val"],
    target_size: Tuple[int, int],
    output_dir: Path,
    num_samples: int,
    seed: int,
    class_id_map: Dict[int, int],
) -> None:
    class_mapping = load_road_sign_class_mapping(version, dataset_hash)
    label_names = retinanet_cls_id_to_name(class_mapping, class_id_map)

    base_dataset = RoadSignRetinaNetDataset.from_split(
        version,
        dataset_hash,
        split_hash,
        split=split,
        target_size=target_size,
        class_id_map=class_id_map,
        transformations=None,
    )

    rng = random.Random(seed)
    indices = rng.sample(
        range(len(base_dataset)),
        min(num_samples, len(base_dataset)),
    )

    augmentations = build_augmentation_presets(
        color_jitter=DEFAULT_COLOR_JITTER,
        rotation_degrees=DEFAULT_ROTATION_DEGREES,
    )

    config_record = {
        "dataset_version": version,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "split": split,
        "target_size": list(target_size),
        "class_id_map": class_id_map,
        "sample_indices": list(indices),
        "augmentations": list(augmentations.keys()),
        "color_jitter": DEFAULT_COLOR_JITTER,
        "rotation_degrees": DEFAULT_ROTATION_DEGREES,
    }
    with open(output_dir / "augmentation_sa_config.json", "w", encoding="utf-8") as handle:
        json.dump(config_record, handle, indent=2)

    print("=" * 60)
    print(f"AUGMENTATION SANITY — {version} / {dataset_hash[:8]}… / {split}")
    print("=" * 60)
    print(f"  output: {output_dir}")
    print(f"  samples: {len(indices)} from {len(base_dataset)} images")

    for name, transform in augmentations.items():
        visualize_augmentation(
            name,
            transform,
            base_dataset,
            label_names,
            output_dir,
            indices,
        )

    print(f"\nDone. Visualizations under {output_dir}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    seed_everything(args.seed)

    versions = resolve_dataset_versions(args)
    failures: List[str] = []

    print("=" * 60)
    print("RETINANET AUGMENTATION SANITY CHECK")
    print("=" * 60)
    print(f"  versions: {versions}")

    for version in versions:
        dataset_hash = args.dataset_hash or resolve_latest_dataset_hash(version)
        if dataset_hash is None:
            failures.append(f"No dataset for {version!r}")
            continue

        split_hash = resolve_split_hash(
            version, dataset_hash, args.split_hash, create_if_missing=False
        )
        if split_hash is None:
            failures.append(f"No split for {version}/{dataset_hash}")
            continue

        try:
            target_size = resolve_target_size(
                version,
                dataset_hash,
                cli_target_size=args.target_size,
            )
            class_mapping = load_road_sign_class_mapping(version, dataset_hash)
            class_id_map = build_retinanet_class_id_map(
                class_mapping, start_index=args.retinanet_label_start
            )
            out_dir = resolve_output_dir(
                version, dataset_hash, split_hash, args.output_dir
            )
            run_augmentation_sanity_check(
                version=version,
                dataset_hash=dataset_hash,
                split_hash=split_hash,
                split=args.split,
                target_size=target_size,
                output_dir=out_dir,
                num_samples=args.num_samples,
                seed=args.seed,
                class_id_map=class_id_map,
            )
        except Exception as exc:
            failures.append(f"{version}: {exc}")
            logger.exception("Augmentation sanity failed for %s", version)

    if failures:
        print("\nFAILURES:")
        for msg in failures:
            print(f"  - {msg}")
        raise SystemExit(1)

    print(f"\nAll augmentation checks passed ({len(versions)} version(s)).")


if __name__ == "__main__":
    main()
