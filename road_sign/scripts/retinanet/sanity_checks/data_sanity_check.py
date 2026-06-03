"""
RetinaNet data sanity check (three stages).

1. Labels on disk (YOLO + class_id_map); fixed on-disk image size (resized / patch)
2. PyTorch dataset output (RoadSignRetinaNetDataset); every tensor (3, H, W)
3. model.transform in train and eval modes; spatial size unchanged

Run from monorepo root (default: **resized** + **patch_based**, **train** + **val**)::

    uv run python road_sign/scripts/retinanet/sanity_checks/data_sanity_check.py

    uv run python road_sign/scripts/retinanet/sanity_checks/data_sanity_check.py \\
        --dataset-version resized

Outputs::

    road_sign/artifacts/retinanet/data_sanity_check/{version}/{dataset_hash}/{split_hash}/
        01_raw_yolo/              # train + val label/size audit
        train/
            02_dataset_output/
            02_parse_consistency/
            03_transform_train/
            03_transform_eval/
        val/
            ...
"""
 
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from script_utils import (  # noqa: E402
    RETINANET_MODEL_NAME,
    VERSION_CHOICES,
    add_monorepo_to_sys_path,
    build_retinanet_from_spec,
    load_finalized_anchor_spec,
    resolve_dataset_versions,
    resolve_split_hash,
    resolve_target_size,
    retinanet_artifact_dir,
)

add_monorepo_to_sys_path()

from home_made_od.general.path_utils import (  # noqa: E402
    RoadSignDatasetVersion,
    model_retinanet_anchor_config_path,
    resolve_latest_dataset_hash,
)
from mypt.code_utils.pytorch_utils import seed_everything  # noqa: E402
from road_sign.utils.data_utils import (  # noqa: E402
    RoadSignRetinaNetDataset,
    _yolo_lines_to_xyxy,
    build_retinanet_class_id_map,
    get_train_image_label_pairs,
    load_road_sign_class_mapping,
    load_split_lists,
    retinanet_cls_id_to_name,
    retinanet_collate_fn,
    retinanet_num_classes,
    validate_retinanet_class_id_map,
)

logger = logging.getLogger(__name__)

SPLIT_CHOICES = ("train", "val")
DEFAULT_SPLITS: Tuple[Literal["train", "val"], ...] = ("train", "val")

FIXED_SIZE_DATASET_VERSIONS = frozenset(VERSION_CHOICES)

DIR_STAGE_1 = "01_raw_yolo"
DIR_STAGE_2 = "02_dataset_output"
DIR_STAGE_2_CONSISTENCY = "02_parse_consistency"
DIR_STAGE_3_TRAIN = "03_transform_train"
DIR_STAGE_3_EVAL = "03_transform_eval"


@dataclass
class ImageSizeAuditStats:
    """On-disk PIL size vs expected (height, width)."""

    num_images: int = 0
    correct: int = 0
    wrong_sizes: Dict[Tuple[int, int], int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def record_wrong(self, actual_wh: Tuple[int, int]) -> None:
        self.wrong_sizes[actual_wh] = self.wrong_sizes.get(actual_wh, 0) + 1


@dataclass
class TensorSizeAuditStats:
    """Dataset __getitem__ tensor shape vs expected (H, W)."""

    num_samples: int = 0
    correct: int = 0
    wrong_shapes: Dict[Tuple[int, ...], int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class LabelAuditStats:
    num_images: int = 0
    num_images_with_boxes: int = 0
    num_boxes: int = 0
    unmapped_class_ids: set[int] = field(default_factory=set)
    unknown_class_ids: set[int] = field(default_factory=set)
    invalid_coord_lines: int = 0
    degenerate_boxes: int = 0
    model_label_counts: Dict[int, int] = field(default_factory=dict)
    dataset_label_counts: Dict[int, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RetinaNet road-sign data sanity check (3 stages). "
            "By default runs resized and patch_based datasets, train and val splits."
        )
    )
    parser.add_argument(
        "--dataset-version",
        choices=VERSION_CHOICES,
        default=None,
        help="Run a single dataset version (default: both resized and patch_based).",
    )
    parser.add_argument("--dataset-hash", default=None, help="Latest per version if omitted.")
    parser.add_argument("--split-hash", default=None, help="Latest per version if omitted.")
    parser.add_argument("--target-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root override; still nested by version/hash/split_hash when running all.",
    )
    parser.add_argument("--anchor-config-path", default=None)
    parser.add_argument(
        "--skip-stage-3",
        action="store_true",
        help="Skip model.transform checks.",
    )
    parser.add_argument("--retinanet-label-start", type=int, default=1)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_CHOICES,
        default=list(DEFAULT_SPLITS),
        help="Which splits to run for stages 2 and 3 (default: train val).",
    )
    return parser.parse_args()


def resolve_run_output_dir(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    output_dir_override: Optional[str],
) -> Path:
    return retinanet_artifact_dir(
        "data_sanity_check",
        version,
        dataset_hash,
        split_hash,
        output_dir_override=output_dir_override,
        mkdir=True,
    )


def build_dataset_transforms(*, train_augment: bool) -> v2.Compose:
    """Same pattern as training: Resize inside dataset; optional flip here."""
    transforms: List[Any] = [v2.ToImage()]
    if train_augment:
        transforms.append(v2.RandomHorizontalFlip(p=0.5))
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    return v2.Compose(transforms)


def _pil_size_from_hw(target_size: Tuple[int, int]) -> Tuple[int, int]:
    """Project convention (H, W) → PIL ``Image.size`` (width, height)."""
    h, w = int(target_size[0]), int(target_size[1])
    return w, h


def requires_fixed_image_size(version: RoadSignDatasetVersion) -> bool:
    return version in FIXED_SIZE_DATASET_VERSIONS


def audit_images_on_disk(
    pairs: Sequence[Tuple[str, str]],
    expected_size_hw: Tuple[int, int],
) -> ImageSizeAuditStats:
    """
    Every image file must match the dataset's fixed ``target_size`` (H, W).

    Resized and patch datasets are written at a fixed resolution; mismatches mean
    a bad prep run or wrong ``--target-size``.
    """
    stats = ImageSizeAuditStats()
    expected_wh = _pil_size_from_hw(expected_size_hw)
    exp_h, exp_w = expected_size_hw

    for img_path, _label_path in pairs:
        stats.num_images += 1
        try:
            with Image.open(img_path) as image:
                actual_wh = image.size
        except OSError as exc:
            stats.errors.append(f"Cannot read image {img_path}: {exc}")
            continue

        if actual_wh == expected_wh:
            stats.correct += 1
        else:
            stats.record_wrong(actual_wh)
            if len(stats.errors) < 30:
                stats.errors.append(
                    f"{img_path}: on-disk (W,H)=({actual_wh[0]}, {actual_wh[1]}) "
                    f"!= expected ({exp_w}, {exp_h})"
                )

    if stats.wrong_sizes:
        stats.errors.append(
            f"Summary: {stats.num_images - stats.correct}/{stats.num_images} images "
            f"not {exp_w}x{exp_h}. Distinct wrong (W,H): {dict(stats.wrong_sizes)}"
        )
    return stats


def audit_all_dataset_output_sizes(
    dataset: RoadSignRetinaNetDataset,
    target_size: Tuple[int, int],
) -> TensorSizeAuditStats:
    """Every ``__getitem__`` image tensor must be ``(3, H, W)``."""
    stats = TensorSizeAuditStats()
    h, w = int(target_size[0]), int(target_size[1])
    expected_shape = (3, h, w)

    try:
        from tqdm import tqdm

        iterator: Any = tqdm(range(len(dataset)), desc="  tensor size scan")
    except ImportError:
        iterator = range(len(dataset))
        print(f"  Scanning {len(dataset)} dataset outputs for shape {expected_shape}...")

    for idx in iterator:
        stats.num_samples += 1
        image, _target = dataset[idx]
        shape = tuple(image.shape)
        if shape == expected_shape:
            stats.correct += 1
        else:
            stats.wrong_shapes[shape] = stats.wrong_shapes.get(shape, 0) + 1
            if len(stats.errors) < 30:
                stats.errors.append(
                    f"idx={idx} ({dataset.sample_ids[idx]}): "
                    f"shape {shape} != {expected_shape}"
                )

    if stats.wrong_shapes:
        stats.errors.append(
            f"Summary: {stats.num_samples - stats.correct}/{stats.num_samples} tensors "
            f"wrong shape. Distinct: {dict(stats.wrong_shapes)}"
        )
    return stats


def _print_image_size_audit(stats: ImageSizeAuditStats, expected_hw: Tuple[int, int]) -> None:
    exp_w, exp_h = int(expected_hw[1]), int(expected_hw[0])
    print(f"  Expected on-disk size: {exp_w}x{exp_h} (W×H), tensor (H,W)=({exp_h}, {exp_w})")
    print(f"  Images checked:        {stats.num_images}")
    print(f"  Correct size:        {stats.correct}")
    if stats.wrong_sizes:
        print(f"  Wrong (W,H) counts:    {stats.wrong_sizes}")


def _print_tensor_size_audit(stats: TensorSizeAuditStats, target_size: Tuple[int, int]) -> None:
    h, w = target_size
    print(f"  Expected tensor shape: (3, {h}, {w})")
    print(f"  Samples checked:       {stats.num_samples}")
    print(f"  Correct shape:         {stats.correct}")
    if stats.wrong_shapes:
        print(f"  Wrong shapes:          {stats.wrong_shapes}")


def _parse_yolo_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    parts = line.split()
    if len(parts) < 5:
        return None
    return (
        int(float(parts[0])),
        float(parts[1]),
        float(parts[2]),
        float(parts[3]),
        float(parts[4]),
    )


def _boxes_tensor(boxes: Any) -> torch.Tensor:
    if hasattr(boxes, "data"):
        return boxes.data.detach().cpu()
    if hasattr(boxes, "detach"):
        return boxes.detach().cpu()
    return torch.as_tensor(boxes)


def _draw_boxes(
    ax: plt.Axes,
    boxes: np.ndarray,
    labels: np.ndarray,
    label_names: Dict[int, str],
    edgecolor: str,
) -> None:
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        ax.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=edgecolor, linewidth=2
            )
        )
        name = label_names.get(int(label), f"id={int(label)}")
        ax.text(
            x1, max(y1 - 4, 0), f"{int(label)}:{name}",
            color=edgecolor, fontsize=8, backgroundcolor="black",
        )


def _print_label_audit(stats: LabelAuditStats, class_id_map: Dict[int, int]) -> None:
    print(f"  Images:              {stats.num_images}")
    print(f"  Images with boxes:   {stats.num_images_with_boxes}")
    print(f"  Mapped boxes:        {stats.num_boxes}")
    print(f"  Invalid coord lines: {stats.invalid_coord_lines}")
    print(f"  Degenerate boxes:    {stats.degenerate_boxes}")
    print(f"  class_id_map:        {class_id_map}")
    print("  YOLO class counts:")
    for orig in sorted(stats.dataset_label_counts):
        mid = class_id_map.get(orig, "?")
        print(f"    yolo {orig:2d} -> model {mid!s:>4}: {stats.dataset_label_counts[orig]}")


def _fail_if_errors(stats: LabelAuditStats, stage: str) -> None:
    for msg in stats.warnings:
        print(f"  [WARN] {msg}")
    if stats.errors:
        print(f"\n  [{stage}] FAILED:")
        for msg in stats.errors[:25]:
            print(f"    {msg}")
        if len(stats.errors) > 25:
            print(f"    ... {len(stats.errors) - 25} more")
        raise AssertionError(f"{stage} failed with {len(stats.errors)} error(s).")
    print(f"  [{stage}] PASSED")


def audit_labels_on_disk(
    pairs: Sequence[Tuple[str, str]],
    class_mapping: Dict[int, str],
    class_id_map: Dict[int, int],
) -> LabelAuditStats:
    stats = LabelAuditStats()
    mapping_keys = {int(k) for k in class_mapping}

    for img_path, label_path in pairs:
        stats.num_images += 1
        label_file = Path(label_path)
        if not label_file.is_file():
            stats.errors.append(f"Missing label: {label_path}")
            continue

        with Image.open(img_path) as image:
            width, height = image.size

        with open(label_file, encoding="utf-8") as handle:
            lines = [ln.strip() for ln in handle if ln.strip()]

        if not lines:
            continue

        stats.num_images_with_boxes += 1
        for line in lines:
            parsed = _parse_yolo_line(line)
            if parsed is None:
                stats.invalid_coord_lines += 1
                continue

            orig_cls, cx, cy, bw, bh = parsed
            stats.dataset_label_counts[orig_cls] = (
                stats.dataset_label_counts.get(orig_cls, 0) + 1
            )

            if orig_cls not in mapping_keys:
                stats.unknown_class_ids.add(orig_cls)
            if orig_cls not in class_id_map:
                stats.unmapped_class_ids.add(orig_cls)
                continue

            model_cls = class_id_map[orig_cls]
            stats.model_label_counts[model_cls] = (
                stats.model_label_counts.get(model_cls, 0) + 1
            )
            stats.num_boxes += 1

            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                stats.invalid_coord_lines += 1
                stats.errors.append(
                    f"Invalid normalized coords in {label_path}: "
                    f"cx={cx:.4f} cy={cy:.4f} w={bw:.4f} h={bh:.4f}"
                )

            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            x2 = (cx + bw / 2) * width
            y2 = (cy + bh / 2) * height
            if x2 <= x1 or y2 <= y1:
                stats.degenerate_boxes += 1
                stats.errors.append(f"Degenerate box in {label_path}: {line}")

    if stats.unmapped_class_ids:
        stats.errors.append(
            f"YOLO class ids not in class_id_map: {sorted(stats.unmapped_class_ids)}"
        )
    if stats.unknown_class_ids:
        stats.warnings.append(
            f"YOLO class ids not in class_mapping.json: {sorted(stats.unknown_class_ids)}"
        )
    return stats


def run_stage_1_raw_yolo(
    pairs: Sequence[Tuple[str, str]],
    class_mapping: Dict[int, str],
    class_id_map: Dict[int, int],
    output_dir: Path,
    num_samples: int,
    seed: int,
    *,
    version: RoadSignDatasetVersion,
    target_size: Tuple[int, int],
) -> None:
    print("\n" + "=" * 60)
    print("STAGE 1: Labels on disk (YOLO)")
    print("=" * 60)

    if requires_fixed_image_size(version):
        print("\n  [1a] Fixed image size on disk")
        size_stats = audit_images_on_disk(pairs, target_size)
        _print_image_size_audit(size_stats, target_size)
        _fail_if_errors(
            LabelAuditStats(errors=size_stats.errors),
            "Stage 1 (image size)",
        )
    else:
        print(
            f"\n  [1a] Skipping on-disk size check (version={version!r} is not fixed-size)."
        )

    print("\n  [1b] Label audit")
    stats = audit_labels_on_disk(pairs, class_mapping, class_id_map)
    _print_label_audit(stats, class_id_map)
    _fail_if_errors(stats, "Stage 1 (labels)")

    viz_dir = output_dir / DIR_STAGE_1
    viz_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    indices = rng.sample(range(len(pairs)), min(num_samples, len(pairs)))

    for plot_i, idx in enumerate(indices):
        img_path, label_path = pairs[idx]
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        draw = ImageDraw.Draw(img)

        with open(label_path, encoding="utf-8") as handle:
            for line in handle:
                parsed = _parse_yolo_line(line.strip())
                if parsed is None:
                    continue
                orig_cls, cx, cy, bw, bh = parsed
                x1 = (cx - bw / 2) * width
                y1 = (cy - bh / 2) * height
                x2 = (cx + bw / 2) * width
                y2 = (cy + bh / 2) * height
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
                model_id = class_id_map.get(orig_cls)
                draw.text((x1, y1), f"y{orig_cls}->m{model_id}", fill="lime")

        stem = Path(img_path).stem.replace("/", "_")
        save_path = viz_dir / f"raw_{plot_i:03d}_{stem}.png"
        img.save(save_path)
        print(f"  [saved] {save_path}")


def _assert_dataset_sample(
    idx: int,
    image: torch.Tensor,
    target: Dict[str, Any],
    target_size: Tuple[int, int],
    num_classes: int,
) -> None:
    h, w = int(target_size[0]), int(target_size[1])
    if image.shape != (3, h, w):
        raise AssertionError(
            f"Sample {idx}: image shape {tuple(image.shape)} != (3, {h}, {w})"
        )

    labels = target["labels"].detach().cpu().numpy()
    boxes = _boxes_tensor(target["boxes"]).numpy()

    if len(labels):
        if labels.min() < 1 or labels.max() >= num_classes:
            raise AssertionError(
                f"Sample {idx}: labels {labels.tolist()} outside [1, {num_classes - 1}]"
            )
    for box in boxes:
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise AssertionError(f"Sample {idx}: degenerate box {box}")
        if x1 < -1 or y1 < -1 or x2 > w + 1 or y2 > h + 1:
            raise AssertionError(f"Sample {idx}: box {box} outside {w}x{h}")


def _check_parse_consistency(
    dataset: RoadSignRetinaNetDataset,
    class_id_map: Dict[int, int],
    target_size: Tuple[int, int],
    output_dir: Path,
    num_samples: int,
    seed: int,
    *,
    expect_on_disk_fixed_size: bool,
) -> None:
    """Manual YOLO parse + scale must match dataset output."""
    out = output_dir / DIR_STAGE_2_CONSISTENCY
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(num_samples, len(dataset)))
    mismatches = 0
    size_mismatches = 0
    exp_h, exp_w = int(target_size[0]), int(target_size[1])
    expected_wh = _pil_size_from_hw(target_size)

    for plot_i, idx in enumerate(indices):
        img_path, label_path = dataset.data_pairs[idx]
        with Image.open(img_path) as pil_img:
            orig_w, orig_h = pil_img.size

        if expect_on_disk_fixed_size and (orig_w, orig_h) != expected_wh:
            size_mismatches += 1

        manual_boxes, manual_labels = _yolo_lines_to_xyxy(
            Path(label_path), orig_w, orig_h, class_id_map
        )
        image, target = dataset[idx]
        ds_boxes = _boxes_tensor(target["boxes"])
        ds_labels = target["labels"].detach().cpu()

        h_out, w_out = image.shape[-2:]
        if (h_out, w_out) != (exp_h, exp_w):
            raise AssertionError(
                f"idx={idx}: dataset output {(h_out, w_out)} != target_size {(exp_h, exp_w)}"
            )

        scale_x = w_out / orig_w
        scale_y = h_out / orig_h
        if expect_on_disk_fixed_size:
            if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
                size_mismatches += 1
        expected = manual_boxes.clone()
        expected[:, [0, 2]] *= scale_x
        expected[:, [1, 3]] *= scale_y

        n_m, n_d = len(manual_labels), len(ds_labels)
        box_ok = n_m == n_d and (
            n_m == 0 or torch.allclose(expected, ds_boxes, atol=2.0)
        )
        label_ok = n_m == n_d and (n_m == 0 or torch.equal(manual_labels, ds_labels))

        if not (box_ok and label_ok):
            mismatches += 1

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        img_np = image.permute(1, 2, 0).cpu().numpy()
        axes[0].imshow(img_np)
        _draw_boxes(axes[0], expected.numpy(), manual_labels.numpy(), {}, "yellow")
        axes[0].set_title("Expected (scaled parse)")
        axes[1].imshow(img_np)
        color = "lime" if box_ok and label_ok else "red"
        _draw_boxes(axes[1], ds_boxes.numpy(), ds_labels.numpy(), {}, color)
        axes[1].set_title("Dataset output")
        for ax in axes:
            ax.axis("off")
        status = "ok" if box_ok and label_ok else "BAD"
        fig.suptitle(f"{status} | {Path(img_path).name}")
        plt.savefig(out / f"consistency_{plot_i:03d}_{status}.png", bbox_inches="tight")
        plt.close()

    if size_mismatches and expect_on_disk_fixed_size:
        raise AssertionError(
            f"Fixed-size dataset: {size_mismatches}/{len(indices)} samples had "
            f"on-disk size != {exp_w}x{exp_h} or required rescale in the dataset."
        )
    if mismatches:
        raise AssertionError(
            f"Parse consistency: {mismatches}/{len(indices)} samples mismatched."
        )
    print(f"  Parse consistency: {len(indices)}/{len(indices)} passed.")
    if expect_on_disk_fixed_size:
        print(f"  On-disk fixed size {exp_w}x{exp_h}: no spurious resize in dataset.")


def run_stage_2_dataset_output(
    dataset: RoadSignRetinaNetDataset,
    label_names: Dict[int, str],
    target_size: Tuple[int, int],
    num_classes: int,
    output_dir: Path,
    num_samples: int,
    seed: int,
    *,
    version: RoadSignDatasetVersion,
) -> None:
    print("\n" + "=" * 60)
    print("STAGE 2: PyTorch dataset output")
    print("=" * 60)

    print("\n  [2a] Full-dataset tensor size check")
    tensor_stats = audit_all_dataset_output_sizes(dataset, target_size)
    _print_tensor_size_audit(tensor_stats, target_size)
    _fail_if_errors(
        LabelAuditStats(errors=tensor_stats.errors),
        "Stage 2 (tensor size)",
    )

    class_id_map = dataset.class_id_map
    fixed_on_disk = requires_fixed_image_size(version)
    print("\n  [2b] Parse consistency (sampled)")
    _check_parse_consistency(
        dataset,
        class_id_map,
        target_size,
        output_dir,
        num_samples,
        seed,
        expect_on_disk_fixed_size=fixed_on_disk,
    )

    print("\n  [2c] Visual samples")
    viz_dir = output_dir / DIR_STAGE_2
    viz_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + 1)
    indices = rng.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for plot_i, idx in enumerate(indices):
        image, target = dataset[idx]
        _assert_dataset_sample(idx, image, target, target_size, num_classes)

        boxes = _boxes_tensor(target["boxes"]).numpy()
        labels = target["labels"].cpu().numpy()

        plt.figure(figsize=(8, 8))
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        _draw_boxes(plt.gca(), boxes, labels, label_names, "lime")
        sid = dataset.sample_ids[idx].replace("/", "_")
        plt.title(f"idx={idx} | {sid} | n={len(labels)}")
        plt.axis("off")
        save_path = viz_dir / f"dataset_{plot_i:03d}_{sid}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"  [saved] {save_path}")

    print("  [Stage 2] PASSED")


def denormalize(
    tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    mean_t = torch.as_tensor(mean, device=tensor.device).view(-1, 1, 1)
    std_t = torch.as_tensor(std, device=tensor.device).view(-1, 1, 1)
    return torch.clamp((tensor * std_t + mean_t) * 255, 0, 255).byte().permute(1, 2, 0).cpu().numpy()


def run_stage_3_transform(
    model: torch.nn.Module,
    dataset: RoadSignRetinaNetDataset,
    label_names: Dict[int, str],
    target_size: Tuple[int, int],
    output_dir: Path,
    *,
    is_training: bool,
    num_samples: int,
) -> None:
    mode = "train" if is_training else "eval"
    subdir = DIR_STAGE_3_TRAIN if is_training else DIR_STAGE_3_EVAL
    print(f"\n  --- model.transform ({mode}) ---")

    if is_training:
        model.train()
    else:
        model.eval()

    device = next(model.parameters()).device
    h, w = int(target_size[0]), int(target_size[1])
    subset = Subset(dataset, list(range(min(num_samples, len(dataset)))))
    loader = DataLoader(
        subset,
        batch_size=len(subset),
        shuffle=False,
        collate_fn=retinanet_collate_fn,
    )
    images, targets = next(iter(loader))

    for i, img in enumerate(images):
        if img.shape != (3, h, w):
            raise AssertionError(
                f"Pre-transform sample {i}: shape {tuple(img.shape)} != (3, {h}, {w})"
            )

    images_list = [img.to(device) for img in images]
    targets_list = [{k: v.to(device) for k, v in t.items()} for t in targets]
    n_boxes_before = [len(t["boxes"]) for t in targets_list]

    transformed_images, transformed_targets = model.transform(images_list, targets_list)

    batch_tensors = transformed_images.tensors.cpu()
    mean = model.transform.image_mean
    std = model.transform.image_std

    viz_dir = output_dir / subdir
    viz_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(images_list)):
        if batch_tensors[i].shape[-2:] != (h, w):
            raise AssertionError(
                f"Post-transform sample {i}: shape {tuple(batch_tensors[i].shape[-2:])} "
                f"!= ({h}, {w})"
            )

        t_target = transformed_targets[i]
        n_after = len(t_target["boxes"])
        if n_after != n_boxes_before[i]:
            raise AssertionError(
                f"Sample {i}: box count {n_boxes_before[i]} -> {n_after} after transform"
            )

        labels = t_target["labels"].cpu().numpy()
        boxes = t_target["boxes"].cpu().numpy()
        if not torch.equal(targets_list[i]["labels"].cpu(), t_target["labels"].cpu()):
            raise AssertionError(f"Sample {i}: labels changed after transform")

        img_np = denormalize(batch_tensors[i], mean, std)
        plt.figure(figsize=(8, 8))
        plt.imshow(img_np)
        _draw_boxes(plt.gca(), boxes, labels, label_names, "cyan")
        plt.title(f"transform {mode} | sample {i}")
        plt.axis("off")
        save_path = viz_dir / f"transform_{i:02d}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"  [saved] {save_path}")

    print(f"  [transform {mode}] PASSED")


def run_stage_3_model_transform(
    dataset: RoadSignRetinaNetDataset,
    label_names: Dict[int, str],
    target_size: Tuple[int, int],
    output_dir: Path,
    model: torch.nn.Module,
    num_samples: int,
    *,
    split_name: str,
) -> None:
    print("\n" + "=" * 60)
    print(f"STAGE 3: RetinaNet model.transform ({split_name})")
    print("=" * 60)

    run_stage_3_transform(
        model, dataset, label_names, target_size, output_dir,
        is_training=True, num_samples=num_samples,
    )
    run_stage_3_transform(
        model, dataset, label_names, target_size, output_dir,
        is_training=False, num_samples=num_samples,
    )
    print(f"  [Stage 3 {split_name}] PASSED")


def run_all_stages(
    *,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    splits: Sequence[Literal["train", "val"]],
    target_size: Tuple[int, int],
    class_id_map: Dict[int, int],
    output_dir: Path,
    num_samples: int,
    seed: int,
    anchor_config_path: Optional[Path],
    skip_stage_3: bool,
) -> None:
    seed_everything(seed)
    class_id_map = validate_retinanet_class_id_map(class_id_map)
    class_mapping = load_road_sign_class_mapping(version, dataset_hash)
    label_names = retinanet_cls_id_to_name(class_mapping, class_id_map)
    num_classes = retinanet_num_classes(class_id_map)

    train_ids, val_ids, _ = load_split_lists(version, dataset_hash, split_hash)
    train_pairs = get_train_image_label_pairs(version, dataset_hash, train_ids)
    val_pairs = get_train_image_label_pairs(version, dataset_hash, val_ids)
    all_pairs = train_pairs + val_pairs

    record = {
        "dataset_version": version,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "splits_stages_2_3": list(splits),
        "target_size": list(target_size),
        "class_id_map": class_id_map,
        "num_classes": num_classes,
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "fixed_size_dataset": requires_fixed_image_size(version),
    }
    with open(output_dir / "sanity_config.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    print("=" * 60)
    print("RETINANET DATA SANITY CHECK")
    print("=" * 60)
    print(f"  version={version}  hash={dataset_hash}  split_hash={split_hash}")
    print(f"  target_size={target_size}  num_classes={num_classes}")
    print(f"  splits (2/3)={list(splits)}  output={output_dir}")

    # Stage 1: full train+val label and on-disk size audit
    run_stage_1_raw_yolo(
        all_pairs,
        class_mapping,
        class_id_map,
        output_dir,
        num_samples,
        seed,
        version=version,
        target_size=target_size,
    )

    model: Optional[torch.nn.Module] = None
    if not skip_stage_3:
        resolved_anchor = anchor_config_path or model_retinanet_anchor_config_path(
            RETINANET_MODEL_NAME, version, dataset_hash, split_hash
        )
        if not resolved_anchor.is_file():
            raise FileNotFoundError(
                f"anchor_config.json required for stage 3: {resolved_anchor}"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        anchor_spec = load_finalized_anchor_spec(resolved_anchor)
        model = build_retinanet_from_spec(
            anchor_spec,
            num_classes=num_classes,
            img_size=target_size,
            device=device,
        )
    else:
        print("\nStage 3 skipped (--skip-stage-3).")

    for split_name in splits:
        print("\n" + "#" * 60)
        print(f"# Split: {split_name}")
        print("#" * 60)
        split_out = output_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)

        dataset = RoadSignRetinaNetDataset.from_split(
            version,
            dataset_hash,
            split_hash,
            split=split_name,
            target_size=target_size,
            class_id_map=class_id_map,
            transformations=build_dataset_transforms(train_augment=False),
        )
        run_stage_2_dataset_output(
            dataset,
            label_names,
            target_size,
            num_classes,
            split_out,
            num_samples,
            seed + (0 if split_name == "train" else 1),
            version=version,
        )

        if model is not None:
            run_stage_3_model_transform(
                dataset,
                label_names,
                target_size,
                split_out,
                model,
                num_samples,
                split_name=split_name,
            )

    print(f"\nAll stages completed for {version}/{dataset_hash}/{split_hash}.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    versions = resolve_dataset_versions(args)
    anchor_path = Path(args.anchor_config_path) if args.anchor_config_path else None

    print("=" * 60)
    print("RETINANET DATA SANITY CHECK — RUN")
    print("=" * 60)
    print(f"  Dataset versions: {versions}")
    print(f"  Splits (stages 2–3): {args.splits}")

    failures: List[str] = []

    for version in versions:
        dataset_hash = args.dataset_hash or resolve_latest_dataset_hash(version)
        if dataset_hash is None:
            msg = f"No dataset for {version!r}. Run data prep first."
            failures.append(msg)
            logger.error(msg)
            continue

        split_hash = resolve_split_hash(
            version, dataset_hash, args.split_hash, create_if_missing=False
        )
        if split_hash is None:
            msg = f"No split for {version}/{dataset_hash}. Run create_split.py first."
            failures.append(msg)
            logger.error(msg)
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
            output_dir = resolve_run_output_dir(
                version, dataset_hash, split_hash, args.output_dir
            )

            run_all_stages(
                version=version,
                dataset_hash=dataset_hash,
                split_hash=split_hash,
                splits=args.splits,
                target_size=target_size,
                class_id_map=class_id_map,
                output_dir=output_dir,
                num_samples=args.num_samples,
                seed=args.seed,
                anchor_config_path=anchor_path,
                skip_stage_3=args.skip_stage_3,
            )
        except Exception as exc:
            msg = f"{version}/{dataset_hash}: {exc}"
            failures.append(msg)
            logger.exception("Sanity check failed for %s", version)

    if failures:
        print("\n" + "=" * 60)
        print("FAILURES")
        print("=" * 60)
        for msg in failures:
            print(f"  - {msg}")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print(f"All sanity checks passed ({len(versions)} dataset version(s)).")
    print("=" * 60)


if __name__ == "__main__":
    main()
