"""
Object size vs RetinaNet confidence analysis (resized vs patch-based).

Two parts:

1. **Dataset area ratios** — object area / image area on the val split for each dataset
   version (YOLO ``w * h`` in normalized coords = fraction of the training image).

2. **Model confidence vs size** — binned mean ``best_score`` per area-ratio bin from
   trained RetinaNet checkpoints on their val loaders.

Hypothesis: patch-based training helps because objects occupy a larger fraction of the
input image, so the detector sees them at a more favorable scale.

Run from monorepo root::

    uv run python road_sign/scripts/analysis/retinanet_size_confidence_analysis.py \\
        --resized-dataset-hash HASH1 --patch-dataset-hash HASH2 --split-hash SPLIT \\
        --resized-artifact-dir road_sign/artifacts/retinanet/resized/.../EXPERIMENT \\
        --patch-artifact-dir road_sign/artifacts/retinanet/patch_based/.../EXPERIMENT

Omit artifact dirs to run dataset-only analysis (part 1).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_current = Path(__file__).resolve().parent
while _current != _current.parent:
    if (_current / "road_sign").is_dir() and (_current / "home_made_od").is_dir():
        _MONOREPO_ROOT = _current
        break
    _current = _current.parent
else:
    raise RuntimeError("Could not find monorepo root (road_sign + home_made_od).")

_RETINANET_SCRIPTS = _MONOREPO_ROOT / "road_sign" / "scripts" / "retinanet"
for _path in (_RETINANET_SCRIPTS, _MONOREPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from home_made_od.general.path_utils import (  # noqa: E402
    DATASET_VERSION_PATCH,
    DATASET_VERSION_RESIZED,
    EXPERIMENT_CONFIG_FILENAME,
    RoadSignDatasetVersion,
    get_train_image_label_pairs,
    load_split_lists,
    model_artifact_root,
    resolve_latest_dataset_hash,
    resolve_latest_split_hash,
    road_sign_artifacts_root,
)
from home_made_od.od_metrics.core_metrics import per_gt_best_pred_stats  # noqa: E402
from home_made_od.retinanet.error_analysis.retinanet_obj_ea import (  # noqa: E402
    configure_retinanet_postprocess,
)
from script_utils import (  # noqa: E402
    RETINANET_MODEL_NAME,
    build_retinanet_from_spec,
    load_finalized_anchor_spec,
    resolve_target_size,
)
from train_utils import (  # noqa: E402
    build_retinanet_transforms,
    normalize_class_id_map_config,
)
from road_sign.utils.data_utils import (  # noqa: E402
    RoadSignRetinaNetDataset,
    retinanet_collate_fn,
    retinanet_num_classes,
)

logger = logging.getLogger(__name__)

AREA_RATIO_BINS = np.array(
    [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
)
DEFAULT_AUG = {}


@dataclass
class AreaRatioStats:
    version: str
    split: str
    num_objects: int
    num_images: int
    min: float
    max: float
    mean: float
    median: float
    p90: float
    frac_gt_50pct: float
    frac_gt_20pct: float
    frac_lt_1pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GtConfidenceRecord:
    area_ratio: float
    best_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Object area ratio + RetinaNet confidence analysis (resized vs patch)."
    )
    parser.add_argument("--resized-dataset-hash", default=None)
    parser.add_argument("--patch-dataset-hash", default=None)
    parser.add_argument("--split-hash", default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--resized-artifact-dir", type=Path, default=None)
    parser.add_argument("--patch-artifact-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: road_sign/artifacts/retinanet/size_confidence_analysis/",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--inference-score-thresh", type=float, default=0.01)
    parser.add_argument("--inference-nms-thresh", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-confidence", action="store_true")
    return parser.parse_args()


def yolo_area_ratios_from_label(label_path: Path) -> List[float]:
    """Object area / image area from normalized YOLO ``w * h``."""
    ratios: List[float] = []
    if not label_path.is_file():
        return ratios
    with open(label_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 5:
                continue
            nw = float(parts[3])
            nh = float(parts[4])
            ratios.append(nw * nh)
    return ratios


def collect_area_ratios_for_split(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    split: str,
) -> Tuple[List[float], int]:
    train_ids, val_ids, _ = load_split_lists(version, dataset_hash, split_hash)
    unit_ids = train_ids if split == "train" else val_ids
    pairs = get_train_image_label_pairs(version, dataset_hash, unit_ids)

    ratios: List[float] = []
    for _img, lbl in pairs:
        ratios.extend(yolo_area_ratios_from_label(Path(lbl)))
    return ratios, len(pairs)


def summarize_area_ratios(
    ratios: Sequence[float],
    *,
    version: str,
    split: str,
    num_images: int,
) -> AreaRatioStats:
    arr = np.asarray(ratios, dtype=np.float64)
    if arr.size == 0:
        return AreaRatioStats(
            version=version,
            split=split,
            num_objects=0,
            num_images=num_images,
            min=0.0,
            max=0.0,
            mean=0.0,
            median=0.0,
            p90=0.0,
            frac_gt_50pct=0.0,
            frac_gt_20pct=0.0,
            frac_lt_1pct=0.0,
        )
    return AreaRatioStats(
        version=version,
        split=split,
        num_objects=int(arr.size),
        num_images=num_images,
        min=float(arr.min()),
        max=float(arr.max()),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        p90=float(np.percentile(arr, 90)),
        frac_gt_50pct=float((arr > 0.5).mean()),
        frac_gt_20pct=float((arr > 0.2).mean()),
        frac_lt_1pct=float((arr < 0.01).mean()),
    )


def plot_area_ratio_histograms(
    resized_ratios: Sequence[float],
    patch_ratios: Sequence[float],
    output_path: Path,
    *,
    split: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, ratios, title in (
        (axes[0], resized_ratios, "Resized dataset"),
        (axes[1], patch_ratios, "Patch-based dataset"),
    ):
        data = np.clip(np.asarray(ratios), 0, 1.0)
        ax.hist(
            data,
            bins=np.arange(0, 1.05, 0.02),
            color="steelblue",
            edgecolor="white",
            alpha=0.85,
        )
        ax.set_title(title)
        ax.set_xlabel("Object area / image area")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1.0)
        if data.size:
            ax.axvline(np.median(data), color="crimson", ls="--", label=f"median={np.median(data):.3f}")
            ax.legend()

    fig.suptitle(f"Object area ratio distribution ({split} split)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_area_ratio_overlay(
    resized_ratios: Sequence[float],
    patch_ratios: Sequence[float],
    output_path: Path,
    *,
    split: str,
) -> None:
    plt.figure(figsize=(9, 6))
    bins = np.arange(0, 1.05, 0.02)
    plt.hist(
        np.clip(resized_ratios, 0, 1),
        bins=bins,
        alpha=0.55,
        label=f"resized (n={len(resized_ratios)})",
        color="tab:blue",
    )
    plt.hist(
        np.clip(patch_ratios, 0, 1),
        bins=bins,
        alpha=0.55,
        label=f"patch (n={len(patch_ratios)})",
        color="tab:orange",
    )
    plt.xlabel("Object area / image area")
    plt.ylabel("Count")
    plt.title(f"Resized vs patch — area ratio ({split})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def resolve_experiment_dir(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    artifact_dir: Optional[Path],
) -> Optional[Path]:
    if artifact_dir is not None:
        path = artifact_dir.resolve()
        if not (path / EXPERIMENT_CONFIG_FILENAME).is_file():
            raise FileNotFoundError(f"Missing {EXPERIMENT_CONFIG_FILENAME} in {path}")
        return path

    root = model_artifact_root(RETINANET_MODEL_NAME, version, dataset_hash, split_hash)
    if not root.is_dir():
        return None
    candidates = [
        p
        for p in root.iterdir()
        if p.is_dir() and (p / EXPERIMENT_CONFIG_FILENAME).is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _boxes_tensor(boxes: Any) -> torch.Tensor:
    if hasattr(boxes, "data"):
        return boxes.data.detach().float()
    if hasattr(boxes, "detach"):
        return boxes.detach().float()
    return torch.as_tensor(boxes, dtype=torch.float32)


def gt_area_ratios_from_boxes(boxes: torch.Tensor, target_size: Tuple[int, int]) -> List[float]:
    if boxes.numel() == 0:
        return []
    th, tw = target_size
    img_area = float(th * tw)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    return (areas / img_area).tolist()


def collect_gt_confidence_records(
    artifact_dir: Path,
    version: RoadSignDatasetVersion,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    inference_score_thresh: float,
    inference_nms_thresh: float,
) -> List[GtConfidenceRecord]:
    with open(artifact_dir / EXPERIMENT_CONFIG_FILENAME, encoding="utf-8") as handle:
        run_config = json.load(handle)

    dataset_hash = run_config["dataset_hash"]
    split_hash = run_config["split_hash"]
    class_id_map = normalize_class_id_map_config(run_config["class_id_map"])
    target_size = resolve_target_size(version, dataset_hash, train_config=run_config)
    num_classes = int(run_config.get("num_classes", retinanet_num_classes(class_id_map)))

    # anchor_path = run_config.get("anchor_config_path")
    # if not anchor_path:
    #     raise KeyError(f"{artifact_dir}: missing anchor_config_path")
    anchor_path = Path(os.path.join(artifact_dir, "anchor_config.json"))

    anchor_spec = load_finalized_anchor_spec(anchor_path)

    checkpoint = artifact_dir / "checkpoints" / "best_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = build_retinanet_from_spec(
        anchor_spec,
        num_classes=num_classes,
        img_size=target_size,
        device=device,
        checkpoint_path=checkpoint,
        freeze_backbone_layers=0,
        log_build=False,
    )
    model.eval()
    configure_retinanet_postprocess(
        model,
        score_thresh=inference_score_thresh,
        nms_thresh=inference_nms_thresh,
    )

    val_ds = RoadSignRetinaNetDataset.from_split(
        version,
        dataset_hash,
        split_hash,
        split=split,
        target_size=target_size,
        class_id_map=class_id_map,
        transformations=build_retinanet_transforms(DEFAULT_AUG, train=False),
    )
    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=retinanet_collate_fn,
        pin_memory=device.type == "cuda",
    )

    records: List[GtConfidenceRecord] = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=f"Inference {version}", leave=False):
            images_dev = [img.to(device) for img in images]
            outputs = model(images_dev)

            for target, output in zip(targets, outputs):
                gt_boxes = _boxes_tensor(target["boxes"]).to(device)
                if gt_boxes.numel() == 0:
                    continue

                area_ratios = gt_area_ratios_from_boxes(gt_boxes.cpu(), target_size)
                gt_labels = target["labels"]

                pred_boxes = output["boxes"]
                pred_scores = output["scores"]
                pred_labels = output["labels"]

                per_gt = per_gt_best_pred_stats(
                    gt_boxes,
                    pred_boxes,
                    pred_scores,
                    pred_labels=pred_labels,
                    gt_labels=gt_labels,
                    iou_overlap_min=0.1,
                )

                for rec, area_ratio in zip(per_gt, area_ratios):
                    records.append(
                        GtConfidenceRecord(
                            area_ratio=float(area_ratio),
                            best_score=float(rec.best_score),
                        )
                    )
    return records


def binned_confidence_summary(
    records: Sequence[GtConfidenceRecord],
) -> List[Dict[str, Any]]:
    if not records:
        return []

    ratios = np.array([r.area_ratio for r in records])
    scores = np.array([r.best_score for r in records])

    rows: List[Dict[str, Any]] = []
    for lo, hi in zip(AREA_RATIO_BINS[:-1], AREA_RATIO_BINS[1:]):
        mask = (ratios >= lo) & (ratios < hi)
        count = int(mask.sum())
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": count,
                "mean_best_score": (
                    float(scores[mask].mean()) if count > 0 else None
                ),
            }
        )
    return rows


def plot_confidence_vs_area(
    records: Sequence[GtConfidenceRecord],
    output_path: Path,
    *,
    title: str,
) -> None:
    if not records:
        return

    binned = binned_confidence_summary(records)
    bin_centers = [
        0.5 * (row["bin_lo"] + row["bin_hi"])
        for row in binned
        if row["count"] > 0
    ]
    mean_scores = [row["mean_best_score"] for row in binned if row["count"] > 0]
    if not bin_centers:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_centers, mean_scores, "o-", color="tab:blue")
    ax.set_xlabel("Object area / image area (bin center)")
    ax.set_ylabel("Mean best score (IoU ≥ 0.1)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confidence_comparison(
    resized_binned: List[Dict[str, Any]],
    patch_binned: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for binned, label, color in (
        (resized_binned, "resized", "tab:blue"),
        (patch_binned, "patch", "tab:orange"),
    ):
        centers = [
            0.5 * (r["bin_lo"] + r["bin_hi"])
            for r in binned
            if r["count"] and r["mean_best_score"] is not None
        ]
        means = [
            r["mean_best_score"]
            for r in binned
            if r["count"] and r["mean_best_score"] is not None
        ]
        if centers:
            ax.plot(centers, means, "o-", label=label, color=color)

    ax.set_xlabel("Object area / image area (bin center)")
    ax.set_ylabel("Mean best score (IoU ≥ 0.1)")
    ax.set_title("Resized vs patch RetinaNet — binned mean best score")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def default_output_dir(
    resized_hash: str,
    patch_hash: str,
    split_hash: str,
) -> Path:
    return (
        road_sign_artifacts_root()
        / RETINANET_MODEL_NAME
        / "size_confidence_analysis"
        / f"resized_{resized_hash[:8]}"
        / f"patch_{patch_hash[:8]}"
        / split_hash
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    resized_hash = args.resized_dataset_hash or resolve_latest_dataset_hash(
        DATASET_VERSION_RESIZED
    )
    patch_hash = args.patch_dataset_hash or resolve_latest_dataset_hash(
        DATASET_VERSION_PATCH
    )
    if not resized_hash or not patch_hash:
        raise SystemExit("Could not resolve dataset hashes. Pass them explicitly.")

    split_hash = args.split_hash
    if split_hash is None:
        split_hash = (
            resolve_latest_split_hash(DATASET_VERSION_RESIZED, resized_hash)
            or resolve_latest_split_hash(DATASET_VERSION_PATCH, patch_hash)
        )
    if split_hash is None:
        raise SystemExit("No split hash found. Pass --split-hash.")

    output_dir = args.output_dir or default_output_dir(resized_hash, patch_hash, split_hash)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 60)
    print("OBJECT SIZE & CONFIDENCE ANALYSIS")
    print("=" * 60)
    print(f"  resized hash: {resized_hash}")
    print(f"  patch hash:   {patch_hash}")
    print(f"  split hash:   {split_hash} ({args.split})")
    print(f"  output:       {output_dir}")

    # --- Part 1: dataset area ratios ---
    resized_ratios, resized_n_img = collect_area_ratios_for_split(
        DATASET_VERSION_RESIZED, resized_hash, split_hash, args.split
    )
    patch_ratios, patch_n_img = collect_area_ratios_for_split(
        DATASET_VERSION_PATCH, patch_hash, split_hash, args.split
    )

    resized_stats = summarize_area_ratios(
        resized_ratios,
        version=DATASET_VERSION_RESIZED,
        split=args.split,
        num_images=resized_n_img,
    )
    patch_stats = summarize_area_ratios(
        patch_ratios,
        version=DATASET_VERSION_PATCH,
        split=args.split,
        num_images=patch_n_img,
    )

    print("\n--- Area ratio summary (object area / image area) ---")
    for stats in (resized_stats, patch_stats):
        print(f"\n  [{stats.version}] objects={stats.num_objects} images={stats.num_images}")
        print(f"    median={stats.median:.4f}  mean={stats.mean:.4f}  p90={stats.p90:.4f}")
        print(f"    >50%: {stats.frac_gt_50pct:.1%}  >20%: {stats.frac_gt_20pct:.1%}  <1%: {stats.frac_lt_1pct:.1%}")

    plot_area_ratio_histograms(
        resized_ratios,
        patch_ratios,
        output_dir / f"area_ratio_hist_{args.split}.png",
        split=args.split,
    )
    plot_area_ratio_overlay(
        resized_ratios,
        patch_ratios,
        output_dir / f"area_ratio_overlay_{args.split}.png",
        split=args.split,
    )

    summary: Dict[str, Any] = {
        "split_hash": split_hash,
        "split": args.split,
        "resized_dataset_hash": resized_hash,
        "patch_dataset_hash": patch_hash,
        "area_ratio_stats": {
            DATASET_VERSION_RESIZED: resized_stats.to_dict(),
            DATASET_VERSION_PATCH: patch_stats.to_dict(),
        },
    }

    # --- Part 2: model confidence vs size ---
    if not args.skip_confidence:
        resized_exp = resolve_experiment_dir(
            DATASET_VERSION_RESIZED,
            resized_hash,
            split_hash,
            args.resized_artifact_dir,
        )
        patch_exp = resolve_experiment_dir(
            DATASET_VERSION_PATCH,
            patch_hash,
            split_hash,
            args.patch_artifact_dir,
        )

        if resized_exp is None and patch_exp is None:
            logger.warning(
                "No experiment dirs found — skipping confidence analysis. "
                "Pass --resized-artifact-dir / --patch-artifact-dir."
            )
        else:
            resized_records: List[GtConfidenceRecord] = []
            patch_records: List[GtConfidenceRecord] = []

            if resized_exp is not None:
                logger.info("Resized model: %s", resized_exp)
                resized_records = collect_gt_confidence_records(
                    resized_exp,
                    DATASET_VERSION_RESIZED,
                    split=args.split,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                    inference_score_thresh=args.inference_score_thresh,
                    inference_nms_thresh=args.inference_nms_thresh,
                )
                plot_confidence_vs_area(
                    resized_records,
                    output_dir / f"confidence_vs_area_resized_{args.split}.png",
                    title=f"Resized RetinaNet — val ({args.split})",
                )

            if patch_exp is not None:
                logger.info("Patch model: %s", patch_exp)
                patch_records = collect_gt_confidence_records(
                    patch_exp,
                    DATASET_VERSION_PATCH,
                    split=args.split,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                    inference_score_thresh=args.inference_score_thresh,
                    inference_nms_thresh=args.inference_nms_thresh,
                )
                plot_confidence_vs_area(
                    patch_records,
                    output_dir / f"confidence_vs_area_patch_{args.split}.png",
                    title=f"Patch RetinaNet — val ({args.split})",
                )

            if resized_records and patch_records:
                resized_binned = binned_confidence_summary(resized_records)
                patch_binned = binned_confidence_summary(patch_records)
                plot_confidence_comparison(
                    resized_binned,
                    patch_binned,
                    output_dir / f"confidence_comparison_{args.split}.png",
                )
                summary["confidence_analysis"] = {
                    "resized_artifact_dir": str(resized_exp) if resized_exp else None,
                    "patch_artifact_dir": str(patch_exp) if patch_exp else None,
                    "resized": {
                        "num_gt": len(resized_records),
                        "mean_best_score": float(
                            np.mean([r.best_score for r in resized_records])
                        ),
                        "binned": resized_binned,
                    },
                    "patch": {
                        "num_gt": len(patch_records),
                        "mean_best_score": float(
                            np.mean([r.best_score for r in patch_records])
                        ),
                        "binned": patch_binned,
                    },
                }

                print("\n--- Confidence summary (val) ---")
                for name, recs in (("resized", resized_records), ("patch", patch_records)):
                    print(
                        f"  {name}: GT={len(recs)} "
                        f"mean_best_score={np.mean([r.best_score for r in recs]):.3f}"
                    )

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSummary written to {summary_path}")
    print(f"Plots saved under {output_dir}")


if __name__ == "__main__":
    main()
