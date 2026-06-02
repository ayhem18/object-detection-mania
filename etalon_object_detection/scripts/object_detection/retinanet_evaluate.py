"""
RetinaNet objectness evaluation on the validation split.

Mirrors ``retinanet_inference.py`` for path resolution, config loading, val split,
and checkpoint loading. Outputs live under a parameter-encoded subfolder with plots
and extreme-case visualizations (not scattered JSON dumps).

Layer A uses ``model(images)`` with caller-controlled ``score_thresh`` and
``nms_thresh`` (use ``nms_thresh=1.0`` to effectively disable NMS suppression).
Layer B uses head logits + anchors (pre-NMS) via the RetinaNet adapter below.
"""

from __future__ import annotations

import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from torchvision.ops import box_iou
from typing import Any, Dict, List, Optional

from dl_lib.etalon_object_detection.modules.od_metrics.core_metrics import PerGtPredStats
from dl_lib.etalon_object_detection.modules.od_metrics.grid_metrics import PerGtGridStats
from dl_lib.etalon_object_detection.scripts.object_detection.retinanet_ea_scripts.utils import (
    assert_checkpoint_exists,
    build_objectness_ea_output_dir,
    build_val_dataloader,
    default_conf_thresholds,
    get_eval_device,
    image_stem_from_sample_path,
    load_experiment_config,
    resolve_experiment_paths,
    validate_objectness_ea_config,
)
from dl_lib.etalon_object_detection.modules.od_metrics.objectness_ea import (
    aggregate_objectness_reports,
    build_objectness_report,
)
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_detector import (
    build_dl_lab_etalon_detector_from_trained_weights,
)
from dl_lib.etalon_object_detection.modules.retinanet.error_analysis.grid_viz import (
    GridExtremeCandidate,
    save_grid_metric_extremes,
)
from dl_lib.etalon_object_detection.modules.retinanet.error_analysis.retinanet_obj_ea import (
    configure_retinanet_postprocess,
    extract_retinanet_layer_b_bundles,
    gt_metadata_from_boxes,
    infer_foreground_label_ids,
)


# =============================================================================
# Evaluation config — edit here
# =============================================================================

EVAL_CONFIG: Dict[str, Any] = {
    # Experiment discovery (same as retinanet_inference.py)
    "dataset_hash": "latest",
    "split_hash": "latest",
    "exp_hash": "latest",
    # Required evaluation parameters
    "iou_overlap_min": 0.1,
    "iou_eval": 0.5,
    "nms_thresh": 1.0,  # 1.0 => effectively no NMS suppression
    "score_thresh": 0.01,
    "conf_threshold_for_breakdown": 0.3,
    "fg_iou_threshold": 0.5,
    "k_extremes": 8,
    # Optional
    "run_layer_b": True,
    "run_grid_metrics": True,
    "batch_size": 4,
    "conf_thresholds": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    "foreground_label_ids": None,
}


# =============================================================================
# Visualization
# =============================================================================


@dataclass
class GtCoverageVizSample:
    """One GT instance tied to its image for extreme-case panels."""

    best_score: float
    best_iou: float
    gt_idx: int
    image: torch.Tensor
    target: Dict[str, torch.Tensor]
    pred: Dict[str, torch.Tensor]
    img_stem: str


def save_gt_coverage_panel(
    sample: GtCoverageVizSample,
    label_map: Dict[int, str],
    save_path: Path,
    title_prefix: str,
) -> None:
    """
    Draw one GT (green) and exactly two candidate preds for Layer A debugging:

    - **Highest confidence** (blue): argmax score over all image predictions.
    - **Highest IoU** (red): argmax IoU with this GT box.

    When both refer to the same prediction, a single box is drawn with a combined label.
    """
    img_np = sample.image.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    gt_box = sample.target["boxes"][sample.gt_idx].cpu().numpy()
    x1, y1, x2, y2 = gt_box.astype(int)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
    gt_lbl = int(sample.target["labels"][sample.gt_idx].item())
    cv2.putText(
        img_bgr,
        f"GT#{sample.gt_idx} {label_map.get(gt_lbl, gt_lbl)}",
        (x1, max(y1 - 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    p_boxes = sample.pred["boxes"].cpu()
    p_scores = sample.pred["scores"].cpu()
    p_labels = sample.pred["labels"].cpu()

    if p_boxes.numel() > 0:
        gt_t = sample.target["boxes"][sample.gt_idx].unsqueeze(0)
        ious = box_iou(gt_t, p_boxes)[0]
        best_iou_idx = int(ious.argmax().item())
        best_score_idx = int(p_scores.argmax().item())

        panels = [
            (best_score_idx, (255, 128, 0), "max conf"),   # blue-ish BGR
            (best_iou_idx, (0, 0, 255), "max IoU"),       # red BGR
        ]
        drawn: set[int] = set()
        y_offset = 0
        for pred_idx, color, tag in panels:
            if pred_idx in drawn:
                continue
            drawn.add(pred_idx)
            box = p_boxes[pred_idx].numpy()
            bx1, by1, bx2, by2 = box.astype(int)
            cv2.rectangle(img_bgr, (bx1, by1), (bx2, by2), color, 2)
            lbl = int(p_labels[pred_idx].item())
            score = float(p_scores[pred_idx].item())
            iou_val = float(ious[pred_idx].item())
            label_line = (
                f"{tag}: {label_map.get(lbl, lbl)} sc={score:.2f} iou={iou_val:.2f}"
            )
            cv2.putText(
                img_bgr,
                label_line,
                (bx1, min(by2 + 18 + y_offset, img_bgr.shape[0] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
            y_offset += 16

        if best_score_idx == best_iou_idx:
            bx1, by1, bx2, by2 = p_boxes[best_score_idx].numpy().astype(int)
            cv2.putText(
                img_bgr,
                "same pred: max conf & max IoU",
                (bx1, min(by2 + 34, img_bgr.shape[0] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    title = (
        f"{title_prefix} | {sample.img_stem} | GT#{sample.gt_idx} | "
        f"best_score={sample.best_score:.3f} best_iou={sample.best_iou:.3f}"
    )
    cv2.putText(
        img_bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
    )
    cv2.putText(
        img_bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img_bgr)


def plot_layer_a_distributions(
    aggregated: Dict[str, Any],
    per_gt_records: List[PerGtPredStats],
    output_dir: Path,
    iou_eval: float,
) -> None:
    """Save Layer A distribution plots (recall curve + score/IoU histograms)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    la = aggregated["layer_a"]

    # 1) Recall vs confidence
    curve = la["recall_curve"]
    taus = sorted(curve.keys(), key=float)
    recalls = [curve[t]["recall"] * 100.0 for t in taus]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([float(t) for t in taus], recalls, marker="o", linewidth=2)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel(f"GT recall (%) @ IoU>={iou_eval}")
    ax.set_title("Layer A — Objectness recall vs confidence")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(output_dir / "recall_vs_confidence.png", dpi=150)
    plt.close(fig)

    # 2) Best-score histogram
    scores = [r.best_score for r in per_gt_records]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("Best prediction score at GT (iou_overlap_min applied)")
    ax.set_ylabel("Count")
    ax.set_title("Layer A — best_score per GT")
    fig.tight_layout()
    fig.savefig(output_dir / "best_score_histogram.png", dpi=150)
    plt.close(fig)

    # 3) Best-IoU histogram (localization ceiling)
    ious = [r.best_iou for r in per_gt_records]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ious, bins=30, color="darkorange", edgecolor="white")
    ax.axvline(iou_eval, color="red", linestyle="--", label=f"iou_eval={iou_eval}")
    ax.set_xlabel("Best IoU per GT (oracle, any pred)")
    ax.set_ylabel("Count")
    ax.set_title("Layer A — best_iou per GT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "best_iou_histogram.png", dpi=150)
    plt.close(fig)

    # 4) Failure breakdown bar chart
    bd = la["failure_breakdown"]
    names = list(bd["counts"].keys())
    counts = [bd["counts"][n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, counts, color=["seagreen", "indianred", "goldenrod", "slategray"])
    ax.set_ylabel("GT count")
    ax.set_title("Layer A — failure buckets")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "failure_breakdown.png", dpi=150)
    plt.close(fig)

    # 5) Score vs IoU scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(ious, scores, alpha=0.35, s=12)
    ax.axhline(bd["thresholds"]["conf_threshold"], color="blue", linestyle="--", alpha=0.6)
    ax.axvline(iou_eval, color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("best_iou")
    ax.set_ylabel("best_score")
    ax.set_title("Layer A — per-GT coverage")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "best_score_vs_best_iou.png", dpi=150)
    plt.close(fig)


def plot_layer_b_grid_distributions(
    grid_summary: Dict[str, Any],
    per_gt_records: List[PerGtGridStats],
    output_dir: Path,
) -> None:
    """Save Layer B grid / prior-cell distribution plots."""
    grid_dir = output_dir / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    coverages = [r.gt_area_coverage for r in per_gt_records]
    n_overlap = [r.num_overlapping_priors for r in per_gt_records]
    cell_recalls = [r.cell_recall_in_gt for r in per_gt_records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(coverages, bins=30, color="teal", edgecolor="white")
    ax.axvline(
        grid_summary.get("mean_gt_area_coverage", 0.0),
        color="red",
        linestyle="--",
        label="mean",
    )
    ax.set_xlabel("GT area covered (union of best-anchor-per-cell priors)")
    ax.set_ylabel("Count")
    ax.set_title("Grid — gt_area_coverage per GT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(grid_dir / "gt_area_coverage_histogram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(n_overlap, bins=30, color="cadetblue", edgecolor="white")
    ax.set_xlabel("Num priors with IoU >= grid threshold")
    ax.set_ylabel("Count")
    ax.set_title("Grid — overlapping priors per GT")
    fig.tight_layout()
    fig.savefig(grid_dir / "num_overlapping_priors_histogram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cell_recalls, bins=30, color="olive", edgecolor="white")
    ax.set_xlabel("cells_touched / cells_in_gt_bbox (best IoU anchor per cell)")
    ax.set_ylabel("Count")
    ax.set_title("Grid — cell recall within GT bbox")
    fig.tight_layout()
    fig.savefig(grid_dir / "cell_recall_in_gt_histogram.png", dpi=150)
    plt.close(fig)


def save_extreme_gt_coverage_panels(
    viz_samples: List[GtCoverageVizSample],
    k_extremes: int,
    output_dir: Path,
    label_map: Dict[int, str],
) -> None:
    """Save k best and k worst GT by best_score (objectness coverage)."""
    if not viz_samples:
        return

    ranked = sorted(viz_samples, key=lambda s: s.best_score)
    worst = ranked[:k_extremes]
    best = ranked[-k_extremes:][::-1]

    extremes_dir = output_dir / "extremes"
    for i, sample in enumerate(worst):
        save_gt_coverage_panel(
            sample,
            label_map,
            extremes_dir / f"worst_{i:02d}_score{sample.best_score:.3f}_iou{sample.best_iou:.3f}.png",
            "WORST objectness coverage",
        )
    for i, sample in enumerate(best):
        save_gt_coverage_panel(
            sample,
            label_map,
            extremes_dir / f"best_{i:02d}_score{sample.best_score:.3f}_iou{sample.best_iou:.3f}.png",
            "BEST objectness coverage",
        )


def print_objectness_summary(summary: Dict[str, Any]) -> None:
    la = summary["layer_a"]
    print("\n" + "=" * 60)
    print("OBJECTNESS EVAL — Layer A")
    print("=" * 60)
    print(f"Images: {summary['num_images']}  |  GT boxes: {la['num_gt']}")
    for tau, row in sorted(la["recall_curve"].items(), key=lambda x: float(x[0])):
        print(f"  Recall @ conf>={float(tau):<5}: {row['recall'] * 100:6.2f}%")
    bd = la["failure_breakdown"]
    print(f"\nFailure breakdown (conf={bd['thresholds']['conf_threshold']}, "
          f"iou_eval={bd['thresholds']['iou_eval']}):")
    for name, frac in bd["fractions"].items():
        print(f"  {name:22s}: {bd['counts'][name]:5d}  ({frac * 100:5.1f}%)")
    if "layer_b" in summary:
        lb = summary["layer_b"]
        dist = lb["anchor_score_distribution"]
        print(f"\nLayer B — zero positive anchors: "
              f"{dist['fraction_gt_with_zero_positive_anchors'] * 100:.1f}%")
    if "layer_b_grid" in summary:
        g = summary["layer_b_grid"]
        print("\nLayer B grid — prior / cell coverage")
        print(f"  GT boxes: {g['num_gt']}")
        print(f"  mean_gt_area_coverage:     {g['mean_gt_area_coverage']:.3f}")
        print(f"  mean_num_overlapping_priors: {g['mean_num_overlapping_priors']:.1f}")
        print(f"  mean_cell_recall_in_gt:    {g['mean_cell_recall_in_gt']:.3f}")
        print(f"  mean_max_prior_iou:        {g['mean_max_prior_iou']:.3f}")


# =============================================================================
# Main
# =============================================================================


def run_objectness_evaluation(config: Dict[str, Any]) -> Dict[str, Any]:
    validate_objectness_ea_config(config)

    dataset_hash = config["dataset_hash"]
    split_hash = config["split_hash"]
    exp_hash = config["exp_hash"]
    iou_overlap_min = config["iou_overlap_min"]
    iou_eval = config["iou_eval"]
    nms_thresh = config["nms_thresh"]
    score_thresh = config["score_thresh"]
    conf_threshold_for_breakdown = config["conf_threshold_for_breakdown"]
    fg_iou_threshold = config["fg_iou_threshold"]
    k_extremes = config["k_extremes"]
    run_layer_b = config.get("run_layer_b", True)
    batch_size = config.get("batch_size", 4)
    conf_thresholds = config.get("conf_thresholds", default_conf_thresholds())
    foreground_label_ids = config.get("foreground_label_ids")
    run_grid_metrics = config.get("run_grid_metrics", run_layer_b)
    grid_iou_threshold = config.get("grid_iou_threshold", fg_iou_threshold)

    paths = resolve_experiment_paths(dataset_hash, split_hash, exp_hash)
    output_dir = build_objectness_ea_output_dir(
        paths["artifacts_root"],
        iou_overlap_min=iou_overlap_min,
        iou_eval=iou_eval,
        nms_thresh=nms_thresh,
        score_thresh=score_thresh,
        conf_threshold_for_breakdown=conf_threshold_for_breakdown,
        fg_iou_threshold=fg_iou_threshold,
        run_layer_b=run_layer_b,
        k_extremes=k_extremes,
        run_grid_metrics=run_grid_metrics,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {paths['exp_dir'].name}")
    print(f"Output:   {output_dir}")

    experiment_config = load_experiment_config(paths)
    assert_checkpoint_exists(paths)
    device = get_eval_device()

    val_dataset, val_loader = build_val_dataloader(
        paths, experiment_config, batch_size
    )
    print(f"Validation samples: {len(val_dataset)}")

    model = build_dl_lab_etalon_detector_from_trained_weights(
        checkpoint_path=str(paths["checkpoint"]),
        manifest_path=str(paths["anchor_config"]),
        img_size=experiment_config["model_params"]["img_size"],
        device=device,
    )
    configure_retinanet_postprocess(model, score_thresh=score_thresh, nms_thresh=nms_thresh)
    print(
        f"Post-process: score_thresh={score_thresh}, nms_thresh={nms_thresh}, "
        f"topk={model.topk_candidates}, max_dets={model.detections_per_img}"
    )
    model.eval()

    label_map = experiment_config.get("cls_id_2_cls_name", {1: "etalon", 0: "background"})
    fg_ids = infer_foreground_label_ids(
        model.head.classification_head.num_classes, foreground_label_ids
    )

    per_image_reports: List[Dict[str, Any]] = []
    viz_samples: List[GtCoverageVizSample] = []
    grid_viz_candidates: List[GridExtremeCandidate] = []
    img_idx = 0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Objectness eval"):
            images_device = [img.to(device) for img in images]
            outputs = model(images_device)

            layer_b_bundles = None
            if run_layer_b:
                layer_b_bundles = extract_retinanet_layer_b_bundles(
                    model,
                    images,
                    targets,
                    device,
                    anchor_manifest_path=str(paths["anchor_config"]),
                )

            for i in range(len(images)):
                gt_boxes = targets[i]["boxes"]
                if gt_boxes.numel() == 0:
                    img_idx += 1
                    continue

                img_stem = image_stem_from_sample_path(
                    val_dataset.samples[img_idx]["img_path"]
                )

                pred_cpu = {k: v.cpu() for k, v in outputs[i].items()}
                gt_meta = gt_metadata_from_boxes(gt_boxes)

                report_kwargs: Dict[str, Any] = {
                    "gt_boxes": gt_boxes,
                    "pred_boxes": pred_cpu["boxes"],
                    "pred_scores": pred_cpu["scores"],
                    "pred_labels": pred_cpu["labels"],
                    "gt_labels": targets[i].get("labels"),
                    "foreground_label_ids": fg_ids,
                    "iou_overlap_min": iou_overlap_min,
                    "iou_eval": iou_eval,
                    "conf_thresholds": conf_thresholds,
                    "conf_threshold_for_breakdown": conf_threshold_for_breakdown,
                    "gt_metadata": gt_meta,
                }
                if run_layer_b and layer_b_bundles is not None:
                    bundle = layer_b_bundles[i]
                    report_kwargs.update(
                        {
                            "anchor_boxes": bundle.anchor_boxes,
                            "anchor_scores": bundle.anchor_scores,
                            "positive_anchor_indices_per_gt": bundle.positive_anchor_indices_per_gt,
                            "fg_iou_threshold": fg_iou_threshold,
                        }
                    )
                    if run_grid_metrics:
                        report_kwargs.update(
                            {
                                "prior_to_cell": bundle.prior_to_cell,
                                "level_grids": bundle.level_grids,
                                "grid_iou_threshold": grid_iou_threshold,
                            }
                        )

                report = build_objectness_report(**report_kwargs)
                per_image_reports.append(report)

                for rec in report["layer_a"]["per_gt_records"]:
                    viz_samples.append(
                        GtCoverageVizSample(
                            best_score=rec.best_score,
                            best_iou=rec.best_iou,
                            gt_idx=rec.gt_idx,
                            image=images[i],
                            target=targets[i],
                            pred=pred_cpu,
                            img_stem=img_stem,
                        )
                    )

                if (
                    run_grid_metrics
                    and run_layer_b
                    and layer_b_bundles is not None
                    and report.get("layer_b_grid")
                ):
                    bundle = layer_b_bundles[i]
                    for grid_rec in report["layer_b_grid"]["per_gt_records"]:
                        if grid_rec.per_level is None:
                            continue
                        grid_viz_candidates.append(
                            GridExtremeCandidate(
                                image=images[i],
                                gt_idx=grid_rec.gt_idx,
                                gt_box=gt_boxes[grid_rec.gt_idx],
                                anchor_boxes=bundle.anchor_boxes,
                                prior_to_cell=bundle.prior_to_cell,
                                level_grids=bundle.level_grids,
                                per_level=grid_rec.per_level,
                                img_stem=img_stem,
                                grid_iou_threshold=grid_iou_threshold,
                                gt_area_coverage=grid_rec.gt_area_coverage,
                                num_overlapping_priors=float(
                                    grid_rec.num_overlapping_priors
                                ),
                                cell_recall_in_gt=grid_rec.cell_recall_in_gt,
                            )
                        )

                img_idx += 1

    if not per_image_reports:
        print("No GT boxes in validation split.")
        return {}

    aggregated = aggregate_objectness_reports(
        per_image_reports,
        conf_thresholds=conf_thresholds,
        iou_eval=iou_eval,
        conf_threshold_for_breakdown=conf_threshold_for_breakdown,
    )
    flat_records: List[PerGtPredStats] = []
    for rep in per_image_reports:
        flat_records.extend(rep["layer_a"]["per_gt_records"])

    summary = {
        "num_images": aggregated["num_images"],
        "layer_a": aggregated["layer_a"],
        "eval_params": {
            "iou_overlap_min": iou_overlap_min,
            "iou_eval": iou_eval,
            "nms_thresh": nms_thresh,
            "score_thresh": score_thresh,
            "conf_threshold_for_breakdown": conf_threshold_for_breakdown,
            "fg_iou_threshold": fg_iou_threshold,
            "k_extremes": k_extremes,
            "run_layer_b": run_layer_b,
            "run_grid_metrics": run_grid_metrics,
            "grid_iou_threshold": grid_iou_threshold,
            "batch_size": batch_size,
        },
    }
    if "layer_b" in aggregated:
        summary["layer_b"] = aggregated["layer_b"]
    if "layer_b_grid" in aggregated:
        summary["layer_b_grid"] = aggregated["layer_b_grid"]

    flat_grid_records: List[PerGtGridStats] = []
    for rep in per_image_reports:
        if rep.get("layer_b_grid") and rep["layer_b_grid"].get("per_gt_records"):
            flat_grid_records.extend(rep["layer_b_grid"]["per_gt_records"])

    # --- Visual outputs ---
    plot_layer_a_distributions(aggregated, flat_records, output_dir, iou_eval=iou_eval)
    if flat_grid_records and "layer_b_grid" in summary:
        plot_layer_b_grid_distributions(
            summary["layer_b_grid"], flat_grid_records, output_dir
        )
        save_grid_metric_extremes(grid_viz_candidates, k_extremes, output_dir)
    save_extreme_gt_coverage_panels(viz_samples, k_extremes, output_dir, label_map)

    # Single compact JSON for reproducibility (not per-image dumps)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print_objectness_summary(summary)
    print(f"\nArtifacts saved under:\n  {output_dir}")
    return summary


def main() -> None:
    run_objectness_evaluation(EVAL_CONFIG)


if __name__ == "__main__":
    main()
