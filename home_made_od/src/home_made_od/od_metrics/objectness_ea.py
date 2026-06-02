"""
Objectness error analysis — Layer A (post-processed preds) and Layer B (anchor grid).

This module is model-agnostic. It only imports from ``core_metrics`` (plus torch/numpy).

-------------------------------------------------------------------------------
QUICK ADAPTER GUIDE
-------------------------------------------------------------------------------

**Single image — Layer A only**

    from dl_lib.etalon_object_detection.modules.od_metrics.objectness_ea import (
        build_objectness_report,
    )

    report = build_objectness_report(
        gt_boxes=targets["boxes"],          # [N_gt, 4] xyxy
        pred_boxes=preds["boxes"],          # [N_pred, 4] xyxy
        pred_scores=preds["scores"],        # [N_pred]
        # Optional:
        pred_labels=preds.get("labels"),
        gt_labels=targets.get("labels"),
        foreground_label_ids={1, 2, 3},   # exclude background 0
    )

**Single image — Layer A + Layer B**

    report = build_objectness_report(
        gt_boxes=...,
        pred_boxes=...,
        pred_scores=...,
        anchor_boxes=anchors,               # [N_anchor, 4] xyxy
        anchor_scores=anchor_fg_scores,     # [N_anchor]
        # Optional: use training matcher output instead of default IoU rule
        positive_anchor_indices_per_gt=my_assigner_indices,
    )

**Dataset / validation loop**

    layer_a_records = []
    layer_b_records = []
    for preds, targets, anchors, anchor_scores in loader:
        report = build_objectness_report(...)
        layer_a_records.append(report["layer_a"]["per_gt_records"])
        if "layer_b" in report:
            layer_b_records.append(report["layer_b"]["per_gt_records"])

    agg = aggregate_objectness_reports(
        [build_objectness_report(...) for each image],
        conf_thresholds=[0.05, 0.1, 0.3, 0.5, 0.7],
    )

-------------------------------------------------------------------------------
INTERPRETATION CHEAT SHEET
-------------------------------------------------------------------------------

| Symptom | Layer A signal | Layer B signal | Likely cause |
|---------|----------------|----------------|--------------|
| Head never fires near GT | low ``best_score`` | low ``max_positive_score`` | objectness / cls loss, imbalance |
| Head fires, boxes wrong | low ``best_iou``, ok anchor IoU | high anchor IoU, low pred ``best_iou`` | bbox regression |
| Anchors cannot overlap GT | n/a | low ``max_anchor_iou`` | anchor optimization / scales |
| Only some sizes fail | stratified ``best_score`` bins | stratified anchor scores | FPN level / aspect ratio gap |
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch

from dl_lib.etalon_object_detection.modules.od_metrics.core_metrics import (
    PerGtAnchorStats,
    PerGtPredStats,
    flatten_per_image_records,
    per_gt_anchor_objectness_records,
    per_gt_best_pred_stats,
)

# from dl_lib.etalon_object_detection.modules.od_metrics.grid_metrics import (
#     MIN_CELL_AXIS_COVERAGE_FRAC,
#     PerGtGridStats,
#     aggregate_grid_stats,
#     per_gt_grid_stats,
# )


# ---------------------------------------------------------------------------
# Layer A — recall vs confidence (overview curve)
# ---------------------------------------------------------------------------


def recall_vs_confidence(
    per_gt_records: Sequence[PerGtPredStats],
    conf_thresholds: Sequence[float],
    iou_eval: float = 0.5,
    score_field: str = "best_score",
) -> Dict[float, Dict[str, float]]:
    """
    Overview metric: fraction of GT "detected" as confidence threshold varies.

    A GT is counted as **detected** at threshold ``tau`` when BOTH hold:

      1. ``best_iou >= iou_eval``   — localization good enough (oracle box quality)
      2. ``getattr(record, score_field) >= tau``   — objectness high enough

    Typical use
    -----------
    - Fix ``iou_eval=0.5`` (COCO-style) and sweep ``tau`` → objectness calibration curve.
    - If recall is low at ``tau=0.05`` but ``best_iou`` histogram is healthy, the head
      fires in the wrong place (localization), not objectness.
    - If recall is low even at ``tau=0.01`` AND ``best_iou`` is low, predictions are
      missing or misplaced before score even matters.

    Parameters
    ----------
    per_gt_records :
        Output of ``per_gt_best_pred_stats`` for one or many images (concatenated).
    conf_thresholds :
        Confidence values on the SAME scale as ``pred_scores`` you passed in.
    iou_eval :
        Strict IoU gate defining "localized well enough to count as found".
    score_field :
        Which record field to threshold. Default ``best_score``; rarely changed.

    Returns
    -------
    dict[tau, {"recall": float, "num_detected": int, "num_gt": int}]
    """
    num_gt = len(per_gt_records)
    if num_gt == 0:
        return {float(t): {"recall": 0.0, "num_detected": 0, "num_gt": 0} for t in conf_thresholds}

    curve: Dict[float, Dict[str, float]] = {}
    for tau in conf_thresholds:
        tau_f = float(tau)
        detected = 0
        for rec in per_gt_records:
            score = float(getattr(rec, score_field))
            if rec.best_iou >= iou_eval and score >= tau_f:
                detected += 1
        curve[tau_f] = {
            "recall": detected / num_gt,
            "num_detected": detected,
            "num_gt": num_gt,
        }
    return curve


# ---------------------------------------------------------------------------
# Layer A — score distribution & failure buckets
# ---------------------------------------------------------------------------


def best_score_distribution(
    per_gt_records: Sequence[PerGtPredStats],
    histogram_bins: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Distribution of ``best_score`` across GT (objectness strength near each object).

    Also summarizes ``best_iou`` because objectness scores are meaningless when no pred
    overlaps the GT (``best_score`` will be 0 by construction).

    Parameters
    ----------
    histogram_bins :
        Monotonic bin edges for ``best_score`` histogram, e.g.
        ``[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]``.
        Default uses sensible detection thresholds.

    Returns
    -------
    dict with keys:
        - ``num_gt``
        - ``best_score``: mean, median (p50), p10, p90, min, max, histogram
        - ``best_iou``: same summary (localization ceiling)
    """
    if histogram_bins is None:
        histogram_bins = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01]

    num_gt = len(per_gt_records)
    empty_score = {
        "mean": 0.0,
        "p10": 0.0,
        "p50": 0.0,
        "p90": 0.0,
        "min": 0.0,
        "max": 0.0,
        "histogram": {},
    }
    if num_gt == 0:
        return {"num_gt": 0, "best_score": empty_score, "best_iou": empty_score}

    scores = torch.tensor([r.best_score for r in per_gt_records], dtype=torch.float32)
    ious = torch.tensor([r.best_iou for r in per_gt_records], dtype=torch.float32)

    def _summarize(values: torch.Tensor) -> Dict[str, Any]:
        return {
            "mean": float(values.mean().item()),
            "p10": float(torch.quantile(values, 0.10).item()),
            "p50": float(torch.quantile(values, 0.50).item()),
            "p90": float(torch.quantile(values, 0.90).item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "histogram": _build_histogram(values, histogram_bins),
        }

    return {
        "num_gt": num_gt,
        "best_score": _summarize(scores),
        "best_iou": _summarize(ious),
    }


def _build_histogram(
    values: torch.Tensor,
    bin_edges: Sequence[float],
) -> Dict[str, int]:
    """Count values in half-open intervals [edge_i, edge_{i+1})."""
    counts: Dict[str, int] = {}
    edges = list(bin_edges)
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        mask = (values >= low) & (values < high)
        key = f"[{low:.3f}, {high:.3f})"
        counts[key] = int(mask.sum().item())
    return counts


def objectness_failure_breakdown(
    per_gt_records: Sequence[PerGtPredStats],
    conf_threshold: float = 0.3,
    iou_eval: float = 0.5,
    iou_overlap_for_score: float = 0.1,
) -> Dict[str, Any]:
    """
    Classify each GT into a single primary failure bucket for debugging.

    Buckets (mutually exclusive, evaluated in order):

    1. **detected** —
       ``best_iou >= iou_eval`` AND ``best_score >= conf_threshold``

    2. **localization_failure** —
       ``best_iou < iou_eval`` (no well-localized prediction; objectness score irrelevant)

    3. **objectness_failure** —
       ``best_iou >= iou_eval`` BUT ``best_score < conf_threshold``
       (a decent box exists nearby but is suppressed / low confidence)

    4. **no_overlap_failure** —
       ``best_iou < iou_overlap_for_score`` (no prediction even loosely covers GT;
       often complete miss or wrong region)

    Note: bucket 4 is a subset of localization issues but highlights "no fire at all"
    near the GT at score-evaluation overlap ``iou_overlap_for_score``.

    Parameters
    ----------
    conf_threshold :
        Deploy-style score cutoff you care about.
    iou_eval :
        Strict localization requirement for **detected**.
    iou_overlap_for_score :
        Documented for callers; ``best_score`` already used ``iou_overlap_min`` when
        records were built. This bucket uses ``best_iou`` vs a loose overlap constant
        to flag complete spatial misses.
    """
    counts = {
        "detected": 0,
        "objectness_failure": 0,
        "localization_failure": 0,
        "no_overlap_failure": 0,
    }
    indices: Dict[str, List[int]] = {k: [] for k in counts}

    for rec in per_gt_records:
        if rec.best_iou >= iou_eval and rec.best_score >= conf_threshold:
            key = "detected"
        elif rec.best_iou < iou_overlap_for_score:
            key = "no_overlap_failure"
        elif rec.best_iou < iou_eval:
            key = "localization_failure"
        else:
            key = "objectness_failure"

        counts[key] += 1
        indices[key].append(rec.gt_idx)

    num_gt = len(per_gt_records) or 1
    return {
        "num_gt": len(per_gt_records),
        "counts": counts,
        "fractions": {k: v / num_gt for k, v in counts.items()},
        "gt_indices_by_bucket": indices,
        "thresholds": {
            "conf_threshold": conf_threshold,
            "iou_eval": iou_eval,
            "iou_overlap_for_score": iou_overlap_for_score,
        },
    }


# ---------------------------------------------------------------------------
# Layer B — anchor score distributions & stratification
# ---------------------------------------------------------------------------


def anchor_score_distribution(
    per_gt_anchor_records: Sequence[PerGtAnchorStats],
    histogram_bins: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Dataset-level distribution of anchor-level objectness on positive assignments.

    Key fields per GT (from ``PerGtAnchorStats``):
      - ``max_positive_score`` : peak head response on assigned anchors
      - ``max_anchor_iou``     : structural anchor–GT overlap (coverage)
      - ``num_positive_anchors``: how many anchors matched (assignment density)

    Low ``max_anchor_iou`` with low scores → anchor design problem.
    High ``max_anchor_iou`` with low ``max_positive_score`` → training / loss problem.
    """
    if histogram_bins is None:
        histogram_bins = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01]

    num_gt = len(per_gt_anchor_records)
    if num_gt == 0:
        return {"num_gt": 0}

    max_pos = torch.tensor(
        [r.max_positive_score for r in per_gt_anchor_records], dtype=torch.float32
    )
    max_iou = torch.tensor(
        [r.max_anchor_iou for r in per_gt_anchor_records], dtype=torch.float32
    )
    num_pos = torch.tensor(
        [r.num_positive_anchors for r in per_gt_anchor_records], dtype=torch.float32
    )

    zero_positive_fraction = float((num_pos == 0).float().mean().item())

    return {
        "num_gt": num_gt,
        "fraction_gt_with_zero_positive_anchors": zero_positive_fraction,
        "max_positive_score": {
            "mean": float(max_pos.mean().item()),
            "p50": float(torch.quantile(max_pos, 0.5).item()),
            "histogram": _build_histogram(max_pos, histogram_bins),
        },
        "max_anchor_iou": {
            "mean": float(max_iou.mean().item()),
            "p50": float(torch.quantile(max_iou, 0.5).item()),
            "histogram": _build_histogram(max_iou, [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]),
        },
        "num_positive_anchors": {
            "mean": float(num_pos.mean().item()),
            "p50": float(torch.quantile(num_pos, 0.5).item()),
        },
    }


def stratify_per_gt_records(
    per_gt_records: Sequence[Union[PerGtPredStats, PerGtAnchorStats]],
    metadata_key: str,
    default_value: str = "unknown",
) -> Dict[str, List[Union[PerGtPredStats, PerGtAnchorStats]]]:
    """
    Group records by a metadata field for per-stratum dashboards.

    Example
    -------
        by_level = stratify_per_gt_records(records, "assigned_fpn_level")
        for level, group in by_level.items():
            print(level, best_score_distribution(group))
    """
    groups: Dict[str, List] = {}
    for rec in per_gt_records:
        if rec.metadata and metadata_key in rec.metadata:
            key = str(rec.metadata[metadata_key])
        else:
            key = default_value
        groups.setdefault(key, []).append(rec)
    return groups


def stratified_objectness_report(
    per_gt_pred_records: Sequence[PerGtPredStats],
    metadata_key: str,
    conf_thresholds: Sequence[float],
    iou_eval: float = 0.5,
) -> Dict[str, Any]:
    """
    Layer A metrics recomputed per metadata stratum (FPN level, size bin, class, ...).
    """
    groups = stratify_per_gt_records(per_gt_pred_records, metadata_key)
    out: Dict[str, Any] = {}
    for stratum, group in groups.items():
        out[stratum] = {
            "num_gt": len(group),
            "recall_curve": recall_vs_confidence(group, conf_thresholds, iou_eval=iou_eval),
            "score_distribution": best_score_distribution(group),
            "failure_breakdown": objectness_failure_breakdown(group, iou_eval=iou_eval),
        }
    return out


def stratified_anchor_report(
    per_gt_anchor_records: Sequence[PerGtAnchorStats],
    metadata_key: str,
) -> Dict[str, Any]:
    """Layer B metrics per metadata stratum."""
    groups = stratify_per_gt_records(per_gt_anchor_records, metadata_key)
    return {
        stratum: anchor_score_distribution(group)
        for stratum, group in groups.items()
    }


# ---------------------------------------------------------------------------
# Unified per-image report + dataset aggregation
# ---------------------------------------------------------------------------


def build_objectness_report(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: Optional[torch.Tensor] = None,
    gt_labels: Optional[torch.Tensor] = None,
    foreground_label_ids: Optional[Set[int]] = None,
    iou_overlap_min: float = 0.1,
    conf_thresholds: Optional[Sequence[float]] = None,
    iou_eval: float = 0.5,
    conf_threshold_for_breakdown: float = 0.3,
    gt_metadata: Optional[Sequence[Optional[Dict]]] = None,
    stratify_key: Optional[str] = None,
    # --- Layer B (all optional; omit for pred-only analysis) ---
    anchor_boxes: Optional[torch.Tensor] = None,
    anchor_scores: Optional[torch.Tensor] = None,
    positive_anchor_indices_per_gt: Optional[List[torch.Tensor]] = None,
    fg_iou_threshold: float = 0.5,
    # --- Layer B grid (optional; requires prior_to_cell + level_grids from adapter) ---
    prior_to_cell: Optional[Sequence] = None,
    level_grids: Optional[Sequence] = None,
    grid_iou_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Single-image objectness report (Layer A mandatory, Layer B if anchors provided).

    This is the main entry point to call from training loops or evaluation scripts.
    Returns a JSON-serializable-friendly dict (tensors converted to counts/lists).

    Parameters
    ----------
    iou_overlap_min :
        Passed to ``per_gt_best_pred_stats`` — how close a pred must be to count its
        score toward ``best_score``.
    conf_thresholds :
        Confidence sweep for ``recall_vs_confidence``. Defaults to common cutoffs.
    stratify_key :
        If set AND ``gt_metadata`` contains this key, adds ``stratified`` sub-report.

    Layer B activation
    ------------------
    Supply BOTH ``anchor_boxes`` and ``anchor_scores`` to enable Layer B.
    """
    if conf_thresholds is None:
        conf_thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    # ----- Layer A -----
    layer_a_records = per_gt_best_pred_stats(
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_labels=gt_labels,
        foreground_label_ids=foreground_label_ids,
        iou_overlap_min=iou_overlap_min,
        gt_metadata=gt_metadata,
    )

    layer_a: Dict[str, Any] = {
        "num_gt": len(layer_a_records),
        "per_gt_records": layer_a_records,
        "recall_curve": recall_vs_confidence(
            layer_a_records, conf_thresholds, iou_eval=iou_eval
        ),
        "score_distribution": best_score_distribution(layer_a_records),
        "failure_breakdown": objectness_failure_breakdown(
            layer_a_records,
            conf_threshold=conf_threshold_for_breakdown,
            iou_eval=iou_eval,
        ),
        "params": {
            "iou_overlap_min": iou_overlap_min,
            "iou_eval": iou_eval,
            "conf_threshold_for_breakdown": conf_threshold_for_breakdown,
        },
    }

    if stratify_key is not None and gt_metadata is not None:
        layer_a["stratified"] = stratified_objectness_report(
            layer_a_records,
            metadata_key=stratify_key,
            conf_thresholds=conf_thresholds,
            iou_eval=iou_eval,
        )

    report: Dict[str, Any] = {"layer_a": layer_a}

    # ----- Layer B (optional) -----
    if anchor_boxes is not None and anchor_scores is not None:
        layer_b_records = per_gt_anchor_objectness_records(
            gt_boxes=gt_boxes,
            anchor_boxes=anchor_boxes,
            anchor_scores=anchor_scores,
            gt_labels=gt_labels,
            positive_anchor_indices_per_gt=positive_anchor_indices_per_gt,
            fg_iou_threshold=fg_iou_threshold,
            gt_metadata=gt_metadata,
        )

        layer_b: Dict[str, Any] = {
            "num_gt": len(layer_b_records),
            "per_gt_records": layer_b_records,
            "anchor_score_distribution": anchor_score_distribution(layer_b_records),
            "params": {"fg_iou_threshold": fg_iou_threshold},
        }

        if stratify_key is not None and gt_metadata is not None:
            layer_b["stratified"] = stratified_anchor_report(
                layer_b_records, metadata_key=stratify_key
            )

        report["layer_b"] = layer_b

    # ----- Layer B grid -----
    if (
        anchor_boxes is not None
        and prior_to_cell is not None
        and level_grids is not None
    ):
        grid_t = grid_iou_threshold if grid_iou_threshold is not None else fg_iou_threshold
        grid_records = per_gt_grid_stats(
            gt_boxes=gt_boxes,
            prior_boxes=anchor_boxes,
            prior_to_cell=prior_to_cell,
            level_grids=level_grids,
            iou_threshold=grid_t,
            gt_labels=gt_labels,
            gt_metadata=gt_metadata,
        )
        report["layer_b_grid"] = {
            "num_gt": len(grid_records),
            "per_gt_records": grid_records,
            "aggregate": aggregate_grid_stats(grid_records),
            "params": {
                "grid_iou_threshold": grid_t,
                "min_cell_axis_coverage_frac": MIN_CELL_AXIS_COVERAGE_FRAC,
            },
        }

    return report


def aggregate_objectness_reports(
    per_image_reports: Sequence[Dict[str, Any]],
    conf_thresholds: Optional[Sequence[float]] = None,
    iou_eval: float = 0.5,
    conf_threshold_for_breakdown: float = 0.3,
) -> Dict[str, Any]:
    """
    Merge per-image ``build_objectness_report`` outputs into dataset-level metrics.

    Use this at the end of a validation epoch:

        reports = []
        for batch in val_loader:
            for preds, targets in zip(batch_preds, batch_targets):
                reports.append(build_objectness_report(...))
        summary = aggregate_objectness_reports(reports)

    Note: ``per_gt_records`` in the returned aggregate are flattened lists suitable
    for TensorBoard histograms or saving to parquet.
    """
    if conf_thresholds is None:
        conf_thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    layer_a_lists: List[List[PerGtPredStats]] = []
    layer_b_lists: List[List[PerGtAnchorStats]] = []
    grid_lists: List[List[PerGtGridStats]] = []

    for rep in per_image_reports:
        if rep.get("layer_a") and rep["layer_a"].get("per_gt_records") is not None:
            layer_a_lists.append(rep["layer_a"]["per_gt_records"])
        if rep.get("layer_b") and rep["layer_b"].get("per_gt_records") is not None:
            layer_b_lists.append(rep["layer_b"]["per_gt_records"])
        if rep.get("layer_b_grid") and rep["layer_b_grid"].get("per_gt_records") is not None:
            grid_lists.append(rep["layer_b_grid"]["per_gt_records"])

    flat_a = flatten_per_image_records(layer_a_lists)  # type: ignore[arg-type]
    flat_b = flatten_per_image_records(layer_b_lists)  # type: ignore[arg-type]
    flat_grid = flatten_per_image_records(grid_lists)  # type: ignore[arg-type]

    aggregated: Dict[str, Any] = {
        "num_images": len(per_image_reports),
        "layer_a": {
            "num_gt": len(flat_a),
            "recall_curve": recall_vs_confidence(flat_a, conf_thresholds, iou_eval=iou_eval),
            "score_distribution": best_score_distribution(flat_a),
            "failure_breakdown": objectness_failure_breakdown(
                flat_a,
                conf_threshold=conf_threshold_for_breakdown,
                iou_eval=iou_eval,
            ),
        },
    }

    if flat_b:
        aggregated["layer_b"] = {
            "num_gt": len(flat_b),
            "anchor_score_distribution": anchor_score_distribution(flat_b),
        }

    if flat_grid:
        aggregated["layer_b_grid"] = aggregate_grid_stats(flat_grid)

    return aggregated


def records_to_serializable(
    records: Sequence[Union[PerGtPredStats, PerGtAnchorStats]],
) -> List[Dict[str, Any]]:
    """
    Convert dataclass records to plain dicts for JSON logging.

    Tensor fields (anchor indices) are converted to Python lists.
    """
    out: List[Dict[str, Any]] = []
    for rec in records:
        d = {
            "gt_idx": rec.gt_idx,
            "gt_label": rec.gt_label,
            "metadata": rec.metadata,
        }
        if isinstance(rec, PerGtPredStats):
            d.update(
                {
                    "best_iou": rec.best_iou,
                    "best_score": rec.best_score,
                    "best_score_pred_idx": rec.best_score_pred_idx,
                    "best_iou_pred_idx": rec.best_iou_pred_idx,
                }
            )
        elif isinstance(rec, PerGtAnchorStats):
            d.update(
                {
                    "num_positive_anchors": rec.num_positive_anchors,
                    "max_positive_score": rec.max_positive_score,
                    "mean_positive_score": rec.mean_positive_score,
                    "max_anchor_iou": rec.max_anchor_iou,
                    "positive_anchor_indices": rec.positive_anchor_indices.cpu().tolist(),
                }
            )
        out.append(d)
    return out
