"""
Test 2 — per-GT cell recall (batched candidates, references = anchors).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from home_made_od.anchors.anchor_eval_utils import (
    FlatCandidateBatch,
    reference_tensors_for_grid,
)
from home_made_od.od_metrics.grid_metrics import (
    BBoxGridMetrics,
    Grid,
    ReferenceCellAssignment,
    compute_bbox_grid_metrics,
    level_stats_dict,
)

logger = logging.getLogger(__name__)


@dataclass
class CellRecallReport:
    """
    Test 2 results — one row per candidate (GT) box.

    Tensor fields are the canonical output; use ``image_index`` / ``box_index`` /
    ``image_stems`` to map back to images. Optional ``per_gt`` rows are built only
    for JSON export via ``to_dict(include_per_gt=True)``.
    """

    metric_threshold: float
    min_gt_threshold_ratio: float
    bbox_level_iou_threshold: float
    background_iou_threshold: float
    image_index: torch.Tensor
    box_index: torch.Tensor
    cell_recall: torch.Tensor
    passes: torch.Tensor
    best_level_id: List[Optional[str]]
    cells_non_background: torch.Tensor
    cells_in_footprint: torch.Tensor
    max_reference_iou: torch.Tensor
    image_stems: Optional[Sequence[str]] = None
    per_level_by_candidate: Optional[List[Dict[str, Dict[str, float]]]] = None
    _per_gt_cache: Optional[List[Dict[str, Any]]] = field(default=None, repr=False)

    @property
    def total_boxes(self) -> int:
        return int(self.cell_recall.shape[0])

    @property
    def passed(self) -> int:
        return int(self.passes.sum().item())

    @property
    def failed(self) -> int:
        return self.total_boxes - self.passed

    @property
    def pass_fraction(self) -> float:
        return self.passed / self.total_boxes if self.total_boxes > 0 else 0.0

    @property
    def mean_cell_recall(self) -> float:
        if self.total_boxes == 0:
            return 0.0
        return float(self.cell_recall.mean().item())

    def meets_gt_fraction_threshold(self) -> bool:
        return (
            self.total_boxes > 0
            and self.pass_fraction >= self.min_gt_threshold_ratio
        )

    def per_gt_rows(self) -> List[Dict[str, Any]]:
        """Materialize per-GT dict rows (for JSON or viz)."""
        if self._per_gt_cache is not None:
            return self._per_gt_cache

        rows: List[Dict[str, Any]] = []
        stems = self.image_stems
        for i in range(self.total_boxes):
            img_i = int(self.image_index[i].item())
            stem = stems[img_i] if stems and img_i < len(stems) else str(img_i)
            rows.append(
                {
                    "img_stem": stem,
                    "gt_idx": int(self.box_index[i].item()),
                    "cell_recall_in_gt": float(self.cell_recall[i].item()),
                    "cell_recall_best_level": self.best_level_id[i],
                    "cells_non_background": int(self.cells_non_background[i].item()),
                    "cells_in_gt_bbox": int(self.cells_in_footprint[i].item()),
                    "max_reference_iou": float(self.max_reference_iou[i].item()),
                    "passes": bool(self.passes[i].item()),
                    "per_level": (
                        self.per_level_by_candidate[i]
                        if self.per_level_by_candidate
                        else {}
                    ),
                }
            )
        self._per_gt_cache = rows
        return rows

    def to_dict(self, *, include_per_gt: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "test": "cell_recall",
            "metric_threshold": self.metric_threshold,
            "min_gt_threshold_ratio": self.min_gt_threshold_ratio,
            "bbox_level_iou_threshold": self.bbox_level_iou_threshold,
            "background_iou_threshold": self.background_iou_threshold,
            "total_boxes": self.total_boxes,
            "passed": self.passed,
            "failed": self.failed,
            "pass_fraction": self.pass_fraction,
            "meets_gt_fraction_threshold": self.meets_gt_fraction_threshold(),
            "mean_cell_recall": self.mean_cell_recall,
        }
        if include_per_gt:
            out["per_gt"] = self.per_gt_rows()
        return out


def _metrics_per_level_batched(
    reference_bboxes: torch.Tensor,
    reference_assignments: Sequence[ReferenceCellAssignment],
    candidates: FlatCandidateBatch,
    grids: Sequence[Grid],
    bbox_level_iou_threshold: float,
    background_iou_threshold: float,
) -> List[BBoxGridMetrics]:
    """One ``compute_bbox_grid_metrics`` per FPN level over **all** candidates."""
    metrics_per_level: List[BBoxGridMetrics] = []
    for grid in grids:
        ref_boxes, ref_cells = reference_tensors_for_grid(
            reference_bboxes, reference_assignments, grid
        )
        metrics_per_level.append(
            compute_bbox_grid_metrics(
                reference_bboxes=ref_boxes,
                reference_cells=ref_cells,
                candidate_bboxes=candidates.boxes,
                grid=grid,
                bbox_level_iou_threshold=bbox_level_iou_threshold,
                background_iou_threshold=background_iou_threshold,
            )
        )
    return metrics_per_level


def _aggregate_over_levels(
    metrics_per_level: Sequence[BBoxGridMetrics],
    n_candidates: int,
    *,
    store_per_level: bool,
) -> tuple[
    torch.Tensor,
    List[Optional[str]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[List[Dict[str, Dict[str, float]]]],
]:
    """
    Max ``non_background_ratio`` over FPN levels for each candidate row.

    Returns
    -------
    cell_recall, best_level_ids, cells_non_background, cells_in_footprint,
    max_reference_iou, optional per_level_by_candidate
    """
    device = metrics_per_level[0].non_background_ratio.device
    if n_candidates == 0:
        zf = torch.zeros(0, device=device)
        zi = torch.zeros(0, dtype=torch.long, device=device)
        return zf, [], zi, zi, zf, [] if store_per_level else None

    level_ids = [m.grid.grid_id for m in metrics_per_level]
    ratios = torch.stack(
        [m.non_background_ratio for m in metrics_per_level], dim=0
    )
    max_ref_iou = torch.stack(
        [m.max_reference_iou_per_candidate for m in metrics_per_level], dim=0
    ).max(dim=0).values

    best_vals, best_lv = ratios.max(dim=0)
    best_level_ids: List[Optional[str]] = [
        level_ids[int(i)] for i in best_lv.tolist()
    ]

    non_bg_stack = torch.stack(
        [m.cells_non_background for m in metrics_per_level], dim=0
    )
    cells_non_bg = non_bg_stack.gather(0, best_lv.unsqueeze(0)).squeeze(0)

    footprint_stack = torch.stack(
        [
            torch.tensor(
                [info.covered_cell_count for info in m.candidate_grid_info],
                dtype=torch.long,
                device=device,
            )
            for m in metrics_per_level
        ],
        dim=0,
    )
    cells_in = footprint_stack.gather(0, best_lv.unsqueeze(0)).squeeze(0)

    per_level_list: Optional[List[Dict[str, Dict[str, float]]]] = None
    if store_per_level:
        per_level_list = [{} for _ in range(n_candidates)]
        for lv, metrics in enumerate(metrics_per_level):
            for cand_idx in range(n_candidates):
                per_level_list[cand_idx][level_ids[lv]] = level_stats_dict(
                    metrics, cand_idx
                )

    return (
        best_vals,
        best_level_ids,
        cells_non_bg,
        cells_in,
        max_ref_iou,
        per_level_list,
    )


def evaluate_cell_recall(
    reference_bboxes: torch.Tensor,
    reference_assignments: Sequence[ReferenceCellAssignment],
    grids: Sequence[Grid],
    candidates: FlatCandidateBatch,
    bbox_level_iou_threshold: float = 0.5,
    background_iou_threshold: float = 0.4,
    metric_threshold: float = 0.5,
    min_gt_threshold_ratio: float = 0.95,
    *,
    store_per_level: bool = False,
) -> CellRecallReport:
    """
    Batched cell recall for all candidate (GT) boxes.

    Parameters
    ----------
    candidates :
        Flat ``[N, 4]`` boxes plus ``image_index`` / ``box_index`` for mapping results.
    store_per_level :
        If True, attach per-FPN breakdown per candidate (heavier; for diagnostics).
    """
    n_gt = candidates.num_boxes
    n_images = candidates.num_images
    logger.info(
        "Test 2 (cell recall): %d images, %d GT boxes (batched), recall>=%.2f, "
        "pass_fraction>=%.2f, fg_cell>=%.2f, bg_cell<%.2f",
        n_images,
        n_gt,
        metric_threshold,
        min_gt_threshold_ratio,
        bbox_level_iou_threshold,
        background_iou_threshold,
    )

    if n_gt == 0:
        device = reference_bboxes.device
        empty_f = torch.zeros(0, device=device)
        empty_l = torch.zeros(0, dtype=torch.long, device=device)
        return CellRecallReport(
            metric_threshold=metric_threshold,
            min_gt_threshold_ratio=min_gt_threshold_ratio,
            bbox_level_iou_threshold=bbox_level_iou_threshold,
            background_iou_threshold=background_iou_threshold,
            image_index=empty_l,
            box_index=empty_l,
            cell_recall=empty_f,
            passes=torch.zeros(0, dtype=torch.bool, device=device),
            best_level_id=[],
            cells_non_background=empty_l,
            cells_in_footprint=empty_l,
            max_reference_iou=empty_f,
            image_stems=candidates.image_stems,
        )

    metrics_per_level = _metrics_per_level_batched(
        reference_bboxes,
        reference_assignments,
        candidates,
        grids,
        bbox_level_iou_threshold,
        background_iou_threshold,
    )

    (
        cell_recall,
        best_level_ids,
        cells_non_bg,
        cells_in,
        max_ref_iou,
        per_level_list,
    ) = _aggregate_over_levels(
        metrics_per_level,
        n_gt,
        store_per_level=store_per_level,
    )

    passes = cell_recall >= metric_threshold

    report = CellRecallReport(
        metric_threshold=metric_threshold,
        min_gt_threshold_ratio=min_gt_threshold_ratio,
        bbox_level_iou_threshold=bbox_level_iou_threshold,
        background_iou_threshold=background_iou_threshold,
        image_index=candidates.image_index,
        box_index=candidates.box_index,
        cell_recall=cell_recall,
        passes=passes,
        best_level_id=best_level_ids,
        cells_non_background=cells_non_bg,
        cells_in_footprint=cells_in,
        max_reference_iou=max_ref_iou,
        image_stems=candidates.image_stems,
        per_level_by_candidate=per_level_list,
    )
    logger.info(
        "Test 2 done: %d/%d GT passed (%.1f%%), mean recall=%.4f, dataset pass=%s",
        report.passed,
        report.total_boxes,
        report.pass_fraction * 100,
        report.mean_cell_recall,
        "YES" if report.meets_gt_fraction_threshold() else "NO",
    )
    return report


def save_cell_recall_histogram(report: CellRecallReport, output_dir: Path) -> None:
    """Per-GT cell recall histogram."""
    import matplotlib.pyplot as plt

    if report.total_boxes == 0:
        logger.warning("No per-GT recalls to plot.")
        return

    output_dir = Path(output_dir)
    cell_dir = output_dir / "cell_recall"
    cell_dir.mkdir(parents=True, exist_ok=True)

    recalls = report.cell_recall.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(recalls, bins=30, color="olive", edgecolor="white")
    ax.axvline(
        report.metric_threshold,
        color="red",
        linestyle="--",
        label=f"metric_threshold={report.metric_threshold}",
    )
    ax.set_xlabel("cell_recall (max non_background_ratio over FPN)")
    ax.set_ylabel("Count")
    ax.set_title("Test 2 — cell recall per GT")
    ax.legend()
    fig.tight_layout()
    hist_path = cell_dir / "cell_recall_histogram.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote cell recall histogram: %s", hist_path)


def format_cell_recall_report(report: CellRecallReport) -> str:
    if report.total_boxes == 0:
        return "No GT boxes to evaluate (Test 2)."

    return "\n".join(
        [
            "=======================================================",
            "              CELL RECALL ANCHOR TEST (2)              ",
            "=======================================================",
            f"Total GT boxes:            {report.total_boxes}",
            f"Passed:                    {report.passed} ({report.pass_fraction * 100:.1f}%)",
            f"Failed:                    {report.failed}",
            f"Mean cell recall:          {report.mean_cell_recall:.4f}",
            (
                f"Thresholds:                per-GT recall >= {report.metric_threshold}, "
                f"GT pass fraction >= {report.min_gt_threshold_ratio}, "
                f"fg cell IoU >= {report.bbox_level_iou_threshold}, "
                f"bg cell IoU < {report.background_iou_threshold}"
            ),
            (
                f"Dataset pass (Test 2):     "
                f"{'YES' if report.meets_gt_fraction_threshold() else 'NO'}"
            ),
        ]
    )
