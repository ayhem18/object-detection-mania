"""
Pre-training anchor evaluation — thin entry point.

- ``anchor_matching_test`` — Test 1
- ``cell_recall_test`` — Test 2 (batched flat candidates)
- ``anchor_eval_utils`` — ``FlatCandidateBatch``, grid slicing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from home_made_od.anchors.anchor_eval_utils import (
    AnchorEvaluationReport,
    FlatCandidateBatch,
)
from home_made_od.anchors.anchor_matching_test import (
    DetectorMatchingReport,
    evaluate_detector_anchor_matching,
    format_detector_matching_report,
)
from home_made_od.anchors.cell_recall_test import (
    CellRecallReport,
    evaluate_cell_recall,
    format_cell_recall_report,
    save_cell_recall_histogram,
)
from home_made_od.od_metrics.grid_metrics import (
    Grid,
    ReferenceCellAssignment,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FlatCandidateBatch",
    "AnchorEvaluationReport",
    "DetectorMatchingReport",
    "CellRecallReport",
    "evaluate_anchors",
    "evaluate_detector_anchor_matching",
    "evaluate_cell_recall",
    "format_anchor_evaluation_report",
    "format_detector_matching_report",
    "format_cell_recall_report",
    "save_cell_recall_histogram",
]


def evaluate_anchors(
    *,
    reference_bboxes: torch.Tensor,
    reference_assignments: Sequence[ReferenceCellAssignment],
    grids: Sequence[Grid],
    candidates: FlatCandidateBatch,
    assigned_levels: Sequence[str],
    model: Optional[nn.Module] = None,
    fg_iou_threshold: float = 0.5,
    bbox_level_iou_threshold: float = 0.5,
    background_iou_threshold: float = 0.4,
    metric_threshold: float = 0.5,
    min_gt_threshold_ratio: float = 0.95,
    run_detector_matching: bool = False,
    run_cell_recall: bool = True,
    output_dir: Optional[Path] = None,
    store_per_level: bool = False,
) -> AnchorEvaluationReport:
    """
    Run anchor tests.

    ``candidates`` is a flat ``[N, 4]`` batch; ``image_index`` / ``box_index`` map results.
    ``assigned_levels`` is parallel to ``candidates.boxes`` (Test 1 only).
    """
    out_path = Path(output_dir) if output_dir is not None else None
    logger.info(
        "Anchor evaluation: %d images, %d GT boxes, test1=%s, test2=%s",
        candidates.num_images,
        candidates.num_boxes,
        run_detector_matching,
        run_cell_recall,
    )

    dm_report: Optional[DetectorMatchingReport] = None
    cr_report: Optional[CellRecallReport] = None

    if run_detector_matching:
        if model is None:
            raise ValueError("run_detector_matching=True requires ``model``.")
        dm_report = evaluate_detector_anchor_matching(
            model=model,
            reference_bboxes=reference_bboxes,
            candidates=candidates,
            assigned_levels=assigned_levels,
            fg_iou_threshold=fg_iou_threshold,
        )

    if run_cell_recall:
        cr_report = evaluate_cell_recall(
            reference_bboxes=reference_bboxes,
            reference_assignments=reference_assignments,
            grids=grids,
            candidates=candidates,
            bbox_level_iou_threshold=bbox_level_iou_threshold,
            background_iou_threshold=background_iou_threshold,
            metric_threshold=metric_threshold,
            min_gt_threshold_ratio=min_gt_threshold_ratio,
            store_per_level=store_per_level,
        )
        if out_path is not None:
            save_cell_recall_histogram(cr_report, out_path)

    return AnchorEvaluationReport(
        detector_matching=dm_report,
        cell_recall=cr_report,
        output_dir=out_path,
    )


def format_anchor_evaluation_report(eval_report: AnchorEvaluationReport) -> str:
    parts: list[str] = []
    if eval_report.detector_matching is not None:
        parts.append(format_detector_matching_report(eval_report.detector_matching))
    if eval_report.cell_recall is not None:
        if parts:
            parts.append("")
        parts.append(format_cell_recall_report(eval_report.cell_recall))
    if eval_report.output_dir is not None:
        parts.append(f"\nArtifacts: {eval_report.output_dir}")
    return "\n".join(parts) if parts else "No tests were run."
