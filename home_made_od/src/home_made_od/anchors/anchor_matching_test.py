"""
Test 1 — detector anchor matching (RetinaNet ``proposal_matcher`` fidelity).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
from torchvision.ops.boxes import box_iou
from tqdm import tqdm

from home_made_od.anchors.anchor_eval_utils import (
    DEFAULT_FPN_LEVEL_IDS,
    FlatCandidateBatch,
)

logger = logging.getLogger(__name__)


@dataclass
class LevelMatchingStats:
    """Per manifest FPN level (Test 1)."""

    level_id: str
    total: int = 0
    high_quality: int = 0
    sum_assigned_iou: float = 0.0
    sum_absolute_max_iou: float = 0.0

    @property
    def high_quality_fraction(self) -> float:
        return self.high_quality / self.total if self.total > 0 else 0.0

    @property
    def mean_assigned_iou(self) -> float:
        return self.sum_assigned_iou / self.total if self.total > 0 else 0.0

    @property
    def mean_absolute_max_iou(self) -> float:
        return self.sum_absolute_max_iou / self.total if self.total > 0 else 0.0


@dataclass
class DetectorMatchingReport:
    """Dataset-level result of Test 1."""

    total_boxes: int
    high_quality_covered: int
    low_quality_covered: int
    unassigned: int
    sum_best_assigned_iou: float
    sum_absolute_max_iou: float
    fg_iou_threshold: float
    num_anchors: int
    level_stats: Dict[str, LevelMatchingStats] = field(default_factory=dict)

    @property
    def high_quality_fraction(self) -> float:
        return self.high_quality_covered / self.total_boxes if self.total_boxes > 0 else 0.0

    @property
    def low_quality_fraction(self) -> float:
        return self.low_quality_covered / self.total_boxes if self.total_boxes > 0 else 0.0

    @property
    def unassigned_fraction(self) -> float:
        return self.unassigned / self.total_boxes if self.total_boxes > 0 else 0.0

    @property
    def mean_best_assigned_iou(self) -> float:
        return self.sum_best_assigned_iou / self.total_boxes if self.total_boxes > 0 else 0.0

    @property
    def mean_absolute_max_iou(self) -> float:
        return self.sum_absolute_max_iou / self.total_boxes if self.total_boxes > 0 else 0.0

    def meets_high_quality_threshold(self, min_fraction: float = 0.95) -> bool:
        return self.high_quality_fraction >= min_fraction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test": "detector_anchor_matching",
            "fg_iou_threshold": self.fg_iou_threshold,
            "num_anchors": self.num_anchors,
            "total_boxes": self.total_boxes,
            "high_quality_covered": self.high_quality_covered,
            "low_quality_covered": self.low_quality_covered,
            "unassigned": self.unassigned,
            "high_quality_fraction": self.high_quality_fraction,
            "low_quality_fraction": self.low_quality_fraction,
            "unassigned_fraction": self.unassigned_fraction,
            "mean_best_assigned_iou": self.mean_best_assigned_iou,
            "mean_absolute_max_iou": self.mean_absolute_max_iou,
            "per_level": {
                lid: {
                    "total": s.total,
                    "high_quality": s.high_quality,
                    "high_quality_fraction": s.high_quality_fraction,
                    "mean_assigned_iou": s.mean_assigned_iou,
                    "mean_absolute_max_iou": s.mean_absolute_max_iou,
                }
                for lid, s in self.level_stats.items()
                if s.total > 0
            },
        }


def evaluate_detector_anchor_matching(
    model: nn.Module,
    reference_bboxes: torch.Tensor,
    candidates: FlatCandidateBatch,
    assigned_levels: Sequence[str],
    fg_iou_threshold: float = 0.5,
    level_ids: Sequence[str] = DEFAULT_FPN_LEVEL_IDS,
) -> DetectorMatchingReport:
    """
    Match each GT to training-style positive anchors via ``proposal_matcher``.

    ``assigned_levels[i]`` aligns with ``candidates.boxes[i]`` (flat batch).
    """
    if len(assigned_levels) != candidates.num_boxes:
        raise ValueError(
            f"assigned_levels length ({len(assigned_levels)}) != "
            f"candidate boxes ({candidates.num_boxes})"
        )

    n_gt = candidates.num_boxes
    logger.info(
        "Test 1 (detector matching): %d images, %d GT boxes, %d priors, fg_iou>=%.2f",
        candidates.num_images,
        n_gt,
        reference_bboxes.shape[0],
        fg_iou_threshold,
    )

    level_stats = {level: LevelMatchingStats(level_id=level) for level in level_ids}
    total_boxes = 0
    high_quality_covered = 0
    low_quality_covered = 0
    unassigned = 0
    sum_best_assigned_iou = 0.0
    sum_absolute_max_iou = 0.0

    if n_gt == 0:
        return DetectorMatchingReport(
            total_boxes=0,
            high_quality_covered=0,
            low_quality_covered=0,
            unassigned=0,
            sum_best_assigned_iou=0.0,
            sum_absolute_max_iou=0.0,
            fg_iou_threshold=fg_iou_threshold,
            num_anchors=int(reference_bboxes.shape[0]),
            level_stats=level_stats,
        )

    ious = box_iou(candidates.boxes, reference_bboxes)
    matched_idxs = model.proposal_matcher(ious)

    for gt_idx in tqdm(range(n_gt), desc="Test 1: detector matching", unit="gt"):
        total_boxes += 1
        level = assigned_levels[gt_idx]
        if level in level_stats:
            level_stats[level].total += 1

        absolute_max_iou = float(ious[gt_idx].max().item())
        sum_absolute_max_iou += absolute_max_iou
        if level in level_stats:
            level_stats[level].sum_absolute_max_iou += absolute_max_iou

        assigned_anchor_indices = torch.where(matched_idxs == gt_idx)[0]
        if assigned_anchor_indices.numel() == 0:
            unassigned += 1
            continue

        max_assigned_iou = float(ious[gt_idx, assigned_anchor_indices].max().item())
        sum_best_assigned_iou += max_assigned_iou
        if level in level_stats:
            level_stats[level].sum_assigned_iou += max_assigned_iou

        if max_assigned_iou >= fg_iou_threshold:
            high_quality_covered += 1
            if level in level_stats:
                level_stats[level].high_quality += 1
        else:
            low_quality_covered += 1

    report = DetectorMatchingReport(
        total_boxes=total_boxes,
        high_quality_covered=high_quality_covered,
        low_quality_covered=low_quality_covered,
        unassigned=unassigned,
        sum_best_assigned_iou=sum_best_assigned_iou,
        sum_absolute_max_iou=sum_absolute_max_iou,
        fg_iou_threshold=fg_iou_threshold,
        num_anchors=int(reference_bboxes.shape[0]),
        level_stats=level_stats,
    )
    logger.info(
        "Test 1 done: HQ %.1f%% (%d/%d), unassigned=%d, mean assigned IoU=%.4f",
        report.high_quality_fraction * 100,
        report.high_quality_covered,
        report.total_boxes,
        report.unassigned,
        report.mean_best_assigned_iou,
    )
    return report


def format_detector_matching_report(report: DetectorMatchingReport) -> str:
    if report.total_boxes == 0:
        return "No GT boxes to evaluate."

    lines = [
        "=======================================================",
        "           DETECTOR-MATCHING SANITY RESULTS            ",
        "=======================================================",
        f"Total Objects:             {report.total_boxes}",
        (
            f"High-Quality Coverage:     {report.high_quality_fraction * 100:6.2f}% "
            f"(IoU >= {report.fg_iou_threshold})"
        ),
        (
            f"Low-Quality (Fallback):    {report.low_quality_fraction * 100:6.2f}% "
            f"(IoU < {report.fg_iou_threshold})"
        ),
    ]
    if report.unassigned > 0:
        lines.append(
            f"Unassigned:                {report.unassigned_fraction * 100:6.2f}% "
            f"(no positive anchor)"
        )
    lines.extend(
        [
            (
                f"Average Max Match IoU:     {report.mean_best_assigned_iou:.4f} "
                f"(from assigned anchors)"
            ),
            (
                f"Average Max Overall IoU:   {report.mean_absolute_max_iou:.4f} "
                f"(from all anchors)"
            ),
            "-------------------------------------------------------",
            "FPN Level Coverage (High Quality Only):",
        ]
    )
    for level, stats in report.level_stats.items():
        if stats.total > 0:
            lines.append(
                f"  {level:11s} | HQ Coverage: {stats.high_quality_fraction * 100:6.2f}% | "
                f"Avg Assigned IoU: {stats.mean_assigned_iou:.4f} | "
                f"Avg Overall Max IoU: {stats.mean_absolute_max_iou:.4f}"
            )
    return "\n".join(lines)
