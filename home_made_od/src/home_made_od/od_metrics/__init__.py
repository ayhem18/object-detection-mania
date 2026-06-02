"""
Model-agnostic object-detection error-analysis metrics.
"""

from dl_lib.etalon_object_detection.modules.od_metrics.core_metrics import (
    PerGtPredStats,
    PerGtAnchorStats,
    compute_pred_gt_iou_matrix,
    per_gt_best_pred_stats,
    assign_positive_anchors_per_gt,
)
from dl_lib.etalon_object_detection.modules.od_metrics.objectness_ea import (
    recall_vs_confidence,
    best_score_distribution,
    objectness_failure_breakdown,
    per_gt_anchor_objectness_records,
    anchor_score_distribution,
    build_objectness_report,
)
from dl_lib.etalon_object_detection.modules.od_metrics.grid_metrics import (
    BBoxGridMetrics,
    BBoxSetsMetric,
    BboxGridInformation,
    Cell,
    Grid,
    PriorCellAssignment,
    ReferenceCellAssignment,
    bbox_central_cells,
    compute_bbox_grid_information,
    compute_bbox_grid_metrics,
    compute_bbox_sets_metric,
    covered_cell_counts,
    level_stats_dict,
    max_non_background_ratio_over_levels,
    snap_bbox_extent_to_grid,
)

__all__ = [
    "PerGtPredStats",
    "PerGtAnchorStats",
    "compute_pred_gt_iou_matrix",
    "per_gt_best_pred_stats",
    "assign_positive_anchors_per_gt",
    "recall_vs_confidence",
    "best_score_distribution",
    "objectness_failure_breakdown",
    "per_gt_anchor_objectness_records",
    "anchor_score_distribution",
    "build_objectness_report",
    "Grid",
    "Cell",
    "BboxGridInformation",
    "BBoxSetsMetric",
    "BBoxGridMetrics",
    "compute_bbox_grid_information",
    "compute_bbox_sets_metric",
    "compute_bbox_grid_metrics",
    "snap_bbox_extent_to_grid",
    "covered_cell_counts",
    "bbox_central_cells",
    "ReferenceCellAssignment",
    "PriorCellAssignment",
    "level_stats_dict",
    "max_non_background_ratio_over_levels",
]
