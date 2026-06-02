"""Anchor computation and pre-training evaluation."""

from home_made_od.anchors.anchor_computation_strategies import (
    ASSIGNMENT_METHODS,
    AnchorOptimizationResult,
    BoxDimensions,
    FPN_LEVEL_ORDER,
    LevelBoxMap,
    METHOD_REQUIRED_PARAMETERS,
    assign_to_level_by_area,
    assign_to_level_by_dimension,
    calculate_fpn_specs,
    cluster_1d_values,
    compute_anchor_config_hash,
    compute_optimized_anchors,
    generate_anchors,
    group_box_dimensions_by_fpn_level,
    validate_method_parameters,
)
from home_made_od.anchors.anchor_eval_utils import (
    DEFAULT_FPN_LEVEL_IDS,
    AnchorEvaluationReport,
    FlatCandidateBatch,
    flatten_candidate_boxes,
    reference_tensors_for_grid,
)
from home_made_od.anchors.anchor_evaluation import (
    evaluate_anchors,
    format_anchor_evaluation_report,
)
from home_made_od.anchors.anchor_matching_test import (
    DetectorMatchingReport,
    LevelMatchingStats,
    evaluate_detector_anchor_matching,
    format_detector_matching_report,
)
from home_made_od.anchors.cell_recall_test import (
    CellRecallReport,
    evaluate_cell_recall,
    format_cell_recall_report,
    save_cell_recall_histogram,
)

__all__ = [
    "ASSIGNMENT_METHODS",
    "AnchorOptimizationResult",
    "BoxDimensions",
    "FPN_LEVEL_ORDER",
    "LevelBoxMap",
    "METHOD_REQUIRED_PARAMETERS",
    "validate_method_parameters",
    "compute_anchor_config_hash",
    "compute_optimized_anchors",
    "assign_to_level_by_area",
    "assign_to_level_by_dimension",
    "calculate_fpn_specs",
    "cluster_1d_values",
    "generate_anchors",
    "group_box_dimensions_by_fpn_level",
    "DEFAULT_FPN_LEVEL_IDS",
    "FlatCandidateBatch",
    "flatten_candidate_boxes",
    "AnchorEvaluationReport",
    "reference_tensors_for_grid",
    "evaluate_anchors",
    "format_anchor_evaluation_report",
    "DetectorMatchingReport",
    "LevelMatchingStats",
    "evaluate_detector_anchor_matching",
    "format_detector_matching_report",
    "CellRecallReport",
    "evaluate_cell_recall",
    "format_cell_recall_report",
    "save_cell_recall_histogram",
]
