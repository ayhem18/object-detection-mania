"""Anchor computation and pre-training evaluation."""

from dl_lib.etalon_object_detection.modules.anchors.anchor_computation_strategies import (
    ASSIGNMENT_METHODS,
    AnchorOptimizationResult,
    FPN_LEVEL_ORDER,
    METHOD_REQUIRED_PARAMETERS,
    anchor_config_output_dir,
    assign_to_level_by_area,
    assign_to_level_by_dimension,
    compute_anchor_config_hash,
    compute_optimized_anchors,
    validate_method_parameters,
)
from dl_lib.etalon_object_detection.modules.anchors.anchor_eval_utils import (
    DEFAULT_FPN_LEVEL_IDS,
    AnchorEvaluationReport,
    FlatCandidateBatch,
    flatten_candidate_boxes,
    reference_tensors_for_grid,
)
from dl_lib.etalon_object_detection.modules.anchors.anchor_evaluation import (
    evaluate_anchors,
    format_anchor_evaluation_report,
)
from dl_lib.etalon_object_detection.modules.anchors.anchor_matching_test import (
    DetectorMatchingReport,
    LevelMatchingStats,
    evaluate_detector_anchor_matching,
    format_detector_matching_report,
)
from dl_lib.etalon_object_detection.modules.anchors.cell_recall_test import (
    CellRecallReport,
    evaluate_cell_recall,
    format_cell_recall_report,
    save_cell_recall_histogram,
)

__all__ = [
    "ASSIGNMENT_METHODS",
    "AnchorOptimizationResult",
    "FPN_LEVEL_ORDER",
    "METHOD_REQUIRED_PARAMETERS",
    "validate_method_parameters",
    "anchor_config_output_dir",
    "compute_anchor_config_hash",
    "compute_optimized_anchors",
    "assign_to_level_by_area",
    "assign_to_level_by_dimension",
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
