"""
Shared helpers for pre-training anchor evaluation.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from home_made_od.od_metrics.grid_metrics import (
    Grid,
    ReferenceCellAssignment,
)

DEFAULT_FPN_LEVEL_IDS = ("P3", "P4", "P5", "P6", "P7", "None (< P3)")


@dataclass
class FlatCandidateBatch:
    """
    All candidate (GT) boxes in one tensor for batched ``grid_metrics``.

    Attributes
    ----------
    boxes :
        ``[N, 4]`` xyxy in the same eval image space as ``reference_bboxes``.
    image_index :
        ``[N]`` long — which image each row belongs to (0 .. num_images-1).
    box_index :
        ``[N]`` long — GT index within that image (0 .. n_gt_on_image-1).
    image_stems :
        Optional ``[num_images]`` stems for logging / JSON export (not used in metrics).
    """

    boxes: torch.Tensor
    image_index: torch.Tensor
    box_index: torch.Tensor
    image_stems: Optional[Sequence[str]] = None

    @property
    def num_boxes(self) -> int:
        return int(self.boxes.shape[0])

    @property
    def num_images(self) -> int:
        if self.image_index.numel() == 0:
            return 0
        return int(self.image_index.max().item()) + 1


def flatten_candidate_boxes(
    boxes_per_image: Sequence[torch.Tensor],
    image_stems: Optional[Sequence[str]] = None,
) -> FlatCandidateBatch:
    """
    Concatenate per-image GT tensors into one batch.

    Parameters
    ----------
    boxes_per_image :
        One ``[N_i, 4]`` tensor per image (may be empty — skipped).
    image_stems :
        Optional labels, length = number of images passed in (including empty).
    """
    if image_stems is not None and len(image_stems) != len(boxes_per_image):
        raise ValueError(
            f"image_stems length ({len(image_stems)}) != "
            f"boxes_per_image ({len(boxes_per_image)})"
        )

    chunks: List[torch.Tensor] = []
    image_indices: List[int] = []
    box_indices: List[int] = []

    for img_i, boxes in enumerate(boxes_per_image):
        n = boxes.shape[0]
        if n == 0:
            continue
        chunks.append(boxes)
        image_indices.extend([img_i] * n)
        box_indices.extend(range(n))

    if not chunks:
        device = (
            boxes_per_image[0].device
            if boxes_per_image
            else torch.device("cpu")
        )
        return FlatCandidateBatch(
            boxes=torch.zeros(0, 4, device=device),
            image_index=torch.zeros(0, dtype=torch.long, device=device),
            box_index=torch.zeros(0, dtype=torch.long, device=device),
            image_stems=image_stems,
        )

    device = chunks[0].device
    return FlatCandidateBatch(
        boxes=torch.cat(chunks, dim=0),
        image_index=torch.tensor(image_indices, dtype=torch.long, device=device),
        box_index=torch.tensor(box_indices, dtype=torch.long, device=device),
        image_stems=image_stems,
    )


@dataclass
class AnchorEvaluationReport:
    """Combined output from ``evaluate_anchors``."""

    detector_matching: Any = None
    cell_recall: Any = None
    output_dir: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.detector_matching is not None:
            out["detector_matching"] = self.detector_matching.to_dict()
        if self.cell_recall is not None:
            out["cell_recall"] = self.cell_recall.to_dict()
        if self.output_dir is not None:
            out["output_dir"] = str(self.output_dir)
        return out


def reference_tensors_for_grid(
    reference_bboxes: torch.Tensor,
    assignments: Sequence[ReferenceCellAssignment],
    grid: Grid,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[N_level, 4]`` anchors and ``[N_level, 2]`` cells on one FPN grid."""
    device = reference_bboxes.device
    rows: List[int] = []
    cells: List[Tuple[int, int]] = []
    for i, assign in enumerate(assignments):
        if assign.grid_id == grid.grid_id:
            rows.append(i)
            cells.append((assign.row, assign.col))

    if not rows:
        return (
            torch.zeros(0, 4, device=device),
            torch.zeros(0, 2, dtype=torch.long, device=device),
        )

    idx = torch.tensor(rows, dtype=torch.long, device=device)
    cell_t = torch.tensor(cells, dtype=torch.long, device=device)
    return reference_bboxes[idx], cell_t


def gt_passes_cell_recall(cell_recall: torch.Tensor, metric_threshold: float) -> bool:
    """Scalar or 0-d tensor pass check."""
    if isinstance(cell_recall, torch.Tensor):
        return bool((cell_recall >= metric_threshold).item())
    return float(cell_recall) >= metric_threshold
