"""
Grid–bbox metrics for object-detection error analysis.

Three relation kinds:

1. **Bbox × Grid** — ``compute_bbox_grid_information``
2. **Bbox × Bbox** — ``compute_bbox_sets_metric``
3. **Bbox × Bbox × Grid** — ``compute_bbox_grid_metrics``

Roles (layer 3)
---------------
**Reference** boxes are grid-anchored slots (e.g. RetinaNet priors): exactly **one cell**
per reference via ``reference_cells``. They do not define a multi-cell footprint here.

**Candidate** boxes are the flexible side of the analysis — GT, predictions, or any xyxy
set. Each candidate gets a snapped **extent** on the grid (cells fully inside the bbox).
The cell metric is computed **per candidate**.

Cell metric (per candidate ``c``)
---------------------------------
Inside candidate ``c``'s footprint, cell ``k`` is scored using the reference anchor at
``k`` (if any): ``best_iou[k] = IoU(reference_at_k, candidate_c)``. Then:

- **background** — ``best_iou < background_iou_threshold`` (default 0.4), or no reference at ``k``
- **foreground** — ``best_iou >= bbox_level_iou_threshold`` (default 0.5)
- **ambiguous** — between the two thresholds

``non_background_ratio`` = (foreground + ambiguous) / ``covered_cell_count``.

Typical anchor–GT call::

    compute_bbox_grid_metrics(
        reference_bboxes=anchors,
        reference_cells=anchor_cells,  # [N_ref, 2] from detector layout
        candidate_bboxes=gt_boxes,
        grid=grid,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torchvision.ops import box_iou

# -----------------------------------------------------------------------------
# Core types
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """One FPN feature-map tiling (``grid_id``, ``y_dim`` × ``x_dim``, strides)."""

    grid_id: str
    y_dim: int
    x_dim: int
    stride_y: float
    stride_x: float

    @property
    def num_cells(self) -> int:
        return self.y_dim * self.x_dim


@dataclass(frozen=True)
class Cell:
    """Integer ``(row, col)`` on a grid."""

    row: int
    col: int

    def flat_id(self, grid: Grid) -> int:
        return self.row * grid.x_dim + self.col


@dataclass(frozen=True)
class ReferenceCellAssignment:
    """
    Maps one reference (anchor) index to its grid cell.

    Filled by the detector adapter (e.g. ``build_retinanet_grid_layout``).
    """

    grid_id: str
    row: int
    col: int


# RetinaNet adapters historically used this name.
PriorCellAssignment = ReferenceCellAssignment


@dataclass
class BboxGridInformation:
    """
    Bbox × grid geometry for one box on one level (from ``compute_bbox_grid_information``).

    **Snapped extent:** inner bbox corners are rounded outward to grid lines
    (``ceil`` on min edges, ``floor`` on max edges). The axis-aligned rectangle between
    those lines defines which cell *tiles* are fully inside the box.

    ``central_cell`` is informational (bbox center); scoring in layer 3 uses the
    extent ranges, not the center alone.
    """

    grid_id: str
    bbox_idx: int
    central_cell: Cell
    covered_cell_count: int
    extent_min_x: float
    extent_min_y: float
    extent_max_x: float
    extent_max_y: float
    col_start: int
    col_end_exclusive: int
    row_start: int
    row_end_exclusive: int
    covers_any_cell: bool


@dataclass
class BBoxSetsMetric:
    """
    Bbox × Bbox: pairwise IoU between references and candidates.

    ``iou_matrix[r, c]`` = IoU(reference ``r``, candidate ``c``), shape ``[N_ref, N_cand]``.
    """

    reference_bboxes: torch.Tensor  # [N_ref, 4]
    candidate_bboxes: torch.Tensor  # [N_cand, 4]
    iou_matrix: torch.Tensor  # [N_ref, N_cand]


@dataclass
class BBoxGridMetrics:
    """
    Bbox × Bbox × Grid for one FPN level.

    Per-**candidate** cell assignment counts and ratios (see module docstring).
    """

    grid: Grid
    candidate_grid_info: List[BboxGridInformation]
    bbox_sets: BBoxSetsMetric
    bbox_level_iou_threshold: float
    background_iou_threshold: float
    cells_foreground: torch.Tensor  # [N_cand] long
    cells_background: torch.Tensor  # [N_cand] long
    cells_ambiguous: torch.Tensor  # [N_cand] long
    cells_non_background: torch.Tensor  # [N_cand] long
    non_background_ratio: torch.Tensor  # [N_cand] float
    max_reference_iou_per_candidate: torch.Tensor  # [N_cand] float
    num_references_above_bbox_threshold: torch.Tensor  # [N_cand] long


# -----------------------------------------------------------------------------
# Bbox × Grid
# -----------------------------------------------------------------------------


def snap_bbox_extent_to_grid(
    bboxes: torch.Tensor,
    grid: Grid,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Snap each bbox to a stride-aligned footprint.

    Parameters
    ----------
    bboxes :
        ``[N, 4]`` xyxy.

    Returns
    -------
    min_x, min_y, max_x, max_y :
        Each ``[N]`` float.
    valid :
        ``[N]`` bool.
    """
    n = bboxes.shape[0]
    device = bboxes.device
    if n == 0:
        zf = torch.zeros(0, device=device, dtype=bboxes.dtype)
        return zf, zf, zf, zf, torch.zeros(0, dtype=torch.bool, device=device)

    sx, sy = float(grid.stride_x), float(grid.stride_y)
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    min_x = torch.ceil(x1 / sx) * sx
    min_y = torch.ceil(y1 / sy) * sy
    max_x = torch.floor(x2 / sx) * sx
    max_y = torch.floor(y2 / sy) * sy
    valid = (max_x > min_x) & (max_y > min_y)
    return min_x, min_y, max_x, max_y, valid


def bbox_grid_coverage_from_extent(
    min_x: torch.Tensor,
    min_y: torch.Tensor,
    max_x: torch.Tensor,
    max_y: torch.Tensor,
    valid: torch.Tensor,
    grid: Grid,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cell counts and index ranges from a snapped extent.

    Parameters
    ----------
    min_x, min_y, max_x, max_y, valid :
        From ``snap_bbox_extent_to_grid``; each ``[N]``.

    Returns
    -------
    counts, col_start, col_end_exclusive, row_start, row_end_exclusive :
        Each ``[N]`` long.
    """
    n = min_x.shape[0]
    device = min_x.device
    zero = torch.zeros(n, dtype=torch.long, device=device)
    if n == 0:
        return zero, zero, zero, zero, zero

    sx, sy = float(grid.stride_x), float(grid.stride_y)
    counts = torch.where(
        valid,
        ((max_x - min_x) / sx).long() * ((max_y - min_y) / sy).long(),
        zero,
    )

    col_start = torch.where(valid, (min_x / sx).long(), zero).clamp(0, grid.x_dim)
    col_end = torch.where(valid, (max_x / sx).long(), zero).clamp(0, grid.x_dim)
    row_start = torch.where(valid, (min_y / sy).long(), zero).clamp(0, grid.y_dim)
    row_end = torch.where(valid, (max_y / sy).long(), zero).clamp(0, grid.y_dim)

    return counts, col_start, col_end, row_start, row_end


def covered_cell_counts(bboxes: torch.Tensor, grid: Grid) -> torch.Tensor:
    """``[N]`` long — fully-covered cells per bbox."""
    min_x, min_y, max_x, max_y, valid = snap_bbox_extent_to_grid(bboxes, grid)
    counts, _, _, _, _ = bbox_grid_coverage_from_extent(
        min_x, min_y, max_x, max_y, valid, grid
    )
    return counts


def bbox_central_cells(
    bboxes: torch.Tensor,
    grid: Grid,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[N]`` center cell (row, col) per bbox — metadata only in layer 3."""
    n = bboxes.shape[0]
    device = bboxes.device
    if n == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z

    sx, sy = float(grid.stride_x), float(grid.stride_y)
    cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    cols = (cx / sx).floor().long().clamp(0, grid.x_dim - 1)
    rows = (cy / sy).floor().long().clamp(0, grid.y_dim - 1)
    return rows, cols


def _records_from_snap_tensors(
    grid: Grid,
    min_x: torch.Tensor,
    min_y: torch.Tensor,
    max_x: torch.Tensor,
    max_y: torch.Tensor,
    valid: torch.Tensor,
    counts: torch.Tensor,
    col_start: torch.Tensor,
    col_end: torch.Tensor,
    row_start: torch.Tensor,
    row_end: torch.Tensor,
    central_rows: torch.Tensor,
    central_cols: torch.Tensor,
) -> List[BboxGridInformation]:
    n = min_x.shape[0]
    min_x_l = min_x.tolist()
    min_y_l = min_y.tolist()
    max_x_l = max_x.tolist()
    max_y_l = max_y.tolist()
    valid_l = valid.tolist()
    counts_l = counts.tolist()
    col_s_l = col_start.tolist()
    col_e_l = col_end.tolist()
    row_s_l = row_start.tolist()
    row_e_l = row_end.tolist()
    crow_l = central_rows.tolist()
    ccol_l = central_cols.tolist()

    return [
        BboxGridInformation(
            grid_id=grid.grid_id,
            bbox_idx=i,
            central_cell=Cell(row=crow_l[i], col=ccol_l[i]),
            covered_cell_count=counts_l[i],
            extent_min_x=min_x_l[i],
            extent_min_y=min_y_l[i],
            extent_max_x=max_x_l[i],
            extent_max_y=max_y_l[i],
            col_start=col_s_l[i],
            col_end_exclusive=col_e_l[i],
            row_start=row_s_l[i],
            row_end_exclusive=row_e_l[i],
            covers_any_cell=valid_l[i],
        )
        for i in range(n)
    ]


def compute_bbox_grid_information(
    bboxes: torch.Tensor,
    grid: Grid,
) -> List[BboxGridInformation]:
    """
    **Bbox × Grid:** extent + cell ranges per bbox row.

    Parameters
    ----------
    bboxes :
        ``[N, 4]`` xyxy (typically candidates: GT, preds, …).
    """
    n = bboxes.shape[0]
    if n == 0:
        return []

    min_x, min_y, max_x, max_y, valid = snap_bbox_extent_to_grid(bboxes, grid)
    counts, col_start, col_end, row_start, row_end = bbox_grid_coverage_from_extent(
        min_x, min_y, max_x, max_y, valid, grid
    )
    central_rows, central_cols = bbox_central_cells(bboxes, grid)
    return _records_from_snap_tensors(
        grid,
        min_x,
        min_y,
        max_x,
        max_y,
        valid,
        counts,
        col_start,
        col_end,
        row_start,
        row_end,
        central_rows,
        central_cols,
    )


# -----------------------------------------------------------------------------
# Bbox × Bbox
# -----------------------------------------------------------------------------


def compute_bbox_sets_metric(
    reference_bboxes: torch.Tensor,
    candidate_bboxes: torch.Tensor,
) -> BBoxSetsMetric:
    """
    **Bbox × Bbox:** ``iou_matrix[r, c] = IoU(reference_r, candidate_c)``.

    Parameters
    ----------
    reference_bboxes :
        ``[N_ref, 4]`` — anchors / grid slots.
    candidate_bboxes :
        ``[N_cand, 4]`` — GT, predictions, etc.
    """
    n_ref, n_cand = reference_bboxes.shape[0], candidate_bboxes.shape[0]
    device = reference_bboxes.device

    if n_ref == 0 or n_cand == 0:
        iou_matrix = torch.zeros((n_ref, n_cand), device=device)
    else:
        # box_iou(rows, cols) -> [len(rows), len(cols)]
        iou_matrix = box_iou(reference_bboxes, candidate_bboxes)

    return BBoxSetsMetric(
        reference_bboxes=reference_bboxes,
        candidate_bboxes=candidate_bboxes,
        iou_matrix=iou_matrix,
    )


# -----------------------------------------------------------------------------
# Bbox × Bbox × Grid
# -----------------------------------------------------------------------------


def _reference_flat_cell_ids(
    grid: Grid,
    reference_cells: torch.Tensor,
) -> torch.Tensor:
    """
    Flat cell id per reference (anchor).

    Parameters
    ----------
    reference_cells :
        ``[N_ref, 2]`` long — ``(row, col)`` per reference; required.

    Returns
    -------
    cell_ids :
        ``[N_ref]`` long.
    """
    if reference_cells.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=reference_cells.device)
    return reference_cells[:, 0].long() * grid.x_dim + reference_cells[:, 1].long()


def _best_iou_per_cell_from_references(
    iou_ref_vs_candidate: torch.Tensor,
    reference_cell_ids: torch.Tensor,
    num_cells: int,
) -> torch.Tensor:
    """
    Per-cell IoU between one candidate and the reference at that cell.

    For candidate ``c``, ``iou_ref_vs_candidate[r] = iou_matrix[r, c]``. Each reference
    sits on one cell; ``best[k]`` is the IoU of the reference anchor at ``k`` to this
    candidate (``-inf`` if no reference occupies ``k``). If several references share a
    cell, the max IoU is kept.

    Parameters
    ----------
    iou_ref_vs_candidate :
        ``[N_ref]`` — column ``c`` of ``bbox_sets.iou_matrix``.
    reference_cell_ids :
        ``[N_ref]`` long.
    num_cells :
        ``grid.num_cells``.

    Returns
    -------
    best :
        ``[num_cells]`` float.
    """
    if iou_ref_vs_candidate.numel() == 0:
        return torch.full((num_cells,), float("-inf"), device=iou_ref_vs_candidate.device)

    best = torch.full(
        (num_cells,),
        float("-inf"),
        device=iou_ref_vs_candidate.device,
        dtype=iou_ref_vs_candidate.dtype,
    )
    best.scatter_reduce_(
        0, reference_cell_ids, iou_ref_vs_candidate, reduce="amax", include_self=True
    )
    return best


def _cell_assignment_counts_in_extent(
    footprint: BboxGridInformation,
    best_per_cell: torch.Tensor,
    grid: Grid,
    bbox_level_iou_threshold: float,
    background_iou_threshold: float,
) -> Tuple[int, int, int, float]:
    """
    Classify cells inside a candidate footprint using ``best_per_cell``.

    Returns
    -------
    n_foreground, n_background, n_ambiguous, non_background_ratio
    """
    if not footprint.covers_any_cell:
        return 0, 0, 0, 0.0

    n_fg = n_bg = n_amb = 0
    x_dim = grid.x_dim
    for row in range(footprint.row_start, footprint.row_end_exclusive):
        base = row * x_dim
        for col in range(footprint.col_start, footprint.col_end_exclusive):
            flat = base + col
            best_iou = float(best_per_cell[flat].item())
            if best_iou < background_iou_threshold:
                n_bg += 1
            elif best_iou >= bbox_level_iou_threshold:
                n_fg += 1
            else:
                n_amb += 1

    n_in = footprint.covered_cell_count
    non_bg = n_fg + n_amb
    ratio = non_bg / n_in if n_in > 0 else 0.0
    return n_fg, n_bg, n_amb, ratio


def compute_bbox_grid_metrics(
    reference_bboxes: torch.Tensor,
    reference_cells: torch.Tensor,
    candidate_bboxes: torch.Tensor,
    grid: Grid,
    bbox_level_iou_threshold: float = 0.5,
    background_iou_threshold: float = 0.4,
) -> BBoxGridMetrics:
    """
    **Bbox × Bbox × Grid** on one FPN level.

    Algorithm (per candidate index ``c``)
    -------------------------------------
    1. **Candidate footprint** — ``compute_bbox_grid_information(candidate_bboxes)``
       → extent and ``covered_cell_count`` for each candidate (GT, pred, …).
    2. **Pairwise IoU** — ``iou_matrix[r, c] = IoU(reference_r, candidate_c)``.
    3. **Reference → cell** — ``reference_cells`` (one cell per anchor); required.
    4. **Per-cell score** — for fixed ``c``, ``best[k] = IoU(reference at k, candidate_c)``.
    5. **Classify** — only cells inside candidate ``c``'s footprint; bg / fg / amb counts.
    6. **BBox-level extras** — max IoU over references; count of references above threshold.

    Parameters
    ----------
    reference_bboxes :
        ``[N_ref, 4]`` xyxy — anchors on this level.
    reference_cells :
        ``[N_ref, 2]`` long — ``(row, col)`` per reference (detector layout).
    candidate_bboxes :
        ``[N_cand, 4]`` xyxy — GT, predictions, or any evaluated boxes.
    bbox_level_iou_threshold :
        Foreground cell cutoff (e.g. 0.5).
    background_iou_threshold :
        Below this → background cell (e.g. 0.4).
    """
    if background_iou_threshold > bbox_level_iou_threshold:
        raise ValueError(
            "background_iou_threshold must be <= bbox_level_iou_threshold "
            f"({background_iou_threshold} > {bbox_level_iou_threshold})"
        )

    n_ref = reference_bboxes.shape[0]
    n_cand = candidate_bboxes.shape[0]
    device = reference_bboxes.device

    if reference_cells.shape[0] != n_ref:
        raise ValueError(
            f"reference_cells rows ({reference_cells.shape[0]}) != "
            f"reference_bboxes ({n_ref})"
        )

    # Step 1 — candidate footprints (extent), one record per candidate box.
    cand_info = (
        compute_bbox_grid_information(candidate_bboxes, grid) if n_cand > 0 else []
    )

    # Step 2 — IoU between every reference anchor and every candidate box.
    bbox_sets = compute_bbox_sets_metric(reference_bboxes, candidate_bboxes)

    cells_fg = torch.zeros(n_cand, dtype=torch.long, device=device)
    cells_bg = torch.zeros(n_cand, dtype=torch.long, device=device)
    cells_amb = torch.zeros(n_cand, dtype=torch.long, device=device)
    cells_non_bg = torch.zeros(n_cand, dtype=torch.long, device=device)
    non_bg_ratio = torch.zeros(n_cand, dtype=torch.float32, device=device)
    max_ref_iou = torch.zeros(n_cand, dtype=torch.float32, device=device)
    num_ref_above = torch.zeros(n_cand, dtype=torch.long, device=device)

    if n_ref > 0 and n_cand > 0:
        max_ref_iou = bbox_sets.iou_matrix.max(dim=0).values
        num_ref_above = (
            bbox_sets.iou_matrix >= bbox_level_iou_threshold
        ).sum(dim=0).long()

        # Step 3 — flat cell index per reference anchor.
        ref_cell_ids = _reference_flat_cell_ids(grid, reference_cells)
        num_cells = grid.num_cells

        # Step 4 — best IoU per cell for every candidate column (batched scatter).
        best_all = torch.full(
            (n_cand, num_cells),
            float("-inf"),
            device=device,
            dtype=bbox_sets.iou_matrix.dtype,
        )
        for cand_idx in range(n_cand):
            best_all[cand_idx].scatter_reduce_(
                0,
                ref_cell_ids,
                bbox_sets.iou_matrix[:, cand_idx],
                reduce="amax",
                include_self=True,
            )

        # Step 5 — classify footprint cells per candidate (one GT ≠ another).
        for cand_idx in range(n_cand):
            n_fg, n_bg, n_amb, ratio = _cell_assignment_counts_in_extent(
                cand_info[cand_idx],
                best_all[cand_idx],
                grid,
                bbox_level_iou_threshold,
                background_iou_threshold,
            )
            cells_fg[cand_idx] = n_fg
            cells_bg[cand_idx] = n_bg
            cells_amb[cand_idx] = n_amb
            cells_non_bg[cand_idx] = n_fg + n_amb
            non_bg_ratio[cand_idx] = ratio

    return BBoxGridMetrics(
        grid=grid,
        candidate_grid_info=cand_info,
        bbox_sets=bbox_sets,
        bbox_level_iou_threshold=bbox_level_iou_threshold,
        background_iou_threshold=background_iou_threshold,
        cells_foreground=cells_fg,
        cells_background=cells_bg,
        cells_ambiguous=cells_amb,
        cells_non_background=cells_non_bg,
        non_background_ratio=non_bg_ratio,
        max_reference_iou_per_candidate=max_ref_iou,
        num_references_above_bbox_threshold=num_ref_above,
    )


def level_stats_dict(metrics: BBoxGridMetrics, candidate_idx: int) -> dict:
    """Per-level scalar breakdown for one candidate (e.g. one GT) index."""
    info = metrics.candidate_grid_info[candidate_idx]
    return {
        "non_background_ratio": float(metrics.non_background_ratio[candidate_idx].item()),
        "cells_non_background": float(metrics.cells_non_background[candidate_idx].item()),
        "cells_foreground": float(metrics.cells_foreground[candidate_idx].item()),
        "cells_background": float(metrics.cells_background[candidate_idx].item()),
        "cells_ambiguous": float(metrics.cells_ambiguous[candidate_idx].item()),
        "cells_in_footprint": float(info.covered_cell_count),
        "max_reference_iou": float(
            metrics.max_reference_iou_per_candidate[candidate_idx].item()
        ),
    }


def max_non_background_ratio_over_levels(
    metrics_per_level: Sequence[BBoxGridMetrics],
    candidate_idx: int,
) -> Tuple[float, Optional[str], dict]:
    """
    Max ``non_background_ratio`` for one candidate across FPN levels.

    Returns ``(best_ratio, best_grid_id, per_level_stats_dict)``.
    """
    best_ratio = 0.0
    best_grid_id: Optional[str] = None
    per_level: dict = {}

    for metrics in metrics_per_level:
        if candidate_idx >= len(metrics.candidate_grid_info):
            continue
        stats = level_stats_dict(metrics, candidate_idx)
        per_level[metrics.grid.grid_id] = stats
        ratio = stats["non_background_ratio"]
        if ratio >= best_ratio:
            best_ratio = ratio
            best_grid_id = metrics.grid.grid_id

    return best_ratio, best_grid_id, per_level


__all__ = [
    "Grid",
    "Cell",
    "ReferenceCellAssignment",
    "PriorCellAssignment",
    "BboxGridInformation",
    "BBoxSetsMetric",
    "BBoxGridMetrics",
    "snap_bbox_extent_to_grid",
    "bbox_grid_coverage_from_extent",
    "covered_cell_counts",
    "bbox_central_cells",
    "compute_bbox_grid_information",
    "compute_bbox_sets_metric",
    "compute_bbox_grid_metrics",
    "level_stats_dict",
    "max_non_background_ratio_over_levels",
]
