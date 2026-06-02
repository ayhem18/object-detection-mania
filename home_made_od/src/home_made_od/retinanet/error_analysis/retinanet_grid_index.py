"""
Map torchvision RetinaNet flat anchor indices to FPN grid cells.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from home_made_od.od_metrics.grid_metrics import (
    Grid,
    ReferenceCellAssignment,
)

from home_made_od.retinanet.retinanet_anchors import load_anchor_config


def level_ids_from_anchor_config(config_path: Optional[str]) -> Optional[List[str]]:
    if config_path is None:
        return None
    config = load_anchor_config(config_path)
    levels = config.get("used_fpn_levels")
    if not levels:
        return None
    return [str(level) for level in levels]


def level_ids_from_manifest(config_path: Optional[str]) -> Optional[List[str]]:
    """Deprecated alias for :func:`level_ids_from_anchor_config`."""
    return level_ids_from_anchor_config(config_path)


def _strides_for_level(
    image_height: int,
    image_width: int,
    grid_height: int,
    grid_width: int,
) -> Tuple[float, float]:
    stride_y = image_height // grid_height
    stride_x = image_width // grid_width
    return float(stride_y), float(stride_x)


def build_retinanet_grid_layout(
    model: nn.Module,
    image_height: int,
    image_width: int,
    feature_maps: Sequence[torch.Tensor],
    level_ids: List[str],
) -> Tuple[List[ReferenceCellAssignment], List[Grid]]:
    anchor_gen = model.anchor_generator
    anchors_per_location = anchor_gen.num_anchors_per_location()
    num_levels = len(feature_maps)

    if num_levels != len(anchors_per_location):
        raise ValueError(
            f"feature_maps ({num_levels}) and anchors_per_location "
            f"({len(anchors_per_location)}) length mismatch."
        )
    if len(level_ids) != num_levels:
        raise ValueError(
            f"level_ids length ({len(level_ids)}) must match num_levels ({num_levels})."
        )

    prior_to_cell: List[ReferenceCellAssignment] = []
    grids: List[Grid] = []

    for level_i, feat in enumerate(feature_maps):
        grid_h = int(feat.shape[-2])
        grid_w = int(feat.shape[-1])
        stride_y, stride_x = _strides_for_level(
            image_height, image_width, grid_h, grid_w
        )
        level_name = level_ids[level_i]
        n_per_cell = anchors_per_location[level_i]

        grids.append(
            Grid(
                grid_id=level_name,
                y_dim=grid_h,
                x_dim=grid_w,
                stride_y=stride_y,
                stride_x=stride_x,
            )
        )

        for row in range(grid_h):
            for col in range(grid_w):
                for _ in range(n_per_cell):
                    prior_to_cell.append(
                        ReferenceCellAssignment(grid_id=level_name, row=row, col=col)
                    )

    return prior_to_cell, grids


def anchor_index_ranges_per_level(
    model: nn.Module,
    feature_maps: Sequence[torch.Tensor],
    level_ids: Optional[List[str]] = None,
) -> Dict[str, Tuple[int, int]]:
    num_levels = len(feature_maps)
    if level_ids is None:
        level_ids = [f"P{3 + i}" for i in range(num_levels)]

    anchors_per_location = model.anchor_generator.num_anchors_per_location()
    ranges: Dict[str, Tuple[int, int]] = {}
    offset = 0

    for level_i, feat in enumerate(feature_maps):
        grid_h = int(feat.shape[-2])
        grid_w = int(feat.shape[-1])
        n_level = grid_h * grid_w * anchors_per_location[level_i]
        ranges[level_ids[level_i]] = (offset, offset + n_level)
        offset += n_level

    return ranges


def cell_center_xy(grid: Grid, row: int, col: int) -> Tuple[float, float]:
    cx = (col + 0.5) * grid.stride_x
    cy = (row + 0.5) * grid.stride_y
    return cx, cy


def verify_anchor_cell_count(
    num_anchors: int,
    prior_to_cell: Sequence[ReferenceCellAssignment],
) -> None:
    if len(prior_to_cell) != num_anchors:
        raise ValueError(
            f"prior_to_cell length ({len(prior_to_cell)}) != num_anchors ({num_anchors})."
        )


def verify_layout_against_anchors(
    model: nn.Module,
    image_height: int,
    image_width: int,
    feature_maps: Sequence[torch.Tensor],
    anchor_boxes: torch.Tensor,
    level_ids: Optional[List[str]] = None,
) -> Tuple[List[ReferenceCellAssignment], List[Grid]]:
    if level_ids is None:
        level_ids = [f"P{3 + i}" for i in range(len(feature_maps))]
    prior_to_cell, grids = build_retinanet_grid_layout(
        model, image_height, image_width, feature_maps, level_ids=level_ids
    )
    verify_anchor_cell_count(anchor_boxes.shape[0], prior_to_cell)
    return prior_to_cell, grids
