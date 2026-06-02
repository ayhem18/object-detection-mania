# """
# Grid metric extreme panels.

# For each metric, rank GT boxes by the **image-level** value (not per FPN level), pick
# k best and k worst, then render that same GT on every FPN level so you can see how
# coverage differs across the pyramid (e.g. good on P5, sparse on P6).
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Optional, Sequence, Tuple

# import cv2
# import numpy as np
# import torch
# from torchvision.ops import box_iou

# from dl_lib.etalon_object_detection.modules.od_metrics.grid_metrics import (
#     LevelGridSpec,
#     PriorCellRef,
#     best_iou_per_cell,
#     cells_in_gt_bbox_set,
# )

# GRID_METRIC_NAMES = (
#     "gt_area_coverage",
#     "num_overlapping_priors",
#     "cell_recall_in_gt",
# )


# @dataclass
# class GridExtremeCandidate:
#     """One GT instance with grid context for multi-level panels."""

#     image: torch.Tensor
#     gt_idx: int
#     gt_box: torch.Tensor
#     anchor_boxes: torch.Tensor
#     prior_to_cell: Sequence[PriorCellRef]
#     level_grids: Sequence[LevelGridSpec]
#     per_level: Dict[str, Dict[str, float]]
#     img_stem: str
#     grid_iou_threshold: float
#     # Image-level metrics (ranking dimension — not aggregated over FPN)
#     gt_area_coverage: float
#     num_overlapping_priors: int
#     cell_recall_in_gt: float
#     cell_recall_best_level: Optional[str] = None


# def _image_to_bgr(image: torch.Tensor) -> np.ndarray:
#     img_np = image.permute(1, 2, 0).cpu().numpy()
#     img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
#     return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


# def _cell_rect(spec: LevelGridSpec, row: int, col: int) -> Tuple[int, int, int, int]:
#     x1 = int(col * spec.stride_x)
#     y1 = int(row * spec.stride_y)
#     x2 = int(min((col + 1) * spec.stride_x, spec.image_width))
#     y2 = int(min((row + 1) * spec.stride_y, spec.image_height))
#     return x1, y1, x2, y2


# def _level_spec(level_grids: Sequence[LevelGridSpec], level_id: str) -> LevelGridSpec:
#     for spec in level_grids:
#         if spec.level_id == level_id:
#             return spec
#     raise KeyError(f"Unknown level_id {level_id}")


# def _draw_grid_lines(img_bgr: np.ndarray, spec: LevelGridSpec) -> None:
#     color = (70, 70, 70)
#     for row in range(spec.grid_height):
#         for col in range(spec.grid_width):
#             x1, y1, x2, y2 = _cell_rect(spec, row, col)
#             cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 1)


# def _draw_gt_box(img_bgr: np.ndarray, gt_box: torch.Tensor, gt_idx: int) -> None:
#     x1, y1, x2, y2 = gt_box.numpy().astype(int)
#     cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
#     cv2.putText(
#         img_bgr,
#         f"GT#{gt_idx}",
#         (x1, max(y1 - 8, 12)),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.55,
#         (0, 255, 0),
#         2,
#     )


# def _draw_title(img_bgr: np.ndarray, title: str) -> None:
#     cv2.putText(img_bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#     cv2.putText(img_bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


# def _anchor_indices_for_level(
#     prior_to_cell: Sequence[PriorCellRef],
#     level_id: str,
# ) -> List[int]:
#     return [i for i, ref in enumerate(prior_to_cell) if ref.level_id == level_id]


# def _best_anchor_on_level(
#     iou_row: torch.Tensor,
#     prior_to_cell: Sequence[PriorCellRef],
#     level_id: str,
# ) -> Optional[Tuple[int, float]]:
#     """Anchor on ``level_id`` with highest IoU to the GT (for fallback / cell-recall)."""
#     level_indices = _anchor_indices_for_level(prior_to_cell, level_id)
#     if not level_indices:
#         return None
#     best_idx = max(level_indices, key=lambda i: float(iou_row[i].item()))
#     return best_idx, float(iou_row[best_idx].item())


# def _draw_prior_box(
#     img_bgr: np.ndarray,
#     anchor_boxes: torch.Tensor,
#     anchor_idx: int,
#     color: Tuple[int, int, int] = (255, 128, 0),
#     thickness: int = 2,
# ) -> None:
#     box = anchor_boxes[anchor_idx].numpy().astype(int)
#     cv2.rectangle(img_bgr, (box[0], box[1]), (box[2], box[3]), color, thickness)


# def _global_metric(candidate: GridExtremeCandidate, metric: str) -> float:
#     return float(getattr(candidate, metric))


# def render_gt_area_coverage_level(
#     candidate: GridExtremeCandidate,
#     level_id: str,
#     save_path: Path,
#     title_prefix: str,
# ) -> None:
#     """Grid + GT + best-anchor-per-cell priors (IoU >= thresh) on this level."""
#     spec = _level_spec(candidate.level_grids, level_id)
#     img_bgr = _image_to_bgr(candidate.image)
#     _draw_grid_lines(img_bgr, spec)
#     _draw_gt_box(img_bgr, candidate.gt_box, candidate.gt_idx)

#     iou_row = box_iou(
#         candidate.gt_box.unsqueeze(0), candidate.anchor_boxes
#     )[0]
#     best = best_iou_per_cell(iou_row, candidate.prior_to_cell)
#     cells_in_gt = cells_in_gt_bbox_set(candidate.gt_box, candidate.level_grids)
#     thresh = candidate.grid_iou_threshold

#     drew_any = False
#     for key, (anchor_idx, iou) in best.items():
#         if key[0] != level_id or key not in cells_in_gt or iou < thresh:
#             continue
#         _draw_prior_box(img_bgr, candidate.anchor_boxes, anchor_idx, thickness=2)
#         drew_any = True

#     if not drew_any:
#         fallback = _best_anchor_on_level(iou_row, candidate.prior_to_cell, level_id)
#         if fallback is not None:
#             _draw_prior_box(
#                 img_bgr, candidate.anchor_boxes, fallback[0], color=(0, 165, 255), thickness=2
#             )

#     level_val = candidate.per_level[level_id]["gt_area_coverage"]
#     global_val = candidate.gt_area_coverage
#     title = (
#         f"{title_prefix} | {candidate.img_stem} | {level_id} | GT#{candidate.gt_idx} | "
#         f"global={global_val:.3f} level={level_val:.3f}"
#     )
#     _draw_title(img_bgr, title)
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(save_path), img_bgr)


# def render_num_overlapping_priors_level(
#     candidate: GridExtremeCandidate,
#     level_id: str,
#     save_path: Path,
#     title_prefix: str,
# ) -> None:
#     """Grid + GT + all priors on this level with IoU >= threshold."""
#     spec = _level_spec(candidate.level_grids, level_id)
#     img_bgr = _image_to_bgr(candidate.image)
#     _draw_grid_lines(img_bgr, spec)
#     _draw_gt_box(img_bgr, candidate.gt_box, candidate.gt_idx)

#     iou_row = box_iou(
#         candidate.gt_box.unsqueeze(0), candidate.anchor_boxes
#     )[0]
#     thresh = candidate.grid_iou_threshold
#     level_indices = _anchor_indices_for_level(candidate.prior_to_cell, level_id)

#     drew_any = False
#     for idx in level_indices:
#         if float(iou_row[idx].item()) >= thresh:
#             _draw_prior_box(img_bgr, candidate.anchor_boxes, idx, thickness=1)
#             drew_any = True

#     if not drew_any:
#         fallback = _best_anchor_on_level(iou_row, candidate.prior_to_cell, level_id)
#         if fallback is not None:
#             _draw_prior_box(
#                 img_bgr, candidate.anchor_boxes, fallback[0], color=(0, 165, 255), thickness=2
#             )

#     level_n = int(candidate.per_level[level_id]["num_overlapping_priors"])
#     title = (
#         f"{title_prefix} | {candidate.img_stem} | {level_id} | GT#{candidate.gt_idx} | "
#         f"global={int(candidate.num_overlapping_priors)} level={level_n}"
#     )
#     _draw_title(img_bgr, title)
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(save_path), img_bgr)


# def render_cell_recall_level(
#     candidate: GridExtremeCandidate,
#     level_id: str,
#     save_path: Path,
#     title_prefix: str,
# ) -> None:
#     """
#     Grid + GT + cell fill + single best-IoU prior on this level.

#     Cells inside GT: green = touched (best-anchor IoU >= thresh), red = missed.
#     The highest-IoU prior is drawn on top to show anchor/GT alignment.
#     """
#     spec = _level_spec(candidate.level_grids, level_id)
#     base = _image_to_bgr(candidate.image)
#     _draw_grid_lines(base, spec)

#     iou_row = box_iou(
#         candidate.gt_box.unsqueeze(0), candidate.anchor_boxes
#     )[0]
#     best_per_cell = best_iou_per_cell(iou_row, candidate.prior_to_cell)
#     cells_in_gt = cells_in_gt_bbox_set(candidate.gt_box, candidate.level_grids)
#     thresh = candidate.grid_iou_threshold

#     overlay = base.copy()
#     for key in cells_in_gt:
#         if key[0] != level_id:
#             continue
#         row, col = key[1], key[2]
#         x1, y1, x2, y2 = _cell_rect(spec, row, col)
#         entry = best_per_cell.get(key)
#         touched = entry is not None and entry[1] >= thresh
#         color = (0, 180, 0) if touched else (0, 0, 220)
#         cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

#     img_bgr = cv2.addWeighted(overlay, 0.4, base, 0.6, 0)
#     _draw_gt_box(img_bgr, candidate.gt_box, candidate.gt_idx)

#     best_on_level = _best_anchor_on_level(iou_row, candidate.prior_to_cell, level_id)
#     if best_on_level is not None:
#         anchor_idx, best_iou = best_on_level
#         _draw_prior_box(img_bgr, candidate.anchor_boxes, anchor_idx, thickness=2)
#         iou_note = f" best_iou={best_iou:.2f}"
#     else:
#         iou_note = ""

#     pl = candidate.per_level[level_id]
#     best_lv = candidate.cell_recall_best_level or "?"
#     title = (
#         f"{title_prefix} | {candidate.img_stem} | {level_id} | GT#{candidate.gt_idx} | "
#         f"max_recall={candidate.cell_recall_in_gt:.3f}@{best_lv} "
#         f"level={pl['cell_recall_in_gt']:.3f} "
#         f"({int(pl['cells_touched'])}/{int(pl['cells_in_gt_bbox'])}){iou_note}"
#     )
#     _draw_title(img_bgr, title)
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(save_path), img_bgr)


# _RENDERERS = {
#     "gt_area_coverage": render_gt_area_coverage_level,
#     "num_overlapping_priors": render_num_overlapping_priors_level,
#     "cell_recall_in_gt": render_cell_recall_level,
# }


# def save_grid_metric_extremes(
#     candidates: List[GridExtremeCandidate],
#     k_extremes: int,
#     output_dir: Path,
# ) -> None:
#     """
#     For each grid metric, pick k best / k worst **GT boxes** (global metric), then
#     save one panel per FPN level under that GT's folder.
#     """
#     if not candidates:
#         return

#     level_ids = list(candidates[0].per_level.keys())
#     extremes_root = output_dir / "grid" / "extremes"

#     for metric in GRID_METRIC_NAMES:
#         render_fn = _RENDERERS[metric]
#         ranked = sorted(candidates, key=lambda c: _global_metric(c, metric))
#         worst = ranked[:k_extremes]
#         best = ranked[-k_extremes:][::-1]

#         for rank_label, group in (("worst", worst), ("best", best)):
#             for i, cand in enumerate(group):
#                 global_val = _global_metric(cand, metric)
#                 if metric == "num_overlapping_priors":
#                     val_tag = f"{metric}{int(global_val)}"
#                 else:
#                     val_tag = f"{metric}{global_val:.3f}"
#                 gt_dir = (
#                     extremes_root
#                     / metric
#                     / f"{rank_label}_{i:02d}_{cand.img_stem}_gt{cand.gt_idx}_{val_tag}"
#                 )
#                 for level_id in level_ids:
#                     render_fn(
#                         cand,
#                         level_id,
#                         gt_dir / f"{level_id}.png",
#                         rank_label.upper(),
#                     )

