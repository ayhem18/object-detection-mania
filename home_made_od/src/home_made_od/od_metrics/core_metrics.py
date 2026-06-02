"""
Model-agnostic primitives for object-detection error analysis.

This module has NO dependency on RetinaNet, torchvision detectors, or project-specific
training code. The only external numeric helper used is ``torchvision.ops.box_iou``.

-------------------------------------------------------------------------------
ADAPTER CONTRACT — what every detector must provide
-------------------------------------------------------------------------------

**Ground truth (per image)**
    gt_boxes : Tensor[float], shape [N_gt, 4]
        Axis-aligned boxes in ``xyxy`` format: (x1, y1, x2, y2).
        Coordinates must live in the SAME space as predictions (network input pixels,
        letterboxed canvas, etc.) — the toolkit does not resize or remap boxes.

    gt_labels : Tensor[int], shape [N_gt]   (optional for objectness-only analysis)
        Class ids. Background id (if any) is defined by the caller via
        ``foreground_label_ids`` when filtering predictions.

**Post-processed predictions (per image) — used by Layer A**
    pred_boxes : Tensor[float], shape [N_pred, 4]   (xyxy, same coordinate space)
    pred_scores : Tensor[float], shape [N_pred]
        Confidence / objectness score AFTER whatever post-processing your model applies
        (sigmoid, softmax foreground score, NMS survivor score, etc.).
        Higher = more confident. The toolkit never recomputes scores from logits.

    pred_labels : Tensor[int], shape [N_pred]   (optional)
        Predicted class per box. If provided together with ``foreground_label_ids``,
        predictions outside the foreground set are ignored for objectness stats.

**Anchor grid (per image) — used by Layer B**
    anchor_boxes : Tensor[float], shape [N_anchor, 4]   (xyxy)
        All anchor/reference boxes for this image in inference order.

    anchor_scores : Tensor[float], shape [N_anchor]
        One scalar per anchor expressing "how much the model wants to fire here".
        Examples you might map from different architectures:
          - RetinaNet: max foreground class probability at that anchor (pre-NMS)
          - YOLO: objectness × class prob at cell
          - Custom: any monotone score; thresholds are always interpreted on YOUR scale

    positive_anchor_indices_per_gt : list[Tensor[int]]   (optional)
        If your training code already ran an assigner (MaxIoU matcher, ATSS, etc.),
        pass the positive anchor indices per GT and we will NOT re-derive them.
        Otherwise use ``assign_positive_anchors_per_gt`` with a simple IoU rule.

**Stratification metadata (optional, both layers)**
    gt_metadata : list[dict] | None, length N_gt
        Free-form dicts keyed by strings, e.g.:
          {"assigned_fpn_level": "P4", "area": 1200.0, "aspect_ratio": 3.2}
        Used only for grouped histograms; never required for core math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
from torchvision.ops import box_iou


# ---------------------------------------------------------------------------
# Result records — stable outputs you can log, plot, or join across images
# ---------------------------------------------------------------------------


@dataclass
class PerGtPredStats:
    """
    Per-GT summary after comparing ONE image's GT to its post-processed predictions.

    Attributes
    ----------
    gt_idx :
        Index into ``gt_boxes`` for this record (0 .. N_gt-1).

    best_iou :
        Maximum IoU between this GT and ANY prediction box (oracle localization).
        0.0 if there are zero predictions. Does NOT use ``iou_overlap_min``.

    best_score :
        Maximum prediction score among boxes whose IoU with this GT is
        >= ``iou_overlap_min`` (see ``per_gt_best_pred_stats``).
        0.0 if no prediction passes the overlap gate (objectness "never fired nearby").

    best_score_pred_idx :
        Index into ``pred_boxes`` of the prediction that achieved ``best_score``,
        or ``None`` if no qualifying prediction exists.

    best_iou_pred_idx :
        Index into ``pred_boxes`` of the prediction that achieved ``best_iou``,
        or ``None`` if there are no predictions.

    gt_label :
        Optional class id copied from ``gt_labels[gt_idx]`` when labels were passed in.

    metadata :
        Optional caller-provided dict (FPN level, area bin, etc.) for stratified plots.
    """

    gt_idx: int
    best_iou: float
    best_score: float
    best_score_pred_idx: Optional[int] = None
    best_iou_pred_idx: Optional[int] = None
    gt_label: Optional[int] = None
    metadata: Optional[Dict] = field(default=None)


@dataclass
class PerGtAnchorStats:
    """
    Per-GT summary after comparing ONE image's GT to anchor/grid scores (Layer B).

    Attributes
    ----------
    gt_idx :
        Index into ``gt_boxes``.

    num_positive_anchors :
        Count of anchors considered positive for this GT (see assignment helpers).

    max_positive_score / mean_positive_score :
        Statistics over ``anchor_scores`` at positive anchor indices.
        0.0 when there are no positive anchors (assignment or coverage failure).

    max_anchor_iou :
        Best IoU between this GT and ANY anchor (structural coverage — like anchor
        sanity checks, independent of whether the head scored those anchors high).

    positive_anchor_indices :
        Indices into ``anchor_boxes`` / ``anchor_scores`` used for the aggregates.
    """

    gt_idx: int
    num_positive_anchors: int
    max_positive_score: float
    mean_positive_score: float
    max_anchor_iou: float
    positive_anchor_indices: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))
    gt_label: Optional[int] = None
    metadata: Optional[Dict] = field(default=None)


# ---------------------------------------------------------------------------
# IoU primitives
# ---------------------------------------------------------------------------


def compute_pred_gt_iou_matrix(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise IoU matrix between GT and post-processed predictions.

    Parameters
    ----------
    gt_boxes :
        [N_gt, 4] xyxy
    pred_boxes :
        [N_pred, 4] xyxy

    Returns
    -------
    Tensor[float], shape [N_gt, N_pred]
        iou_matrix[i, j] = IoU(gt_boxes[i], pred_boxes[j])

    Notes for model adapters
    ------------------------
    - If your model outputs ``xywh`` or normalized boxes, convert to xyxy in the
      SAME pixel grid as ``gt_boxes`` before calling this toolkit.
    - Empty predictions are valid: returns shape [N_gt, 0].
    """
    if gt_boxes.numel() == 0:
        return torch.zeros((0, pred_boxes.shape[0]), device=gt_boxes.device, dtype=torch.float32)
    if pred_boxes.numel() == 0:
        return torch.zeros((gt_boxes.shape[0], 0), device=gt_boxes.device, dtype=torch.float32)
    return box_iou(gt_boxes, pred_boxes)


def compute_gt_anchor_iou_matrix(
    gt_boxes: torch.Tensor,
    anchor_boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise IoU matrix between GT and anchor/reference boxes (Layer B).

    Returns
    -------
    Tensor[float], shape [N_gt, N_anchor]
    """
    if gt_boxes.numel() == 0:
        return torch.zeros((0, anchor_boxes.shape[0]), device=gt_boxes.device, dtype=torch.float32)
    if anchor_boxes.numel() == 0:
        return torch.zeros((gt_boxes.shape[0], 0), device=gt_boxes.device, dtype=torch.float32)
    return box_iou(gt_boxes, anchor_boxes)


# ---------------------------------------------------------------------------
# Prediction filtering (class-agnostic objectness path)
# ---------------------------------------------------------------------------


def _filter_predictions(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: Optional[torch.Tensor],
    foreground_label_ids: Optional[Set[int]],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Optionally drop background-class predictions before objectness analysis.

    If ``foreground_label_ids`` is None, all predictions are kept (true class-agnostic).
    If ``pred_labels`` is None but ``foreground_label_ids`` is set, no filtering occurs
    (caller must supply labels to exclude background).
    """
    if foreground_label_ids is None or pred_labels is None:
        return pred_boxes, pred_scores, pred_labels

    keep = torch.tensor(
        [int(lbl.item()) in foreground_label_ids for lbl in pred_labels],
        device=pred_labels.device,
        dtype=torch.bool,
    )
    if not keep.any():
        empty = pred_boxes.new_zeros((0, 4))
        return empty, pred_scores.new_zeros((0,)), None
    return pred_boxes[keep], pred_scores[keep], pred_labels[keep]


# ---------------------------------------------------------------------------
# Layer A core: per-GT best IoU and best score
# ---------------------------------------------------------------------------


def per_gt_best_pred_stats(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: Optional[torch.Tensor] = None,
    gt_labels: Optional[torch.Tensor] = None,
    foreground_label_ids: Optional[Set[int]] = None,
    iou_overlap_min: float = 0.1,
    gt_metadata: Optional[Sequence[Optional[Dict]]] = None,
) -> List[PerGtPredStats]:
    """
    Layer A building block — one record per GT summarizing prediction overlap and score.

    This answers two SEPARATE questions for each GT:

    1. **Localization ceiling (oracle):** ``best_iou`` — best overlap with ANY pred.
    2. **Objectness at GT:** ``best_score`` — best pred score among boxes that overlap
       the GT by at least ``iou_overlap_min``.

    Parameters
    ----------
    iou_overlap_min :
        Minimum IoU for a prediction to be considered "covering" the GT when taking
        the max score. Use a small value (0.1) to measure objectness even when boxes
        are poorly localized; use 0.5 to require rough alignment before crediting score.

    foreground_label_ids :
        If set (e.g. ``{1, 2, 3}`` excluding background ``0``), only those predicted
        labels participate. Objectness phase-1 often sets this to None to ignore class.

    gt_metadata :
        Optional per-GT dicts for downstream stratification (FPN level, size quartile).

    Model adapter checklist
    -----------------------
    [ ] Convert model outputs to xyxy tensors on the evaluation image size.
    [ ] ``pred_scores`` must already be calibrated the way you threshold at deploy time.
    [ ] Run YOUR NMS / score filter before calling, OR pass all raw candidates with low
        ``iou_overlap_min`` — the toolkit does not run NMS.
    """
    num_gt = gt_boxes.shape[0]
    if num_gt == 0:
        return []

    pred_boxes, pred_scores, _ = _filter_predictions(
        pred_boxes, pred_scores, pred_labels, foreground_label_ids
    )

    iou_matrix = compute_pred_gt_iou_matrix(gt_boxes, pred_boxes)
    records: List[PerGtPredStats] = []

    for gt_idx in range(num_gt):
        meta = None
        if gt_metadata is not None and gt_idx < len(gt_metadata):
            meta = gt_metadata[gt_idx]

        label = None
        if gt_labels is not None and gt_labels.numel() > 0:
            label = int(gt_labels[gt_idx].item())

        # --- Oracle localization: best IoU over ALL (filtered) predictions ---
        if pred_boxes.shape[0] == 0:
            records.append(
                PerGtPredStats(
                    gt_idx=gt_idx,
                    best_iou=0.0,
                    best_score=0.0,
                    best_score_pred_idx=None,
                    best_iou_pred_idx=None,
                    gt_label=label,
                    metadata=meta,
                )
            )
            continue

        gt_ious = iou_matrix[gt_idx]  # [N_pred]
        best_iou_val, best_iou_pred_idx = gt_ious.max(dim=0)
        best_iou_f = float(best_iou_val.item())
        best_iou_pi = int(best_iou_pred_idx.item())

        # --- Objectness: best score among preds overlapping GT enough ---
        overlap_mask = gt_ious >= iou_overlap_min
        if overlap_mask.any():
            overlapping_scores = pred_scores[overlap_mask]
            overlapping_indices = torch.where(overlap_mask)[0]
            local_max_idx = int(overlapping_scores.argmax().item())
            best_score_pi = int(overlapping_indices[local_max_idx].item())
            best_score_f = float(pred_scores[best_score_pi].item())
        else:
            best_score_pi = None
            best_score_f = 0.0

        records.append(
            PerGtPredStats(
                gt_idx=gt_idx,
                best_iou=best_iou_f,
                best_score=best_score_f,
                best_score_pred_idx=best_score_pi,
                best_iou_pred_idx=best_iou_pi,
                gt_label=label,
                metadata=meta,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Layer B core: positive anchor assignment
# ---------------------------------------------------------------------------


def assign_positive_anchors_per_gt(
    gt_boxes: torch.Tensor,
    anchor_boxes: torch.Tensor,
    fg_iou_threshold: float = 0.5) -> List[torch.Tensor]:
    """
    Simple positive-anchor assignment by IoU threshold (model-agnostic default).

    For each GT ``i``, positive anchors are all ``j`` with
    ``IoU(gt_boxes[i], anchor_boxes[j]) >= fg_iou_threshold``.

    This is intentionally simpler than RetinaNet's MaxIoU matcher (low-quality
    fallback, etc.). If your trainer uses a custom matcher, compute positive indices
    in your training code and pass them directly to ``per_gt_anchor_objectness_records``.

    Parameters
    ----------
    fg_iou_threshold :
        Same semantic as training ``fg_iou_thresh`` when you want fidelity; can be
        lowered for exploratory analysis.

    Returns
    -------
    list[Tensor[int]]
        ``positives[i]`` = 1D long tensor of anchor indices for GT ``i`` (may be empty).
    """
    num_gt = gt_boxes.shape[0]
    if num_gt == 0:
        return []

    iou_matrix = compute_gt_anchor_iou_matrix(gt_boxes, anchor_boxes)
    positives: List[torch.Tensor] = []

    for gt_idx in range(num_gt):
        mask = iou_matrix[gt_idx] >= fg_iou_threshold
        positives.append(torch.where(mask)[0])

    return positives


def per_gt_anchor_objectness_records(
    gt_boxes: torch.Tensor,
    anchor_boxes: torch.Tensor,
    anchor_scores: torch.Tensor,
    gt_labels: Optional[torch.Tensor] = None,
    positive_anchor_indices_per_gt: Optional[List[torch.Tensor]] = None,
    fg_iou_threshold: float = 0.5,
    gt_metadata: Optional[Sequence[Optional[Dict]]] = None) -> List[PerGtAnchorStats]:
    """
    Layer B building block — per-GT statistics over anchor/grid scores.

    Parameters
    ----------
    anchor_scores :
        Shape [N_anchor]. Must align index-wise with ``anchor_boxes``.

    positive_anchor_indices_per_gt :
        When provided (from your model's assigner), ``fg_iou_threshold`` is ignored for
        selection and only used for documentation. Length must be ``N_gt``.

    Model adapter checklist
    -----------------------
    [ ] Run backbone+head to obtain per-anchor scores BEFORE NMS.
    [ ] Flatten all FPN levels into one ``anchor_boxes`` / ``anchor_scores`` pair.
    [ ] Keep the same anchor order your assigner uses during training.
    [ ] Map logits to probabilities if your deploy threshold is on [0, 1].
    """
    num_gt = gt_boxes.shape[0]
    if num_gt == 0:
        return []

    if anchor_scores.shape[0] != anchor_boxes.shape[0]:
        raise ValueError(
            f"anchor_scores length ({anchor_scores.shape[0]}) must match "
            f"anchor_boxes ({anchor_boxes.shape[0]})."
        )

    if positive_anchor_indices_per_gt is None:
        positive_anchor_indices_per_gt = assign_positive_anchors_per_gt(
            gt_boxes, anchor_boxes, fg_iou_threshold=fg_iou_threshold
        )
    elif len(positive_anchor_indices_per_gt) != num_gt:
        raise ValueError(
            f"positive_anchor_indices_per_gt length ({len(positive_anchor_indices_per_gt)}) "
            f"must equal number of GT boxes ({num_gt})."
        )

    iou_matrix = compute_gt_anchor_iou_matrix(gt_boxes, anchor_boxes)
    records: List[PerGtAnchorStats] = []

    for gt_idx in range(num_gt):
        meta = None
        if gt_metadata is not None and gt_idx < len(gt_metadata):
            meta = gt_metadata[gt_idx]

        label = None
        if gt_labels is not None and gt_labels.numel() > 0:
            label = int(gt_labels[gt_idx].item())

        max_anchor_iou = float(iou_matrix[gt_idx].max().item()) if anchor_boxes.shape[0] > 0 else 0.0

        pos_idx = positive_anchor_indices_per_gt[gt_idx]
        # Assigner indices may be on GPU while adapter passes CPU anchor_scores.
        if pos_idx.numel() > 0 and pos_idx.device != anchor_scores.device:
            pos_idx = pos_idx.to(anchor_scores.device)
        if pos_idx.numel() == 0:
            records.append(
                PerGtAnchorStats(
                    gt_idx=gt_idx,
                    num_positive_anchors=0,
                    max_positive_score=0.0,
                    mean_positive_score=0.0,
                    max_anchor_iou=max_anchor_iou,
                    positive_anchor_indices=pos_idx.cpu(),
                    gt_label=label,
                    metadata=meta,
                )
            )
            continue

        pos_scores = anchor_scores[pos_idx]
        records.append(
            PerGtAnchorStats(
                gt_idx=gt_idx,
                num_positive_anchors=int(pos_idx.numel()),
                max_positive_score=float(pos_scores.max().item()),
                mean_positive_score=float(pos_scores.mean().item()),
                max_anchor_iou=max_anchor_iou,
                positive_anchor_indices=pos_idx.cpu(),
                gt_label=label,
                metadata=meta,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Aggregation helpers (batch / dataset level)
# ---------------------------------------------------------------------------


def flatten_per_image_records(
    per_image: Sequence[Sequence[Union[PerGtPredStats, PerGtAnchorStats]]],
) -> List[Union[PerGtPredStats, PerGtAnchorStats]]:
    """Concatenate per-image record lists into one dataset-level list."""
    flat: List[Union[PerGtPredStats, PerGtAnchorStats]] = []
    for image_records in per_image:
        flat.extend(image_records)
    return flat
