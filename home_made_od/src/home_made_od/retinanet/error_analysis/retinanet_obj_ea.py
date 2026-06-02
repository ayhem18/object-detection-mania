"""
RetinaNet-specific adapters for objectness / anchor / grid error analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
from torchvision.ops import box_iou

from home_made_od.od_metrics.grid_metrics import (
    Grid,
    ReferenceCellAssignment,
)
from home_made_od.retinanet.error_analysis.retinanet_grid_index import (
    build_retinanet_grid_layout,
    level_ids_from_manifest,
    verify_anchor_cell_count,
)


@dataclass
class RetinaNetLayerBBundle:
    """Per-image tensors for Layer B + grid metrics."""

    anchor_boxes: torch.Tensor
    anchor_scores: torch.Tensor
    positive_anchor_indices_per_gt: List[torch.Tensor]
    prior_to_cell: List[ReferenceCellAssignment]
    level_grids: List[Grid]
    gt_boxes: torch.Tensor
    gt_labels: torch.Tensor


def configure_retinanet_postprocess(
    model: nn.Module,
    score_thresh: float,
    nms_thresh: float,
) -> None:
    """Set post-process thresholds; use ``nms_thresh=1.0`` to disable NMS suppression."""
    model.score_thresh = score_thresh
    model.nms_thresh = nms_thresh
    model.topk_candidates = 10_000
    model.detections_per_img = 10_000


def infer_foreground_label_ids(
    num_classes: int,
    explicit: Optional[Set[int]] = None,
) -> Optional[Set[int]]:
    if explicit is not None:
        return explicit
    if num_classes <= 1:
        return None
    return set(range(1, num_classes))


def anchor_scores_from_cls_logits(cls_logits: torch.Tensor) -> torch.Tensor:
    """Max foreground class sigmoid per anchor (class 0 = background)."""
    probs = torch.sigmoid(cls_logits)
    if probs.shape[-1] <= 1:
        return probs[:, 0]
    return probs[:, 1:].amax(dim=-1)


def extract_retinanet_layer_b_bundles(
    model: nn.Module,
    images: List[torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    device: torch.device,
    anchor_manifest_path: Optional[str] = None,
) -> List[RetinaNetLayerBBundle]:
    """
    Forward head + anchors for one batch; return Layer B and grid indexing per image.
    """
    model.eval()
    images_dev = [img.to(device) for img in images]
    targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]
    level_ids = level_ids_from_manifest(anchor_manifest_path)

    with torch.no_grad():
        transformed_images, transformed_targets = model.transform(images_dev, targets_dev)
        img_h, img_w = transformed_images.tensors.shape[-2:]
        features = list(model.backbone(transformed_images.tensors).values())
        head_outputs = model.head(features)
        anchors_list = model.anchor_generator(transformed_images, features)
        cls_logits_batch = head_outputs["cls_logits"]

        prior_to_cell, level_grids = build_retinanet_grid_layout(
            model, img_h, img_w, features, level_ids=level_ids
        )

        bundles: List[RetinaNetLayerBBundle] = []
        for i, (anchors_i, targets_i) in enumerate(zip(anchors_list, transformed_targets)):
            logits_i = cls_logits_batch[i]
            anchor_scores_i = anchor_scores_from_cls_logits(logits_i)
            verify_anchor_cell_count(anchors_i.shape[0], prior_to_cell)

            positive_per_gt: List[torch.Tensor] = []
            gt_boxes_i = targets_i["boxes"]
            if gt_boxes_i.numel() > 0:
                match_matrix = box_iou(gt_boxes_i, anchors_i)
                matched_idxs = model.proposal_matcher(match_matrix)
                for gt_idx in range(gt_boxes_i.shape[0]):
                    positive_per_gt.append(
                        torch.where(matched_idxs == gt_idx)[0].cpu()
                    )

            bundles.append(
                RetinaNetLayerBBundle(
                    anchor_boxes=anchors_i.cpu(),
                    anchor_scores=anchor_scores_i.cpu(),
                    positive_anchor_indices_per_gt=positive_per_gt,
                    prior_to_cell=prior_to_cell,
                    level_grids=level_grids,
                    gt_boxes=gt_boxes_i.cpu(),
                    gt_labels=targets_i["labels"].cpu(),
                )
            )

    return bundles


def gt_metadata_from_boxes(gt_boxes: torch.Tensor) -> List[Optional[Dict[str, float]]]:
    if gt_boxes.numel() == 0:
        return []
    metadata: List[Optional[Dict[str, float]]] = []
    for box in gt_boxes:
        x1, y1, x2, y2 = box.tolist()
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        metadata.append(
            {"area": w * h, "aspect_ratio": w / h, "width": w, "height": h}
        )
    return metadata
