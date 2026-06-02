import torch
import numpy as np
from torchvision.ops import box_iou
from typing import List, Dict, Tuple, Any

def _compute_tp_fp(
    f_labels: torch.Tensor, 
    gt_labels: torch.Tensor, 
    f_ious: torch.Tensor, 
    iou_threshold: float, 
    num_gts: int) -> Tuple[int, int]:
    """
    Computes True Positives and False Positives using greedy matching.
    f_ious should already be sorted by prediction score descending.
    """
    tp = 0
    fp = 0
    matched_gts = set()
    
    for p_idx in range(len(f_labels)):
        best_gt_idx = -1
        best_iou = iou_threshold # Must be strictly >= iou_threshold
        
        for gt_idx in range(num_gts):
            if gt_idx in matched_gts:
                continue
            if f_labels[p_idx] != gt_labels[gt_idx]:
                continue
            
            iou_val = f_ious[gt_idx, p_idx].item()
            if iou_val >= best_iou:
                best_iou = iou_val
                best_gt_idx = gt_idx
                
        if best_gt_idx != -1:
            tp += 1
            matched_gts.add(best_gt_idx)
        else:
            fp += 1
            
    return tp, fp

def compute_image_diagnostics(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    iou_threshold: float = 0.5,
    score_thresholds: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.975]) -> Dict[str, Any]:
    """
    Computes diagnostic metrics for a single image, broken down by class.
    Returns: Global and per-class Max IoUs, Consensus, and P/R metrics.
    Includes image-level summaries for edge-case tracking.
    """
    gt_boxes = targets['boxes']
    gt_labels = targets['labels']
    
    pred_boxes = preds['boxes']
    pred_labels = preds['labels']
    pred_scores = preds['scores']
    
    num_gts_total = len(gt_boxes)
    if num_gts_total == 0:
        return {} # Nothing to diagnose if no ground truth
        
    unique_gt_classes = torch.unique(gt_labels).tolist()
    
    if len(pred_boxes) == 0:
        ious = torch.zeros((num_gts_total, 0), device=gt_boxes.device)
    else:
        ious = box_iou(gt_boxes, pred_boxes) # [num_gts_total, num_preds]
        
    # 1. Localization Quality: Max IoU achieved per GT box
    if len(pred_boxes) > 0:
        max_ious_per_gt, _ = ious.max(dim=1)
    else:
        max_ious_per_gt = torch.zeros(num_gts_total, device=gt_boxes.device)
    
    max_ious_per_gt_list = max_ious_per_gt.cpu().numpy().tolist()
    per_class_max_ious = {cls: [] for cls in unique_gt_classes}
    for i, label in enumerate(gt_labels.tolist()):
        per_class_max_ious[label].append(max_ious_per_gt_list[i])

    # 2. Label Consensus (Class Stability)
    consensus_list = []
    per_class_consensus = {cls: [] for cls in unique_gt_classes}
    for i in range(num_gts_total):
        overlapping_idx = torch.where(ious[i] >= iou_threshold)[0]
        label = gt_labels[i].item()
        if len(overlapping_idx) > 0:
            overlapping_labels = pred_labels[overlapping_idx]
            correct_preds = (overlapping_labels == label).float().mean().item()
            consensus_list.append(correct_preds)
            per_class_consensus[label].append(correct_preds)
            
    # Image Summaries for edge-case tracking
    avg_max_iou = sum(max_ious_per_gt_list) / num_gts_total
    avg_consensus = sum(consensus_list) / len(consensus_list) if consensus_list else 0.0
    per_class_avg_max_iou = {cls: (sum(ious) / len(ious) if ious else 0.0) 
                             for cls, ious in per_class_max_ious.items()}

    # 3. Precision / Recall at fixed confidence thresholds
    pr_metrics = {t: {'TP': 0, 'FP': 0, 'num_gts': num_gts_total} for t in score_thresholds}
    per_class_pr_metrics = {t: {cls: {'TP': 0, 'FP': 0, 'num_gts': (gt_labels == cls).sum().item()} 
                               for cls in set(unique_gt_classes) | set(pred_labels.tolist())} 
                            for t in score_thresholds}
    
    for thresh in score_thresholds:
        keep = pred_scores >= thresh
        if not keep.any():
            continue
            
        f_labels = pred_labels[keep]
        f_scores = pred_scores[keep]
        f_ious = ious[:, keep]
        
        # Sort by score descending for greedy matching
        sort_idx = torch.argsort(f_scores, descending=True)
        f_labels_sorted = f_labels[sort_idx]
        f_ious_sorted = f_ious[:, sort_idx]
        
        # Global TP/FP
        tp_total, fp_total = _compute_tp_fp(f_labels_sorted, gt_labels, f_ious_sorted, iou_threshold, num_gts_total)
        pr_metrics[thresh]['TP'] = tp_total
        pr_metrics[thresh]['FP'] = fp_total
        
        # Per-class TP/FP
        current_pred_classes = torch.unique(f_labels_sorted).tolist()
        current_gt_classes = torch.unique(gt_labels).tolist()
        all_relevant_classes = set(current_pred_classes) | set(current_gt_classes)
        
        for cls in all_relevant_classes:
            cls_gt_mask = (gt_labels == cls)
            cls_pred_mask = (f_labels_sorted == cls)
            
            num_gts_cls = cls_gt_mask.sum().item()
            
            if not cls_pred_mask.any() and num_gts_cls == 0:
                continue
            
            f_labels_cls = f_labels_sorted[cls_pred_mask]
            f_ious_cls = f_ious_sorted[cls_gt_mask][:, cls_pred_mask]
            gt_labels_cls = gt_labels[cls_gt_mask]
            
            tp_cls, fp_cls = _compute_tp_fp(f_labels_cls, gt_labels_cls, f_ious_cls, iou_threshold, int(num_gts_cls))
            
            if cls not in per_class_pr_metrics[thresh]:
                per_class_pr_metrics[thresh][cls] = {'TP': 0, 'FP': 0, 'num_gts': num_gts_cls}
            
            per_class_pr_metrics[thresh][cls]['TP'] = tp_cls
            per_class_pr_metrics[thresh][cls]['FP'] = fp_cls

    return {
        'max_ious': max_ious_per_gt_list,
        'consensus': consensus_list,
        'avg_max_iou': avg_max_iou,
        'avg_consensus': avg_consensus,
        'pr_metrics': pr_metrics,
        'per_class_max_ious': per_class_max_ious,
        'per_class_avg_max_iou': per_class_avg_max_iou,
        'per_class_consensus': per_class_consensus,
        'per_class_pr_metrics': per_class_pr_metrics
    }

def aggregate_diagnostics(results: List[Dict[str, Any]], score_thresholds: List[float]) -> Dict[str, Any]:
    """
    Aggregates metrics from individual images into global and per-class metrics.
    """
    all_max_ious = []
    all_consensus = []
    agg_pr = {t: {'TP': 0, 'FP': 0, 'num_gts': 0} for t in score_thresholds}
    
    per_class_max_ious_agg = {}
    per_class_consensus_agg = {}
    per_class_pr_agg = {t: {} for t in score_thresholds}
    
    for res in results:
        if not res:
            continue
            
        # Global aggregation
        all_max_ious.extend(res.get('max_ious', []))
        all_consensus.extend(res.get('consensus', []))
        for t in score_thresholds:
            if t in res['pr_metrics']:
                agg_pr[t]['TP'] += res['pr_metrics'][t]['TP']
                agg_pr[t]['FP'] += res['pr_metrics'][t]['FP']
                agg_pr[t]['num_gts'] += res['pr_metrics'][t]['num_gts']
        
        # Per-class aggregation
        for cls, ious in res.get('per_class_max_ious', {}).items():
            if cls not in per_class_max_ious_agg:
                per_class_max_ious_agg[cls] = []
            per_class_max_ious_agg[cls].extend(ious)
            
        for cls, cons in res.get('per_class_consensus', {}).items():
            if cls not in per_class_consensus_agg:
                per_class_consensus_agg[cls] = []
            per_class_consensus_agg[cls].extend(cons)
            
        for t in score_thresholds:
            for cls, metrics in res.get('per_class_pr_metrics', {}).get(t, {}).items():
                if cls not in per_class_pr_agg[t]:
                    per_class_pr_agg[t][cls] = {'TP': 0, 'FP': 0, 'num_gts': 0}
                per_class_pr_agg[t][cls]['TP'] += metrics['TP']
                per_class_pr_agg[t][cls]['FP'] += metrics['FP']
                per_class_pr_agg[t][cls]['num_gts'] += metrics['num_gts']
            
    # Calculate final means
    mean_consensus = sum(all_consensus) / len(all_consensus) if all_consensus else None
    
    final_pr = {}
    for t in score_thresholds:
        tp = agg_pr[t]['TP']
        fp = agg_pr[t]['FP']
        num_gts = agg_pr[t]['num_gts']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / num_gts if num_gts > 0 else 0.0
        final_pr[t] = {'precision': precision, 'recall': recall}
        
    # Calculate per-class final metrics
    final_per_class = {}
    all_seen_classes = set(per_class_max_ious_agg.keys()) | set(per_class_pr_agg[score_thresholds[0]].keys())
    
    for cls in all_seen_classes:
        cls_ious = per_class_max_ious_agg.get(cls, [])
        cls_consensus = per_class_consensus_agg.get(cls, [])
        
        cls_pr = {}
        for t in score_thresholds:
            metrics = per_class_pr_agg[t].get(cls, {'TP': 0, 'FP': 0, 'num_gts': 0})
            tp = metrics['TP']
            fp = metrics['FP']
            num_gts = metrics['num_gts']
            cls_pr[t] = {
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                'recall': tp / num_gts if num_gts > 0 else 0.0
            }
            
        final_per_class[cls] = {
            'mean_max_iou': sum(cls_ious) / len(cls_ious) if cls_ious else 0.0,
            'mean_consensus': sum(cls_consensus) / len(cls_consensus) if cls_consensus else 0.0,
            'pr_metrics': cls_pr
        }
        
    return {
        'max_ious': all_max_ious,
        'consensus': mean_consensus,
        'pr_metrics': final_pr,
        'per_class': final_per_class
    }
