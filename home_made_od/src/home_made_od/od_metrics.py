import torch
import torchvision.ops as ops
from typing import Tuple, Dict, List
from tqdm import tqdm

def compute_iou_matrix(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the Intersection over Union (IoU) matrix between predictions and ground truth.
    
    Args:
        pred_boxes: Tensor of shape (N, 4) representing [x1, y1, x2, y2].
        gt_boxes: Tensor of shape (M, 4) representing [x1, y1, x2, y2].
        
    Returns:
        Tensor of shape (N, M) where matrix[i, j] is the IoU between 
        the i-th prediction and the j-th ground truth.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return torch.zeros((len(pred_boxes), len(gt_boxes)), device=pred_boxes.device)
    return ops.box_iou(pred_boxes, gt_boxes)


def match_predictions_to_gt(pred_boxes: torch.Tensor, 
                            pred_scores: torch.Tensor, 
                            gt_boxes: torch.Tensor, 
                            iou_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Greedily matches predictions to ground truth boxes to determine True Positives (TP) 
    and False Positives (FP).
    
    The logic follows these steps:
    1. Sort all predictions by confidence score in descending order.
    2. For each prediction (starting from the most confident), find the GT box 
       with the highest IoU.
    3. If the highest IoU is >= threshold:
       - If the GT box has not been claimed by a higher-confidence prediction, 
         mark this prediction as a True Positive and "claim" the GT box.
       - If the GT box was already claimed, mark this prediction as a False Positive 
         (it's a duplicate detection).
    4. If the highest IoU is < threshold, mark the prediction as a False Positive.
    
    Args:
        pred_boxes: Tensor of shape (N, 4).
        pred_scores: Tensor of shape (N,).
        gt_boxes: Tensor of shape (M, 4).
        iou_threshold: Minimum IoU to consider a match valid.
        
    Returns:
        tp: Binary tensor (N,) where 1 indicates a True Positive.
        fp: Binary tensor (N,) where 1 indicates a False Positive.
        sort_idx: The indices used to sort the original predictions.
    """
    device = pred_boxes.device
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)
    
    # 1. Sort predictions by confidence descending
    sort_idx = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sort_idx]
    pred_scores = pred_scores[sort_idx]
    
    tp = torch.zeros(num_preds, device=device)
    fp = torch.zeros(num_preds, device=device)
    
    if num_preds == 0:
        return tp, fp, sort_idx
    if num_gts == 0:
        fp = torch.ones(num_preds, device=device)
        return tp, fp, sort_idx

    # 2. Compute N x M IoU matrix
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    
    # Track which GT boxes have already been "claimed"
    gt_matched = torch.zeros(num_gts, dtype=torch.bool, device=device)
    
    # 3. Greedy Matching Loop
    for i in range(num_preds):
        # Find the GT with the highest IoU for this specific prediction
        best_iou, best_gt_idx = torch.max(iou_matrix[i], dim=0)
        
        if best_iou >= iou_threshold:
            if not gt_matched[best_gt_idx]:
                # Valid match: GT was not claimed by a higher-confidence prediction
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                # GT was already claimed: this prediction is a duplicate/redundant
                fp[i] = 1
        else:
            # IoU too low: prediction does not significantly overlap any GT
            fp[i] = 1
            
    return tp, fp, sort_idx


def compute_pr_curve(tp: torch.Tensor, fp: torch.Tensor, total_gts: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Precision and Recall curves at each confidence rank.
    
    Recall is non-decreasing because as we lower the confidence threshold, 
    we can only find more of the available Ground Truths.
    
    Precision can fluctuate (zigzag) because a low-confidence prediction 
    can be a True Positive even if higher-confidence ones were False Positives.
    
    Args:
        tp: Binary tensor (N,) of True Positives, sorted by confidence.
        fp: Binary tensor (N,) of False Positives, sorted by confidence.
        total_gts: Total number of GT boxes for this class in the dataset.
        
    Returns:
        precision: Tensor where precision[i] is accuracy up to rank i.
        recall: Tensor where recall[i] is fraction of GTs found up to rank i.
    """
    # Cumulative sum to compute metrics as we lower the confidence threshold
    cum_tp = torch.cumsum(tp, dim=0)
    cum_fp = torch.cumsum(fp, dim=0)
    
    recall = cum_tp / total_gts if total_gts > 0 else torch.zeros_like(cum_tp)
    
    # Avoid division by zero
    epsilon = 1e-16
    precision = cum_tp / (cum_tp + cum_fp + epsilon)
    
    return precision, recall


def compute_ap(precision: torch.Tensor, recall: torch.Tensor) -> float:
    """
    Computes the Average Precision (AP) using the all-point interpolation formula (COCO style).
    
    This function creates a "Precision Envelope" to ensure the curve is monotonically 
    decreasing (removing zigzags) before calculating the area under the curve.
    
    Abbreviation meanings:
        mrec (Modified Recall): Recall padded with [0, ..., 1].
        mpre (Modified Precision): Precision padded with [0, ..., 0].
    """
    device = precision.device
    # 1. Pad recall and precision with boundaries for area calculation
    mrec = torch.cat([torch.tensor([0.0], device=device), recall, torch.tensor([1.0], device=device)])
    mpre = torch.cat([torch.tensor([0.0], device=device), precision, torch.tensor([0.0], device=device)])
    
    # 2. Compute the Precision Envelope (Interpolation)
    # Sets precision at recall level r to be the maximum precision for any recall >= r.
    # This smooths out fluctuations where precision increases as recall increases.
    for i in range(mpre.size(0) - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])
        
    # 3. Calculate Area Under Curve (AUC) using Riemann sum
    # Identify indices where recall value changes to define rectangle widths
    indices = torch.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum of (delta_recall * max_precision)
    ap = torch.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap.item()


def collect_detections_and_gts(model: torch.nn.Module, 
                               dataloader: torch.utils.data.DataLoader, 
                               anchors: list, 
                               num_classes: int,
                               conf_threshold: float = 0.05, 
                               nms_iou_threshold: float = 0.4, 
                               device: torch.device = torch.device('cpu')) -> Tuple[Dict, Dict]:
    """
    Runs inference on the dataloader and collects all predictions and ground truths.
    
    Tensors are moved to CPU immediately to preserve GPU memory for large datasets.
    """
    model.eval()
    all_preds = {c: {"boxes": [], "scores": [], "img_idxs": []} for c in range(num_classes)}
    all_gts = {c: [] for c in range(num_classes)}
    img_idx_offset = 0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Collecting Detections"):
            images = images.to(device)
            B = images.size(0)
            img_h, img_w = images.size(2), images.size(3)
            
            batch_preds = model.inference(images, anchors, conf_threshold, nms_iou_threshold)
            targets = targets.to(device)
            
            for b in range(B):
                global_img_idx = img_idx_offset + b
                
                # Extract GTs: [x1, y1, x2, y2]
                img_targets = targets[targets[:, 0] == b]
                if len(img_targets) > 0:
                    for t in img_targets:
                        c = int(t[1].item())
                        cx, cy, w, h = t[2] * img_w, t[3] * img_h, t[4] * img_w, t[5] * img_h
                        box = torch.tensor([cx - w/2, cy - h/2, cx + w/2, cy + h/2], device='cpu')
                        all_gts[c].append({"img_idx": global_img_idx, "box": box})
                
                # Extract Preds: [x1, y1, x2, y2, score, class_id]
                img_preds = batch_preds[b]
                if len(img_preds) > 0:
                    for p in img_preds:
                        c = int(p[5].item())
                        all_preds[c]["boxes"].append(p[:4].detach().cpu().unsqueeze(0))
                        all_preds[c]["scores"].append(p[4].detach().cpu().unsqueeze(0))
                        all_preds[c]["img_idxs"].append(global_img_idx)

            img_idx_offset += B
            
    return all_preds, all_gts


def compute_mAP(all_preds: Dict, all_gts: Dict, num_classes: int, match_iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Computes per-class AP and mAP from collected detections and ground truths.
    
    This function utilizes the match_predictions_to_gt building block for 
    per-image matching while maintaining global confidence ranking.
    """
    metrics = {}
    aps = []
    
    for c in range(num_classes):
        gts_c = all_gts[c]
        preds_c = all_preds[c]
        total_gts_c = len(gts_c)
        
        if total_gts_c == 0:
            continue
            
        if not preds_c["boxes"]:
            metrics[f"AP_class_{c}"] = 0.0
            aps.append(0.0)
            continue
            
        # Group data by image index to respect image boundaries during matching
        gts_by_img = {}
        for gt in gts_c:
            gts_by_img.setdefault(gt["img_idx"], []).append(gt["box"])
            
        preds_by_img = {}
        for b, s, idx in zip(preds_c["boxes"], preds_c["scores"], preds_c["img_idxs"]):
            preds_by_img.setdefault(idx, {"boxes": [], "scores": []})
            preds_by_img[idx]["boxes"].append(b)
            preds_by_img[idx]["scores"].append(s)
            
        class_tp = []
        class_fp = []
        class_scores = []
        
        # All images that have either a prediction or a GT for this class
        all_img_ids = set(preds_by_img.keys()) | set(gts_by_img.keys())
        
        for img_id in all_img_ids:
            img_gt_boxes = gts_by_img.get(img_id, [])
            img_preds = preds_by_img.get(img_id)
            
            if img_preds is None:
                continue
                
            p_boxes_img = torch.cat(img_preds["boxes"], dim=0)
            p_scores_img = torch.cat(img_preds["scores"], dim=0)
            
            if not img_gt_boxes:
                # No GTs in this image -> everything is a False Positive
                tp_img = torch.zeros(len(p_boxes_img))
                fp_img = torch.ones(len(p_boxes_img))
                scores_img = p_scores_img
            else:
                gt_boxes_img = torch.stack(img_gt_boxes)
                # Call the building block to perform greedy matching for this image
                tp_img, fp_img, sort_idx = match_predictions_to_gt(
                    p_boxes_img, p_scores_img, gt_boxes_img, match_iou_threshold
                )
                scores_img = p_scores_img[sort_idx]
                
            class_tp.append(tp_img)
            class_fp.append(fp_img)
            class_scores.append(scores_img)
            
        # Combine results from all images for global sorting
        global_tp = torch.cat(class_tp)
        global_fp = torch.cat(class_fp)
        global_scores = torch.cat(class_scores)
        
        # Sort globally by score to compute the Precision-Recall curve
        global_sort_idx = torch.argsort(global_scores, descending=True)
        global_tp = global_tp[global_sort_idx]
        global_fp = global_fp[global_sort_idx]
        
        precision, recall = compute_pr_curve(global_tp, global_fp, total_gts_c)
        ap = compute_ap(precision, recall)
        
        # Calculate scalar Precision and Recall for the final set of detections
        total_tp = global_tp.sum().item()
        total_fp = global_fp.sum().item()
        
        final_precision = total_tp / (total_tp + total_fp + 1e-16)
        final_recall = total_tp / (total_gts_c + 1e-16)
        
        metrics[f"AP_class_{c}"] = ap
        metrics[f"Precision_class_{c}"] = final_precision
        metrics[f"Recall_class_{c}"] = final_recall
        aps.append(ap)
        
    metrics["mAP"] = sum(aps) / len(aps) if aps else 0.0
    return metrics


def evaluate_model(model: torch.nn.Module, 
                   dataloader: torch.utils.data.DataLoader, 
                   anchors: list, 
                   num_classes: int,
                   conf_threshold: float = 0.05, 
                   nms_iou_threshold: float = 0.4, 
                   match_iou_threshold: float = 0.5,
                   device: torch.device = torch.device('cpu')) -> Dict[str, float]:
    """
    Orchestrates the full evaluation process.
    """
    all_preds, all_gts = collect_detections_and_gts(
        model, dataloader, anchors, num_classes, conf_threshold, nms_iou_threshold, device
    )
    return compute_mAP(all_preds, all_gts, num_classes, match_iou_threshold)
