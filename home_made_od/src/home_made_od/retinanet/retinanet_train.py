import os
import json
import math
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from home_made_od.retinanet.retinanet_anchors import (
        AnchorTrainingSpec,
    )

from home_made_od.general.early_stopping import EarlyStopping
from home_made_od.retinanet.retinanet_diagnose import (
    compute_image_diagnostics,
    aggregate_diagnostics
)
from mypt.loggers.tensorboard_logger import TensorBoardLogger

def _init_epoch_metrics() -> Dict[str, Any]:
    return {
        "total_loss": 0.0,
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "total_samples": 0
    }

def _update_epoch_metrics(epoch_metrics: Dict[str, Any], batch_metrics: Dict[str, Any]):
    epoch_metrics["total_loss"] += batch_metrics["loss"]
    epoch_metrics["loss_classifier"] += batch_metrics["loss_classifier"]
    epoch_metrics["loss_box_reg"] += batch_metrics["loss_box_reg"]
    epoch_metrics["total_samples"] += batch_metrics["batch_size"]

def _finalize_epoch_metrics(epoch_metrics: Dict[str, Any], num_batches: int) -> Dict[str, Any]:
    return {
        "epoch_loss": epoch_metrics["total_loss"] / num_batches if num_batches > 0 else 0.0,
        "loss_classifier": epoch_metrics["loss_classifier"] / num_batches if num_batches > 0 else 0.0,
        "loss_box_reg": epoch_metrics["loss_box_reg"] / num_batches if num_batches > 0 else 0.0,
    }

def _single_iteration(model: nn.Module, 
                      images: List[torch.Tensor], 
                      targets: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """
    RetinaNet in training mode expects List[Tensor] for images and List[Dict] for targets.
    It returns a dictionary of losses.
    """
    loss_dict = model(images, targets)
    
    losses = sum(loss for loss in loss_dict.values())
    
    if not math.isfinite(losses.item()):
        print(f"WARNING: Loss is {losses.item()}, stopping early or skipping update.")
        
    res = {
        "loss": losses.item(),
        "loss_tensor": losses,
        "loss_classifier": loss_dict["classification"].item(),
        "loss_box_reg": loss_dict["bbox_regression"].item(),
        "batch_size": len(images)
    }
    
    return res

def train_single_epoch(model: nn.Module, 
                       dataloader: torch.utils.data.DataLoader, 
                       optimizer: torch.optim.Optimizer, 
                       device: torch.device,
                       scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                       lr_update_level: Optional[str] = None,
                       max_norm: float = 5.0) -> Dict[str, Any]:
    model.train()
    epoch_metrics = _init_epoch_metrics()
    
    for images, targets in tqdm(dataloader, desc="Training epoch", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        batch_metrics = _single_iteration(model, images, targets)
        
        if math.isfinite(batch_metrics["loss"]):
            batch_metrics["loss_tensor"].backward()
            # Gradient clipping is heavily recommended for object detection
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        
        if scheduler is not None and lr_update_level == "batch":
            scheduler.step()
            
        _update_epoch_metrics(epoch_metrics, batch_metrics)
        
    return _finalize_epoch_metrics(epoch_metrics, len(dataloader))

def eval_single_epoch(model: nn.Module, 
                      dataloader: torch.utils.data.DataLoader, 
                      device: torch.device) -> Dict[str, Any]:
    """
    For PyTorch's RetinaNet, the model must be in .train() mode to compute validation loss,
    because .eval() mode will post-process outputs and return detections instead of losses.
    To prevent updating BatchNorm running statistics, we explicitly set them to .eval().
    """
    model.train() 
    
    # Freeze BatchNorm running stats
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm):
            module.eval()
            
    epoch_metrics = _init_epoch_metrics()
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation epoch", leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            batch_metrics = _single_iteration(model, images, targets)
            _update_epoch_metrics(epoch_metrics, batch_metrics)
            
    return _finalize_epoch_metrics(epoch_metrics, len(dataloader))

def save_diagnostic_visualization(
    image_tensor: torch.Tensor, 
    target: Dict[str, torch.Tensor], 
    pred: Dict[str, torch.Tensor], 
    label_map: Dict[int, str], 
    title: str, 
    save_path: str,
    score_thresh: float = 0.3
):
    """Saves a visualization of GT (green dashed) and Preds (red solid)."""
    # 1. Image preparation: [C, H, W] -> [H, W, C]
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(img_np)
    ax = plt.gca()
    
    # 2. Draw GT
    for box, lbl in zip(target['boxes'].cpu().numpy(), target['labels'].cpu().numpy()):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2, linestyle='--')
        ax.add_patch(rect)
        ax.text(x1, y1-5, f"GT: {label_map.get(int(lbl), lbl)}", color='lime', backgroundcolor='black', fontsize=8)
        
    # 3. Draw Preds
    p_boxes = pred['boxes'].cpu().numpy()
    p_scores = pred['scores'].cpu().numpy()
    p_labels = pred['labels'].cpu().numpy()
    
    keep = p_scores >= score_thresh
    for box, lbl, score in zip(p_boxes[keep], p_labels[keep], p_scores[keep]):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y2+15, f"P: {label_map.get(int(lbl), lbl)} ({score:.2f})", color='red', backgroundcolor='black', fontsize=8)
        
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def eval_diagnostics_epoch(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Runs the model in eval mode to extract actual bounding box predictions,
    computes diagnostics, and identifies edge-case images for visualization.
    """
    model.eval()
    all_results = []
    score_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Tracking for edge cases
    # Each entry: (value, image_tensor, target_dict, pred_dict, title_prefix)
    worst_consensus = (float('inf'), None, None, None, "Lowest Consensus")
    worst_avg_max_iou = (float('inf'), None, None, None, "Lowest Avg Max IoU")
    
    # per-class worst IoU: {cls: (min_iou, image, target, pred, title)}
    worst_per_class_iou = {}

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Diagnostics epoch", leave=False):
            images_device = [img.to(device) for img in images]
            targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images_device)
            
            for i in range(len(images)):
                res = compute_image_diagnostics(outputs[i], targets_device[i], score_thresholds=score_thresholds)
                if not res: continue
                
                all_results.append(res)
                
                img_cpu = images[i].cpu()
                target_cpu = {k: v.cpu() for k, v in targets[i].items()}
                pred_cpu = {k: v.cpu() for k, v in outputs[i].items()}

                # Global edge cases
                if res['avg_consensus'] < worst_consensus[0]:
                    worst_consensus = (res['avg_consensus'], img_cpu, target_cpu, pred_cpu, f"Lowest Consensus ({res['avg_consensus']:.3f})")
                
                if res['avg_max_iou'] < worst_avg_max_iou[0]:
                    worst_avg_max_iou = (res['avg_max_iou'], img_cpu, target_cpu, pred_cpu, f"Lowest Avg Max IoU ({res['avg_max_iou']:.3f})")

                # Per-class edge cases
                for cls, iou in res.get('per_class_avg_max_iou', {}).items():
                    if cls not in worst_per_class_iou or iou < worst_per_class_iou[cls][0]:
                        worst_per_class_iou[cls] = (iou, img_cpu, target_cpu, pred_cpu, f"Worst {cls} IoU ({iou:.3f})")
                        
    aggregated = aggregate_diagnostics(all_results, score_thresholds)
    
    edge_cases = {
        "global_worst_consensus": worst_consensus,
        "global_worst_iou": worst_avg_max_iou,
        "per_class_worst_iou": worst_per_class_iou
    }
    
    return aggregated, edge_cases


def run_training_loop(model: nn.Module, 
                      train_loader: torch.utils.data.DataLoader, 
                      val_loader: torch.utils.data.DataLoader, 
                      optimizer: torch.optim.Optimizer, 
                      device: torch.device,
                      epochs: int,
                      artifact_dir: str,
                      cls_id_2_cls_name: Dict[int, str],
                      early_stop_patience: int = 5,
                      scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                      lr_update_level: Optional[str] = None):
    
    os.makedirs(artifact_dir, exist_ok=True)
    checkpoint_dir = os.path.join(artifact_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics_dir = os.path.join(artifact_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    viz_dir = Path(artifact_dir) / "viz"
    viz_dir.mkdir(exist_ok=True)
    
    # Initialize TensorBoard Logger
    tb_log_dir = os.path.join(artifact_dir, "tb_logs")
    logger = TensorBoardLogger(log_dir=tb_log_dir)
    
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    early_stopping = EarlyStopping(path=best_model_path, max_iterations=early_stop_patience)
    early_stopping.initialize()

    for epoch in range(epochs):
        train_metrics = train_single_epoch(model, train_loader, optimizer, device, scheduler, lr_update_level)
        val_metrics = eval_single_epoch(model, val_loader, device)
        diag_metrics, edge_cases = eval_diagnostics_epoch(model, val_loader, device)
        
        if scheduler is not None and lr_update_level == "epoch":
            scheduler.step()

        # Logging to TensorBoard
        logger.log_scalar("Loss/Train_Total", train_metrics['epoch_loss'], epoch)
        logger.log_scalar("Loss/Train_Cls", train_metrics['loss_classifier'], epoch)
        logger.log_scalar("Loss/Train_Reg", train_metrics['loss_box_reg'], epoch)
        
        logger.log_scalar("Loss/Val_Total", val_metrics['epoch_loss'], epoch)
        logger.log_scalar("Loss/Val_Cls", val_metrics['loss_classifier'], epoch)
        logger.log_scalar("Loss/Val_Reg", val_metrics['loss_box_reg'], epoch)
        
        if diag_metrics['consensus'] is not None:
            logger.log_scalar("Diagnostics/Label_Consensus", diag_metrics['consensus'], epoch)
            
        if len(diag_metrics['max_ious']) > 0:
            logger.log_histogram("Diagnostics/Max_IoU", np.array(diag_metrics['max_ious']), epoch)
            
        for thresh, pr in diag_metrics['pr_metrics'].items():
            logger.log_scalar(f"Precision/Thresh_{thresh}", pr['precision'], epoch)
            logger.log_scalar(f"Recall/Thresh_{thresh}", pr['recall'], epoch)

        # Per-class logging to TensorBoard
        for cls, metrics in diag_metrics.get('per_class', {}).items():
            cls_name = cls_id_2_cls_name.get(cls, f"Class_{cls}")
            logger.log_scalar(f"PerClass_Max_IoU/{cls_name}", metrics['mean_max_iou'], epoch)
            logger.log_scalar(f"PerClass_Consensus/{cls_name}", metrics['mean_consensus'], epoch)
            
            for thresh, pr in metrics['pr_metrics'].items():
                if thresh in [0.3, 0.5, 0.7]:
                    logger.log_scalar(f"PerClass_Precision_{thresh}/{cls_name}", pr['precision'], epoch)
                    logger.log_scalar(f"PerClass_Recall_{thresh}/{cls_name}", pr['recall'], epoch)

        # ---------------- Visualization of Edge Cases ----------------
        epoch_viz_dir = viz_dir / f"epoch_{epoch:03d}"
        epoch_viz_dir.mkdir(exist_ok=True)
        
        # Save global edge cases
        for key in ["global_worst_consensus", "global_worst_iou"]:
            val, img, target, pred, title = edge_cases[key]
            if img is not None:
                save_path = epoch_viz_dir / f"{key}.png"
                save_diagnostic_visualization(img, target, pred, cls_id_2_cls_name, title, str(save_path))
                
        # Save per-class edge cases
        for cls, (val, img, target, pred, title) in edge_cases["per_class_worst_iou"].items():
            cls_name = cls_id_2_cls_name.get(cls, f"Class_{cls}")
            save_path = epoch_viz_dir / f"worst_iou_{cls_name}.png"
            save_diagnostic_visualization(img, target, pred, cls_id_2_cls_name, title, str(save_path))
        # -------------------------------------------------------------

        # Simple logging to JSON
        metrics_file = os.path.join(metrics_dir, "metrics.json")
        epoch_results = {
            "epoch": epoch, 
            "train": train_metrics, 
            "val": val_metrics,
            "diagnostics": {
                "consensus": diag_metrics['consensus'],
                "pr_metrics": diag_metrics['pr_metrics'],
                "per_class": diag_metrics.get('per_class', {})
            }
        }
        with open(metrics_file, "a") as f:
            f.write(json.dumps(epoch_results) + "\n")

        # Console Output
        log_items = [
            f"Epoch {epoch+1:03d}/{epochs:03d}",
            f"T-Cls: {train_metrics['loss_classifier']:.4f}",
            f"T-Reg: {train_metrics['loss_box_reg']:.4f}",
            f"V-Cls: {val_metrics['loss_classifier']:.4f}",
            f"V-Reg: {val_metrics['loss_box_reg']:.4f}"
        ]
        
        if diag_metrics['consensus'] is not None:
            log_items.append(f"Cons: {diag_metrics['consensus']:.4f}")
        
        p_at_50 = diag_metrics['pr_metrics'][0.5]['precision']
        r_at_50 = diag_metrics['pr_metrics'][0.5]['recall']
        log_items.append(f"R@0.5: {r_at_50:.4f}")
        log_items.append(f"P@0.5: {p_at_50:.4f}")
        
        print(" | ".join(log_items))
        
        if early_stopping.check_early_stop(model, val_metrics['epoch_loss']):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    logger.close()

def train_retinanet_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
    artifact_dir: str,
    device: torch.device,
    cls_id_2_cls_name: Dict[int, str],
    patience: int = 15,
    anchor_spec: Optional["AnchorTrainingSpec"] = None,
):
    if anchor_spec is not None:
        from home_made_od.retinanet.retinanet_anchors import (
            save_anchor_spec_to_artifact_dir,
        )

        save_anchor_spec_to_artifact_dir(anchor_spec, Path(artifact_dir))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # OneCycleLR is often good for detection
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs
    )

    run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        artifact_dir=artifact_dir,
        early_stop_patience=patience,
        scheduler=scheduler,
        lr_update_level="batch",
        cls_id_2_cls_name=cls_id_2_cls_name
    )
