import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any

from mypt.loggers.base import BaseLogger
from home_made_od.general.early_stopping import EarlyStopping
from home_made_od.yolo_v2.modules.yolov2_loss import YoloV2Loss
from home_made_od.yolo_v2.modules.target_calculation import YoloV2TargetCalculator

# =========================================================================================
# YOLOv2 Training Utilities
# =========================================================================================

def _init_epoch_metrics() -> Dict[str, Any]:
    return {
        "total_loss": 0.0,
        "loss_obj": 0.0,
        "loss_reg": 0.0,
        "loss_cls": 0.0,
        "num_objects": 0.0,
        "total_samples": 0,
        "top_k_samples": [] # List of (loss, img, pred, target)
    }

def _update_top_k(current_list: List[Tuple], new_items: List[Tuple], k: int) -> List[Tuple]:
    current_list.extend(new_items)
    # Sort by loss descending
    current_list.sort(key=lambda x: x[0], reverse=True)
    return current_list[:k]

def _update_epoch_metrics(epoch_metrics: Dict[str, Any], batch_metrics: Dict[str, Any], k_samples: int):
    epoch_metrics["total_loss"] += batch_metrics["total_loss"]
    epoch_metrics["loss_obj"] += batch_metrics["loss_obj"]
    epoch_metrics["loss_reg"] += batch_metrics["loss_reg"]
    epoch_metrics["loss_cls"] += batch_metrics["loss_cls"]
    epoch_metrics["num_objects"] += batch_metrics["num_objects"]
    epoch_metrics["total_samples"] += batch_metrics["batch_size"]
    
    if "top_samples" in batch_metrics:
        epoch_metrics["top_k_samples"] = _update_top_k(epoch_metrics["top_k_samples"], batch_metrics["top_samples"], k_samples)

def _finalize_epoch_metrics(epoch_metrics: Dict[str, Any], num_batches: int) -> Dict[str, Any]:
    if num_batches == 0:
        return {k: 0.0 for k in epoch_metrics if k != "top_k_samples"}
    
    return {
        "epoch_loss": epoch_metrics["total_loss"] / num_batches,
        "loss_obj": epoch_metrics["loss_obj"] / num_batches,
        "loss_reg": epoch_metrics["loss_reg"] / num_batches,
        "loss_cls": epoch_metrics["loss_cls"] / num_batches,
        "avg_objects_per_batch": epoch_metrics["num_objects"] / num_batches,
        "total_samples": epoch_metrics["total_samples"],
        "top_samples": epoch_metrics["top_k_samples"]
    }

def _single_iteration(model: nn.Module, 
                      inputs: torch.Tensor, 
                      raw_targets: torch.Tensor, 
                      target_calculator: YoloV2TargetCalculator,
                      criterion: YoloV2Loss, 
                      device: torch.device,
                      track_top_samples: bool = True) -> Dict[str, Any]:
    """
    Performs a single forward pass and loss calculation.
    """
    batch_size = inputs.size(0)
    
    # 1. Forward pass
    preds = model(inputs)
    
    # 2. Prepare target tensor for loss
    # target_calculator expects normalized targets (batch_idx, cls_id, x, y, w, h)
    with torch.no_grad():
        prepared_targets = target_calculator(raw_targets, batch_size)
    
    # 3. Compute loss
    # Get reduced loss for backward
    loss_dict = criterion(preds, prepared_targets, reduce=True, return_all_losses=True)
    
    res = {
        "loss_tensor": loss_dict["total_loss"],
        "total_loss": loss_dict["total_loss"].item(),
        "loss_obj": loss_dict["loss_obj"].item(),
        "loss_reg": loss_dict["loss_reg"].item(),
        "loss_cls": loss_dict["loss_cls"].item(),
        "num_objects": loss_dict["num_objects"].item(),
        "batch_size": batch_size
    }
    
    # 4. Diagnostics: Track top-loss samples
    if track_top_samples:
        with torch.no_grad():
            # Get unreduced losses per image
            unreduced = criterion(preds, prepared_targets, reduce=False)
            total_unreduced = unreduced["loss_obj"] + unreduced["loss_reg"] + unreduced["loss_cls"]
            
            curr_imgs = inputs.detach().cpu()
            # We don't store the full pred tensor to save memory, 
            # but we could store a slice or post-processed detections if needed.
            # For now, just loss and image reference.
            res["top_samples"] = [
                (total_unreduced[i].item(), curr_imgs[i], None, None) 
                for i in range(batch_size)
            ]
            
    return res

def train_single_epoch(model: nn.Module, 
                       dataloader: torch.utils.data.DataLoader, 
                       optimizer: torch.optim.Optimizer, 
                       target_calculator: YoloV2TargetCalculator,
                       criterion: YoloV2Loss,
                       device: torch.device,
                       scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                       lr_update_level: Optional[str] = None,
                       k_samples: int = 5) -> Dict[str, Any]:
    
    model.train()
    epoch_metrics = _init_epoch_metrics()
    
    pbar = tqdm(dataloader, desc="Training epoch", leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        batch_metrics = _single_iteration(model, inputs, labels, target_calculator, criterion, device)
        
        batch_metrics["loss_tensor"].backward()
        optimizer.step()
        
        if scheduler is not None and lr_update_level == "batch":
            scheduler.step()
            
        _update_epoch_metrics(epoch_metrics, batch_metrics, k_samples)
        pbar.set_postfix({"loss": f"{batch_metrics['total_loss']:.4f}"})
        
    return _finalize_epoch_metrics(epoch_metrics, len(dataloader))

def eval_single_epoch(model: nn.Module, 
                      dataloader: torch.utils.data.DataLoader, 
                      target_calculator: YoloV2TargetCalculator,
                      criterion: YoloV2Loss,
                      device: torch.device,
                      k_samples: int = 5) -> Dict[str, Any]:
    
    model.eval()
    epoch_metrics = _init_epoch_metrics()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation epoch", leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            batch_metrics = _single_iteration(model, inputs, labels, target_calculator, criterion, device)
            _update_epoch_metrics(epoch_metrics, batch_metrics, k_samples)
            pbar.set_postfix({"loss": f"{batch_metrics['total_loss']:.4f}"})
            
    return _finalize_epoch_metrics(epoch_metrics, len(dataloader))

# =========================================================================================
# Artifact and Directory Management
# =========================================================================================

def get_artifact_paths(artifact_dir: str):
    return {
        "root": artifact_dir,
        "checkpoints_dir": os.path.join(artifact_dir, "checkpoints"),
        "best_model": os.path.join(artifact_dir, "checkpoints", "best_model.pt"),
        "interrupted_model": os.path.join(artifact_dir, "checkpoints", "interrupted_model.pt"),
        "metrics_dir": os.path.join(artifact_dir, "metrics"),
        "metrics_file": os.path.join(artifact_dir, "metrics", "metrics.json"),
        "visualizations_dir": os.path.join(artifact_dir, "visualizations"),
        "logger_dir": os.path.join(artifact_dir, "logger_data")
    }

def _log_metrics_to_json(train_metrics: Dict[str, Any], val_metrics: Dict[str, Any], epoch: int, artifact_dir: str):
    paths = get_artifact_paths(artifact_dir)
    file_path = paths["metrics_file"]
    
    history = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                history = json.load(f)
            except: pass

    for k, v in train_metrics.items():
        if isinstance(v, (float, int)):
            history.setdefault(f"train_{k}", []).append(v)
    for k, v in val_metrics.items():
        if isinstance(v, (float, int)):
            history.setdefault(f"val_{k}", []).append(v)
            
    with open(file_path, "w") as f:
        json.dump(history, f, indent=4)

# =========================================================================================
# Main Training Logic
# =========================================================================================

def run_training_loop(model: nn.Module, 
                      train_loader: torch.utils.data.DataLoader, 
                      val_loader: torch.utils.data.DataLoader, 
                      optimizer: torch.optim.Optimizer, 
                      target_calculator: YoloV2TargetCalculator,
                      criterion: YoloV2Loss,
                      device: torch.device,
                      epochs: int,
                      artifact_dir: str,
                      early_stop_patience: int = 10,
                      logger: Optional[BaseLogger] = None,
                      scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                      lr_update_level: Optional[str] = None,
                      k_samples_vis: int = 5):
    
    print(f"\n--- Starting YOLOv2 Training Loop (Device: {device}) ---")
    
    paths = get_artifact_paths(artifact_dir)
    for p in paths.values():
        if not p.endswith(".pt") and not p.endswith(".json"):
            os.makedirs(p, exist_ok=True)

    early_stopping = EarlyStopping(path=paths["best_model"], max_iterations=early_stop_patience)
    early_stopping.initialize()

    try:
        for epoch in range(epochs):
            train_res = train_single_epoch(
                model, train_loader, optimizer, target_calculator, criterion, 
                device, scheduler, lr_update_level, k_samples_vis
            )
            
            val_res = eval_single_epoch(
                model, val_loader, target_calculator, criterion, device, k_samples_vis
            )
            
            if scheduler is not None and lr_update_level == "epoch":
                scheduler.step()

            # Logging
            _log_metrics_to_json(train_res, val_res, epoch, artifact_dir)
            if logger:
                log_dict = {f"train_{k}": v for k, v in train_res.items() if isinstance(v, (float, int))}
                log_dict.update({f"val_{k}": v for k, v in val_res.items() if isinstance(v, (float, int))})
                logger.log_dict(log_dict, epoch)

            # Print
            print(f"\nEpoch [{epoch+1:03d}/{epochs:03d}]")
            print(f"  Train | Loss: {train_res['epoch_loss']:.5f} (Obj: {train_res['loss_obj']:.5f}, Reg: {train_res['loss_reg']:.5f}, Cls: {train_res['loss_cls']:.5f})")
            print(f"  Val   | Loss: {val_res['epoch_loss']:.5f} (Obj: {val_res['loss_obj']:.5f}, Reg: {val_res['loss_reg']:.5f}, Cls: {val_res['loss_cls']:.5f})")

            if early_stopping.check_early_stop(model, val_res['epoch_loss']):
                print("Early stopping triggered.")
                break
                
    except KeyboardInterrupt:
        print("\n[!] Training interrupted. Saving current state...")
        torch.save(model.state_dict(), paths["interrupted_model"])
    finally:
        if logger:
            logger.close()
