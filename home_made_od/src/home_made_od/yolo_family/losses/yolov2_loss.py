import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Tuple, Optional


##
## Legacy implementation of the yolov2Loss
##

class YoloV2LossLegacy(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 num_anchors: int, 
                 reg_loss_type: str = "mse", 
                 background_obj_coeff: float = 0.5):
        """
        YOLOv2 Loss implementation.
        
        Args:
            num_classes (int): Number of classes.
            num_anchors (int): Number of anchors per cell.
            reg_loss_type (str): Type of regression loss ('mse' or 'giou').
            background_obj_coeff (float): Multiplier for objectness loss in cells without objects.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.reg_loss_type = reg_loss_type.lower()
        self.background_obj_coeff = background_obj_coeff
        
        if self.reg_loss_type not in ["mse", "giou"]:
            raise ValueError(f"reg_loss_type must be 'mse' or 'giou'. Got: {reg_loss_type}")

    def _reshape_and_permute(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes (B, K*(5+C), H, W) to (B, K, H, W, 5+C)"""
        B, _, H, W = x.shape
        K = self.num_anchors
        C = self.num_classes
        return x.view(B, K, 5 + C, H, W).permute(0, 1, 3, 4, 2).contiguous()

    def _compute_unreduced_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes loss per image in the batch.
        preds/targets expected in shape (B, K, H, W, 5+C)
        """
        B = preds.shape[0]
        target_obj = targets[..., 4]
        obj_mask = target_obj > 0
        
        weights = torch.ones_like(target_obj)
        weights[~obj_mask] = self.background_obj_coeff
        
        # 1. Objectness Loss (BCE per sample)
        # Using binary_cross_entropy_with_logits with reduction='none'
        loss_obj_elements = F.binary_cross_entropy_with_logits(preds[..., 4], target_obj, weight=weights, reduction='none')
        loss_obj_per_image = loss_obj_elements.view(B, -1).sum(dim=1)
        
        # 2. Regression and Classification (Loop per image to handle varying object counts)
        loss_reg_per_image = torch.zeros(B, device=preds.device)
        loss_cls_per_image = torch.zeros(B, device=preds.device)
        num_objs_per_image = obj_mask.view(B, -1).sum(dim=1)
        
        for b in range(B):
            m = obj_mask[b]
            if m.any():
                # Regression
                p_xy = torch.sigmoid(preds[b][m][:, 0:2])
                t_xy = targets[b][m][:, 0:2]
                p_wh = preds[b][m][:, 2:4]
                t_wh = targets[b][m][:, 2:4]
                
                loss_reg_per_image[b] = F.mse_loss(p_xy, t_xy, reduction='sum') + F.mse_loss(p_wh, t_wh, reduction='sum')
                
                # Classification
                p_cls = preds[b][m][:, 5:]
                t_cls = targets[b][m][:, 5:]
                t_idx = torch.argmax(t_cls, dim=1)
                loss_cls_per_image[b] = F.cross_entropy(p_cls, t_idx, reduction='sum')

        return {
            "loss_obj": loss_obj_per_image,
            "loss_reg": loss_reg_per_image,
            "loss_cls": loss_cls_per_image,
            "num_objects": num_objs_per_image
        }

    def forward(self, 
                preds: torch.Tensor, 
                targets: torch.Tensor, 
                reduce: bool = True,
                return_all_losses: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        B = preds.shape[0]
        preds = self._reshape_and_permute(preds)
        targets = self._reshape_and_permute(targets)
        
        if not reduce:
            return self._compute_unreduced_loss(preds, targets)
        
        # Efficient Reduced Path
        target_obj = targets[..., 4]
        obj_mask = target_obj > 0
        total_objects = obj_mask.sum().item()
        reg_cls_denom = max(1.0, total_objects)
        
        # 1. Objectness Loss (BCE with Logits)
        weights = torch.ones_like(target_obj)
        weights[~obj_mask] = self.background_obj_coeff
        loss_obj = F.binary_cross_entropy_with_logits(preds[..., 4], target_obj, weight=weights, reduction='sum') / B
        
        # 2. Regression and Classification
        loss_reg = torch.tensor(0.0, device=preds.device)
        loss_cls = torch.tensor(0.0, device=preds.device)
        
        if total_objects > 0:
            # Extract only cells with objects
            p_obj_cells = preds[obj_mask]
            t_obj_cells = targets[obj_mask]
            
            # Regression: xy (sigmoid) + wh (raw log-space)
            loss_xy = F.mse_loss(torch.sigmoid(p_obj_cells[:, 0:2]), t_obj_cells[:, 0:2], reduction='sum')
            loss_wh = F.mse_loss(p_obj_cells[:, 2:4], t_obj_cells[:, 2:4], reduction='sum')
            loss_reg = (loss_xy + loss_wh) / reg_cls_denom
            
            # Classification
            t_class_idx = torch.argmax(t_obj_cells[:, 5:], dim=1)
            loss_cls = F.cross_entropy(p_obj_cells[:, 5:], t_class_idx, reduction='sum') / reg_cls_denom
            
        total_loss = loss_obj + loss_reg + loss_cls
        
        if return_all_losses:
            return {
                "total_loss": total_loss,
                "loss_obj": loss_obj,
                "loss_reg": loss_reg,
                "loss_cls": loss_cls,
                "num_objects": torch.tensor(total_objects, device=preds.device)
            }
            
        return total_loss


class YoloV2Loss(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 num_anchors: int, 
                 reg_loss_type: str = "mse", 
                 background_obj_coeff: float = 0.5,
                 lambda_coord: float = 5.0,
                 lambda_obj: float = 1.0,
                 lambda_cls: float = 1.0):
        """
        YOLOv2 Loss implementation.
        
        Args:
            num_classes (int): Number of classes.
            num_anchors (int): Number of anchors per cell.
            reg_loss_type (str): Type of regression loss ('mse' or 'giou').
            background_obj_coeff (float): Multiplier for objectness loss in cells without objects (lambda_noobj).
            lambda_coord (float): Multiplier for coordinate regression loss.
            lambda_obj (float): Multiplier for objectness loss in cells WITH objects.
            lambda_cls (float): Multiplier for classification loss.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.reg_loss_type = reg_loss_type.lower()
        self.background_obj_coeff = background_obj_coeff
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        
        if self.reg_loss_type not in ["mse", "giou"]:
            raise ValueError(f"reg_loss_type must be 'mse' or 'giou'. Got: {reg_loss_type}")

    def _reshape_and_permute(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes (B, K*(5+C), H, W) to (B, K, H, W, 5+C)"""
        B, _, H, W = x.shape
        K = self.num_anchors
        C = self.num_classes
        return x.view(B, K, 5 + C, H, W).permute(0, 1, 3, 4, 2).contiguous()

    def _compute_unreduced_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes loss per image in the batch.
        preds/targets expected in shape (B, K, H, W, 5+C)
        """
        B = preds.shape[0]
        target_obj = targets[..., 4]
        obj_mask = target_obj > 0
        
        weights = torch.ones_like(target_obj) * self.lambda_obj
        weights[~obj_mask] = self.background_obj_coeff
        
        # 1. Objectness Loss (BCE per sample)
        loss_obj_elements = F.binary_cross_entropy_with_logits(preds[..., 4], target_obj, weight=weights, reduction='none')
        loss_obj_per_image = loss_obj_elements.view(B, -1).sum(dim=1)
        
        # 2. Regression and Classification
        loss_reg_per_image = torch.zeros(B, device=preds.device)
        loss_cls_per_image = torch.zeros(B, device=preds.device)
        num_objs_per_image = obj_mask.view(B, -1).sum(dim=1)
        
        for b in range(B):
            m = obj_mask[b]
            if m.any():
                # Regression
                p_xy = torch.sigmoid(preds[b][m][:, 0:2])
                t_xy = targets[b][m][:, 0:2]
                p_wh = preds[b][m][:, 2:4]
                t_wh = targets[b][m][:, 2:4]
                
                reg_loss = F.mse_loss(p_xy, t_xy, reduction='sum') + F.mse_loss(p_wh, t_wh, reduction='sum')
                loss_reg_per_image[b] = reg_loss * self.lambda_coord
                
                # Classification
                p_cls = preds[b][m][:, 5:]
                t_cls = targets[b][m][:, 5:]
                t_idx = torch.argmax(t_cls, dim=1)
                loss_cls_per_image[b] = F.cross_entropy(p_cls, t_idx, reduction='sum') * self.lambda_cls

        return {
            "loss_obj": loss_obj_per_image,
            "loss_reg": loss_reg_per_image,
            "loss_cls": loss_cls_per_image,
            "num_objects": num_objs_per_image
        }

    def forward(self, 
                preds: torch.Tensor, 
                targets: torch.Tensor, 
                reduce: bool = True,
                return_all_losses: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        B = preds.shape[0]
        preds = self._reshape_and_permute(preds)
        targets = self._reshape_and_permute(targets)
        
        if not reduce:
            return self._compute_unreduced_loss(preds, targets)
        
        # Efficient Reduced Path
        target_obj = targets[..., 4]
        obj_mask = target_obj > 0
        total_objects = obj_mask.sum().item()
        
        # 1. Objectness Loss (BCE with Logits)
        weights = torch.ones_like(target_obj) * self.lambda_obj
        weights[~obj_mask] = self.background_obj_coeff
        # Sum over entire batch, then divide by B
        loss_obj = F.binary_cross_entropy_with_logits(preds[..., 4], target_obj, weight=weights, reduction='sum') / B
        
        # 2. Regression and Classification
        loss_reg = torch.tensor(0.0, device=preds.device)
        loss_cls = torch.tensor(0.0, device=preds.device)
        
        if total_objects > 0:
            p_obj_cells = preds[obj_mask]
            t_obj_cells = targets[obj_mask]
            
            # Regression: xy (sigmoid) + wh (raw log-space)
            loss_xy = F.mse_loss(torch.sigmoid(p_obj_cells[:, 0:2]), t_obj_cells[:, 0:2], reduction='sum')
            loss_wh = F.mse_loss(p_obj_cells[:, 2:4], t_obj_cells[:, 2:4], reduction='sum')
            
            # Sum over all objects, divide by B, apply lambda
            loss_reg = ((loss_xy + loss_wh) / B) * self.lambda_coord
            
            # Classification
            t_class_idx = torch.argmax(t_obj_cells[:, 5:], dim=1)
            # Sum over all objects, divide by B, apply lambda
            loss_cls = (F.cross_entropy(p_obj_cells[:, 5:], t_class_idx, reduction='sum') / B) * self.lambda_cls
            
        total_loss = loss_obj + loss_reg + loss_cls
        
        if return_all_losses:
            return {
                "total_loss": total_loss,
                "loss_obj": loss_obj,
                "loss_reg": loss_reg,
                "loss_cls": loss_cls,
                "num_objects": torch.tensor(total_objects, device=preds.device)
            }
            
        return total_loss
