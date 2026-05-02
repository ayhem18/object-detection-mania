import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Tuple, Optional

class YoloV2Loss(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 num_anchors: int, 
                 reg_loss_type: str = "mse", 
                 background_obj_coeff: float = 0.5):
        """
        YOLOv2 Loss implementation with per-sample diagnostic support.
        
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

    def _prepare_data(self, preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshapes and applies activations to raw predictions.
        
        Returns:
            Tuple: (preds_permuted, targets_permuted, obj_mask, weights)
                   permuted shapes: (B, K, H, W, 5+C)
                   obj_mask shape: (B, K, H, W)
                   weights shape: (B, K, H, W)
        """
        B, _, H, W = preds.shape
        K = self.num_anchors
        C = self.num_classes
        
        # Reshape and Permute to (B, K, H, W, 5+C)
        preds = preds.view(B, K, 5 + C, H, W).permute(0, 1, 3, 4, 2).contiguous()
        targets = targets.view(B, K, 5 + C, H, W).permute(0, 1, 3, 4, 2).contiguous()
        
        # Apply activations using concatenation for efficiency
        # [tx, ty] -> sigmoid, [tw, th] -> raw, [obj] -> sigmoid, [classes] -> raw
        activated_preds = torch.cat([
            torch.sigmoid(preds[..., 0:2]), # xy
            preds[..., 2:4],               # wh
            torch.sigmoid(preds[..., 4:5]), # obj
            preds[..., 5:]                  # classes
        ], dim=-1)
        
        target_obj = targets[..., 4]
        obj_mask = target_obj > 0
        
        # Calculate background weights
        weights = torch.ones_like(target_obj)
        weights[~obj_mask] = self.background_obj_coeff
        
        return activated_preds, targets, obj_mask, weights

    def _compute_unreduced_loss(self, preds: torch.Tensor, targets: torch.Tensor, obj_mask: torch.Tensor, weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes loss per image in the batch.
        Returns dict of tensors of shape (B,)
        """
        B = preds.shape[0]
        
        # 1. Objectness Loss (BCE per element, then sum over K,H,W)
        # BCE = -[y*log(p) + (1-y)*log(1-p)]
        p_obj = preds[..., 4]
        t_obj = targets[..., 4]
        # Manual BCE to keep per-element control
        loss_obj_elements = -(t_obj * torch.log(p_obj + 1e-7) + (1 - t_obj) * torch.log(1 - p_obj + 1e-7))
        loss_obj_elements = loss_obj_elements * weights
        loss_obj_per_image = loss_obj_elements.view(B, -1).sum(dim=1)
        
        # 2. Regression and Classification
        loss_reg_per_image = torch.zeros(B, device=preds.device)
        loss_cls_per_image = torch.zeros(B, device=preds.device)
        num_objs_per_image = obj_mask.view(B, -1).sum(dim=1)
        
        for b in range(B):
            m = obj_mask[b]
            if m.any():
                # Regression
                p_xy = preds[b][m][:, 0:2]
                t_xy = targets[b][m][:, 0:2]
                p_wh = preds[b][m][:, 2:4]
                t_wh = targets[b][m][:, 2:4]
                
                l_xy = F.mse_loss(p_xy, t_xy, reduction='sum')
                l_wh = F.mse_loss(p_wh, t_wh, reduction='sum')
                loss_reg_per_image[b] = l_xy + l_wh
                
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
        p, t, mask, w = self._prepare_data(preds, targets)
        
        unreduced = self._compute_unreduced_loss(p, t, mask, w)
        
        if not reduce:
            return unreduced
        
        # Reduction logic:
        # loss_obj reduced by batch_size
        # loss_reg/loss_cls reduced by total number of objects
        total_objects = unreduced["num_objects"].sum()
        reg_cls_denom = max(1.0, total_objects.item())
        
        red_obj = unreduced["loss_obj"].sum() / B
        red_reg = unreduced["loss_reg"].sum() / reg_cls_denom
        red_cls = unreduced["loss_cls"].sum() / reg_cls_denom
        
        total_loss = red_obj + red_reg + red_cls
        
        if return_all_losses:
            return {
                "total_loss": total_loss,
                "loss_obj": red_obj,
                "loss_reg": red_reg,
                "loss_cls": red_cls,
                "num_objects": total_objects
            }
            
        return total_loss
