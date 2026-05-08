import torch
from typing import List, Tuple

class YoloV2TargetCalculator:
    def __init__(self, num_classes: int, anchors: List[Tuple[float, float]], feature_map_shape: Tuple[int, int]):
        """This class is used to calculate the targets for the YOLOv2 model.

        Args:
            num_classes (int): The number of classes in the dataset.
            anchors (List[Tuple[float, float]]): The anchors for the model, as [w, h] in range [0, 1].
            feature_map_shape (Tuple[int, int]): The shape of the feature map (H, W).
        """
        self.num_classes = num_classes
        self.feature_map_shape = feature_map_shape
        self.num_anchors = len(anchors)
        
        # Convert anchors to tensor (num_anchors, 2) -> [w, h]
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        
        if self.anchors.ndim != 2 or self.anchors.shape[1] != 2:
            raise ValueError(f"Each anchor is expected to be a tuple of length 2 [w, h]. Found shape: {self.anchors.shape}")

        if (self.anchors < 0).any() or (self.anchors > 1).any():
            raise ValueError("All anchor dimensions must be in the range [0, 1].")

        if not isinstance(num_classes, int) or num_classes <= 1:
            raise ValueError(f"The number of classes is expected to be a positive integer greater than 1. Found: {num_classes}")
    
        self.num_channels_per_anchor = 5 + self.num_classes
        self.num_channels = self.num_anchors * self.num_channels_per_anchor

    def _wh_iou(self, wh1: torch.Tensor, wh2: torch.Tensor) -> torch.Tensor:
        """Computes IoU between two sets of widths and heights, assuming they are center-aligned."""
        inter = torch.min(wh1[:, None, 0], wh2[None, :, 0]) * \
                torch.min(wh1[:, None, 1], wh2[None, :, 1])
        
        area1 = wh1[:, 0] * wh1[:, 1]
        area2 = wh2[:, 0] * wh2[:, 1]
        
        union = area1[:, None] + area2[None, :] - inter
        return inter / (union + 1e-7)

    def compute_targets(self, targets: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Calculates the target tensor for the YOLOv2 model."""
        H, W = self.feature_map_shape
        device = targets.device
        
        # Initialize target tensor: (B, num_anchors, 5 + C, H, W)
        target_tensor = torch.zeros((batch_size, self.num_anchors, self.num_channels_per_anchor, H, W), device=device)
        
        if targets.shape[0] == 0:
            return target_tensor.view(batch_size, -1, H, W)

        # 1. Target Indices
        batch_idx = targets[:, 0].long()
        cls_id = targets[:, 1].long()
        
        grid_x = targets[:, 2] * W
        grid_y = targets[:, 3] * H
        
        j = torch.floor(grid_x).long().clamp(0, W - 1)
        i = torch.floor(grid_y).long().clamp(0, H - 1)
        
        # 2. Anchor Matching
        gt_wh = targets[:, 4:6]
        anchors = self.anchors.to(device)
        
        ious = self._wh_iou(gt_wh, anchors)
        best_anchor_indices = torch.argmax(ious, dim=1)

        # 3. Fill Target Tensor
        tx = grid_x - j
        ty = grid_y - i
        
        matched_anchors = anchors[best_anchor_indices]
        tw = torch.log(targets[:, 4] / (matched_anchors[:, 0] + 1e-7) + 1e-7)
        th = torch.log(targets[:, 5] / (matched_anchors[:, 1] + 1e-7) + 1e-7)
        
        # Vectorized Population
        target_tensor[batch_idx, best_anchor_indices, 0, i, j] = tx
        target_tensor[batch_idx, best_anchor_indices, 1, i, j] = ty
        target_tensor[batch_idx, best_anchor_indices, 2, i, j] = tw
        target_tensor[batch_idx, best_anchor_indices, 3, i, j] = th
        target_tensor[batch_idx, best_anchor_indices, 4, i, j] = 1.0 # Objectness
        
        # Vectorized Classification Target
        target_tensor[batch_idx, best_anchor_indices, 5 + cls_id, i, j] = 1.0

        return target_tensor.view(batch_size, -1, H, W)

    def __call__(self, targets: torch.Tensor, batch_size: int) -> torch.Tensor:
        return self.compute_targets(targets, batch_size)
