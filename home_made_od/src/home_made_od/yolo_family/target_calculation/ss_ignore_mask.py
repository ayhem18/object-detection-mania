import torch
from typing import List, Tuple


class SingScaleIgnoreTargetCalculator:
    def __init__(self, 
                    num_classes: int, 
                    anchors: List[Tuple[float, float]], 
                    feature_map_shape: Tuple[int, int],
                    iou_threshold_ignore_anchor: float
                    ):
        """This class is used to calculate the targets for the YOLOv2 model.

        Args:
            num_classes (int): The number of classes in the dataset.
            anchors (List[Tuple[float, float]]): The anchors for the model, as [w, h] in range [0, 1].
            feature_map_shape (Tuple[int, int]): The shape of the feature map (H, W).
            iou_threshold_ignore_anchor (float): The IoU threshold for which an anchor is ignored so it does not contribute to the non-objectness loss.
        """
        self.num_classes = num_classes
        self.feature_map_shape = feature_map_shape
        self.num_anchors = len(anchors)
        self.iou_threshold_ignore_anchor = iou_threshold_ignore_anchor
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

    def compute_targets(self, targets: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the target tensor for the YOLOv2 model. How exactly ?

        This function expects the targets to be in the following shape: [B, 6] -> [image_index, class_index, x_center, y_center, width, height]
        with x_center and y_center in the range [0, 1] and width and height in the range [0, 1].

        The output of this function is a target tensor of the shape [B, H, W, num_anchors * (5 + num_classes)] and an ignore mask of the shape [B, H, W, num_anchors]     

        # H: height of the feature map
        # W: width of the feature map
        # 5: tx, ty, th, tw, obj 
        # tx: represents the relative distance between the center of the anchor box and the feature map top left x-coordinate
        # ty: represents the relative distance between the center of the anchor box and the feature map top left y-coordinate
        # th: log(height / anchor_height)
        # tw: log(width / anchor_width)
        # obj: 1 if the anchor box contains an object, 0 otherwise
        

        """
        H, W = self.feature_map_shape
        device = targets.device
        
        # Initialize target tensor: (B, num_anchors, 5 + C, H, W)
        target_tensor = torch.zeros((batch_size, self.num_anchors, self.num_channels_per_anchor, H, W), device=device)
        
        if targets.shape[0] == 0:
            return target_tensor.view(batch_size, -1, H, W)

        # 1. Target Indices
        batch_idx = targets[:, 0].long()
        cls_id = targets[:, 1].long()

        # 2. Compute tx and ty        
        grid_x = targets[:, 2] * W # [0, 1] -> [0, W]
        grid_y = targets[:, 3] * H # [0, 1] -> [0, H]
        
        grid_x_floor = torch.floor(grid_x).long().clamp(0, W - 1)
        grid_y_floor = torch.floor(grid_y).long().clamp(0, H - 1)

        tx = grid_x - grid_x_floor
        ty = grid_y - grid_y_floor

        # 3. Compute tw and th
        # 3.a Compute the best anchor for the ground truth box
        gt_wh = targets[:, 4:6]
        anchors = self.anchors.to(device)
        
        # compute the IoU between each ground truth box and each anchor if they have the same center
        # why is this enough: because for each cell we have anchors that all have the same center (the center of the cell)
        # and a gt bbox will be compared against anchors whose centers share the same grid cell
        # so gt  bbox -> cell -> candidate anchors
        # best anchor for this bbox -> anchor with the closest aspect ratio (since the center of the bbox is almost the same as the center of the cell which the same as the center of the anchor )
        ious = self._wh_iou(gt_wh, anchors)
        best_anchor_indices = torch.argmax(ious, dim=1)

        
        matched_anchors = anchors[best_anchor_indices]
        th = torch.log(targets[:, 4] / (matched_anchors[:, 0] + 1e-7) + 1e-7)
        tw = torch.log(targets[:, 5] / (matched_anchors[:, 1] + 1e-7) + 1e-7)
        
        # Vectorized Population
        target_tensor[batch_idx, best_anchor_indices, 0, grid_y_floor, grid_x_floor] = tx
        target_tensor[batch_idx, best_anchor_indices, 1, grid_y_floor, grid_x_floor] = ty
        target_tensor[batch_idx, best_anchor_indices, 2, grid_y_floor, grid_x_floor] = th
        target_tensor[batch_idx, best_anchor_indices, 3, grid_y_floor, grid_x_floor] = tw
        target_tensor[batch_idx, best_anchor_indices, 4, grid_y_floor, grid_x_floor] = 1.0 # Objectness
        
        # Vectorized Classification Target
        target_tensor[batch_idx, best_anchor_indices, 5 + cls_id, grid_y_floor, grid_x_floor] = 1.0
        target_tensor_final = target_tensor.view(batch_size, -1, H, W)

        # the next step is to implement the ignore mask
        # anchors are ignored if they are not the best anchor for any ground truth bbox
        # but have a iou with a ground truth bbox that is greater than the iou_threshold_ignore_anchor

        true_best_anchor_indices = best_anchor_indices.unsqueeze(1) == torch.arange(self.num_anchors, device=device).unsqueeze(0)
        gt_anchors_ignore = ious > self.iou_threshold_ignore_anchor & ~ true_best_anchor_indices
        
        # gt_anchors_ignore is a (GT * num_anchors) matrix 
        # we need to convert this matrix into a (batch_size, H, W, num_anchors)
        # the non-vectorized solution would be to first create a (batch_size, H, W, num_anchors) tensor of zeros
        ignore_mask = torch.zeros((batch_size, H, W, self.num_anchors), device=device)

        # for each gt find its h and w coordinates and then set the ignore_mask[batch_index, h, w, :] to the gt_anchors_ignore[gt_index, :]
        # the assignment below is quite tricky: 
        # batch_idx: [T,] (where T is the number of ground truth boxes)
        # grid_y_floor: [T,]
        # grid_x_floor: [T,]
        ignore_mask[batch_idx, grid_y_floor, grid_x_floor, :] = gt_anchors_ignore
        return target_tensor_final, ignore_mask

    def __call__(self, targets: torch.Tensor, batch_size: int) -> torch.Tensor:
        return self.compute_targets(targets, batch_size)