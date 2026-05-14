import torch
import torch.nn as nn
from typing import List, Tuple
from ...components.conv_head import ConvHead

class YoloV2(nn.Module):
    def __init__(self, 
                 backbone: nn.Module, 
                 backbone_out_channels: int, 
                 num_anchors: int,
                 num_classes: int, 
                 num_conv_blocks: int = 2):
        """
        YOLOv2 Model Architecture.
        
        Args:
            backbone (nn.Module): The feature extractor (e.g., Darknet, ResNet).
            backbone_out_channels (int): Number of output channels from the backbone.
            num_anchors (int): Number of anchors per grid cell.
            num_classes (int): Number of classes to detect.
            num_conv_blocks (int): Number of 3x3 conv blocks in the detection head.
        """
        super().__init__()
        self.backbone = backbone
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # 1. Detection Head (Feature extraction)
        # Standard hidden channels for YOLOv2 is 1024
        self.head = ConvHead(backbone_out_channels, num_conv_blocks, hidden_channels=1024)
        
        # 2. Prediction Layer (1x1 Conv)
        # Output channels: num_anchors * (5 + num_classes)
        # 5: tx, ty, th, tw, obj
        self.out_channels = num_anchors * (5 + num_classes)
        self.predictor = nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input image batch (B, 3, H, W)
        Returns:
            torch.Tensor: Raw predictions (B, num_anchors * (5 + num_classes), H_feat, W_feat)
        """
        # Feature extraction from backbone
        features = self.backbone(x)
        
        # Apply detection head
        head_features = self.head(features)
        
        # Final prediction
        out = self.predictor(head_features)
        
        return out

    def decode_predictions(self, 
                           raw_output: torch.Tensor, 
                           anchors: List[List[float] | Tuple[float, float]], 
                           img_size: Tuple[int, int],
                           return_all_probs: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decodes raw model output into absolute bounding boxes without confidence filtering or NMS.
        Args:
            raw_output: Model output (B, K*(5+C), fh, fw)
            anchors: List of [w, h] in range [0, 1].
            img_size: Tuple of (img_h, img_w) to scale predictions to absolute pixels.
        Returns:
            Tuple of Tensors:
                - boxes: (B, fh*fw*num_anchors, 4) -> [x1, y1, x2, y2]
                - scores: (B, fh*fw*num_anchors) -> obj_prob * max_cls_prob
                - cls_ids: (B, fh*fw*num_anchors) -> predicted class index
        """
        B, _, fh, fw = raw_output.shape
        img_h, img_w = img_size
        stride_w = img_w / fw
        stride_h = img_h / fh

        anchors_tensor = torch.tensor(anchors, dtype=torch.float32, device=raw_output.device)
        if anchors_tensor.shape != (self.num_anchors, 2):
            raise ValueError(f"Expected anchors shape ({self.num_anchors}, 2), got {anchors_tensor.shape}")

        out = raw_output.view(B, self.num_anchors, 5 + self.num_classes, fh, fw)
        out = out.permute(0, 3, 4, 1, 2).contiguous()

        tx = out[..., 0]
        ty = out[..., 1]
        tw = out[..., 2]
        th = out[..., 3]
        obj_logits = out[..., 4]
        cls_logits = out[..., 5:]

        cx = torch.sigmoid(tx)
        cy = torch.sigmoid(ty)
        obj_probs = torch.sigmoid(obj_logits)
        cls_probs = torch.softmax(cls_logits, dim=-1)

        max_cls_probs, cls_ids = torch.max(cls_probs, dim=-1)
        scores = obj_probs * max_cls_probs

        grid_y, grid_x = torch.meshgrid(torch.arange(fh, device=raw_output.device), 
                                        torch.arange(fw, device=raw_output.device), 
                                        indexing='ij')
        grid_x = grid_x.unsqueeze(-1).expand(-1, -1, self.num_anchors)
        grid_y = grid_y.unsqueeze(-1).expand(-1, -1, self.num_anchors)

        anchors_w = anchors_tensor[:, 0].view(1, 1, self.num_anchors)
        anchors_h = anchors_tensor[:, 1].view(1, 1, self.num_anchors)

        actual_cx = (cx + grid_x) * stride_w
        actual_cy = (cy + grid_y) * stride_h
        actual_w = torch.exp(tw) * anchors_w * img_w
        actual_h = torch.exp(th) * anchors_h * img_h

        x1 = actual_cx - actual_w / 2
        y1 = actual_cy - actual_h / 2
        x2 = actual_cx + actual_w / 2
        y2 = actual_cy + actual_h / 2

        boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(B, -1, 4)
        scores = scores.view(B, -1)
        cls_ids = cls_ids.view(B, -1)

        if return_all_probs:
            obj_probs = obj_probs.view(B, -1)
            max_cls_probs = max_cls_probs.view(B, -1)
            return boxes, scores, cls_ids, obj_probs, max_cls_probs

        return boxes, scores, cls_ids

    def inference(self, 
                  x: torch.Tensor,
                  anchors: List[List[float] | Tuple[float, float]],
                  conf_threshold: float = 0.5, 
                  nms_iou_threshold: float = 0.4) -> List[torch.Tensor]:
        """
        Performs full inference pass: Forward -> Decode -> NMS.
        Args:
            x: Input image batch (B, 3, H, W)
            anchors: List of [w, h] in range [0, 1].
            conf_threshold: Minimum confidence score (obj * class) to keep a box.
            nms_iou_threshold: IoU threshold for Non-Maximum Suppression.
        Returns:
            List of Tensors, one per image. Shape (N, 6) -> [x1, y1, x2, y2, score, class_id]
        """
        self.eval()
        with torch.no_grad():
            raw_output = self.forward(x)

        img_h, img_w = x.shape[2], x.shape[3]
        boxes, scores, cls_ids = self.decode_predictions(raw_output, anchors, (img_h, img_w))
        B = raw_output.shape[0]

        batch_results = []
        import torchvision.ops as ops

        for i in range(B):
            img_boxes = boxes[i]
            img_scores = scores[i]
            img_cls = cls_ids[i]

            # Confidence Filter
            mask = img_scores > conf_threshold
            if not mask.any():
                batch_results.append(torch.zeros((0, 6), device=x.device))
                continue

            b = img_boxes[mask]
            s = img_scores[mask]
            c = img_cls[mask]

            # Non-Maximum Suppression
            # Using class-aware NMS trick (offsetting boxes by class ID * large constant)
            max_wh = 4096 # larger than image size
            offsets = c.float() * max_wh
            keep = ops.nms(b + offsets.unsqueeze(1), s, nms_iou_threshold)

            batch_results.append(torch.cat([
                b[keep], 
                s[keep].unsqueeze(1), 
                c[keep].unsqueeze(1).float()
            ], dim=1))

        return batch_results