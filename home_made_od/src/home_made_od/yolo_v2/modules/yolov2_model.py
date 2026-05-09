import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class ConvBlock(nn.Module):
    """
    A standard building block: Conv3x3 -> Norm -> Activation
    Includes an optional residual connection.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn(self.conv(x)))
        if self.use_residual:
            out = out + identity
        return out

class ConvHead(nn.Module):
    """
    The detection head consisting of multiple convolutional blocks.
    """
    def __init__(self, in_channels: int, num_blocks: int, hidden_channels: int = 1024):
        super().__init__()
        blocks = []
        # First block to transition to hidden_channels
        blocks.append(ConvBlock(in_channels, hidden_channels, use_residual=False))
        
        # Subsequent blocks with residual connections
        for _ in range(num_blocks - 1):
            blocks.append(ConvBlock(hidden_channels, hidden_channels, use_residual=True))
            
        self.feature_extractor = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

class YoloV2(nn.Module):
    def __init__(self, 
                 backbone: nn.Module, 
                 backbone_out_channels: int, 
                 num_anchors: int, # to be deprecated
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
        # 5: tx, ty, tw, th, obj
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

        B, _, fh, fw = raw_output.shape
        img_h, img_w = x.shape[2], x.shape[3]
        stride_w = img_w / fw
        stride_h = img_h / fh

        # Convert anchors to tensor and move to device
        anchors_tensor = torch.tensor(anchors, dtype=torch.float32, device=x.device)
        if anchors_tensor.shape != (self.num_anchors, 2):
            raise ValueError(f"Expected anchors shape ({self.num_anchors}, 2), got {anchors_tensor.shape}")

        # 1. Reshape to (B, fh, fw, num_anchors, 5 + num_classes)
        # Input: (B, K*(5+C), H, W)
        out = raw_output.view(B, self.num_anchors, 5 + self.num_classes, fh, fw)
        out = out.permute(0, 3, 4, 1, 2).contiguous()

        # 2. Unpack
        tx = out[..., 0]
        ty = out[..., 1]
        tw = out[..., 2]
        th = out[..., 3]
        obj_logits = out[..., 4]
        cls_logits = out[..., 5:]

        # 3. Activations
        cx = torch.sigmoid(tx)
        cy = torch.sigmoid(ty)
        obj_probs = torch.sigmoid(obj_logits)
        cls_probs = torch.softmax(cls_logits, dim=-1)

        # 4. Overall Scores
        max_cls_probs, cls_ids = torch.max(cls_probs, dim=-1)
        scores = obj_probs * max_cls_probs

        # 5. Decode BBoxes
        # Create grid offsets
        grid_y, grid_x = torch.meshgrid(torch.arange(fh, device=x.device), 
                                        torch.arange(fw, device=x.device), 
                                        indexing='ij')
        grid_x = grid_x.unsqueeze(-1).expand(-1, -1, self.num_anchors)
        grid_y = grid_y.unsqueeze(-1).expand(-1, -1, self.num_anchors)

        # Map anchors to correct shape
        anchors_w = anchors_tensor[:, 0].view(1, 1, self.num_anchors)
        anchors_h = anchors_tensor[:, 1].view(1, 1, self.num_anchors)

        # Decode to absolute pixels
        actual_cx = (cx + grid_x) * stride_w
        actual_cy = (cy + grid_y) * stride_h
        actual_w = torch.exp(tw) * anchors_w * img_w
        actual_h = torch.exp(th) * anchors_h * img_h

        # Convert to x1y1x2y2
        x1 = actual_cx - actual_w / 2
        y1 = actual_cy - actual_h / 2
        x2 = actual_cx + actual_w / 2
        y2 = actual_cy + actual_h / 2

        # 6. Flatten and NMS
        batch_results = []
        import torchvision.ops as ops

        for i in range(B):
            # Flatten image-specific predictions
            img_x1 = x1[i].view(-1)
            img_y1 = y1[i].view(-1)
            img_x2 = x2[i].view(-1)
            img_y2 = y2[i].view(-1)
            img_scores = scores[i].view(-1)
            img_cls = cls_ids[i].view(-1)

            # Confidence Filter
            mask = img_scores > conf_threshold
            if not mask.any():
                batch_results.append(torch.zeros((0, 6), device=x.device))
                continue

            boxes = torch.stack([img_x1[mask], img_y1[mask], img_x2[mask], img_y2[mask]], dim=1)
            s = img_scores[mask]
            c = img_cls[mask]

            # Non-Maximum Suppression
            # Using class-aware NMS trick (offsetting boxes by class ID * large constant)
            # This ensures NMS only suppresses boxes of the same class.
            max_wh = 4096 # larger than image size
            offsets = c.float() * max_wh
            keep = ops.nms(boxes + offsets.unsqueeze(1), s, nms_iou_threshold)

            batch_results.append(torch.cat([
                boxes[keep], 
                s[keep].unsqueeze(1), 
                c[keep].unsqueeze(1).float()
            ], dim=1))

        return batch_results

