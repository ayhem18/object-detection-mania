import os
import json
import torch
from typing import Tuple, Dict, Optional

# torchvision related imports
from torch import nn
from torchvision.ops import nms
from torchvision.models import resnet50
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor

# New model class IDs
CLASS_ID_BACKGROUND = 0
CLASS_ID_WIRED = 1
CLASS_ID_GROOVE = 2
CLASS_ID_DUPLEX = 3

# Legacy mask values expected by DPMM postprocessing
LEGACY_MASK_VALUE_DUPLEX = 1.0
LEGACY_MASK_VALUE_GROOVE = 2.0

# New model configuration
NUM_CLASSES = 4

DEFAULT_IMG_RESOLUTION = 1244

def build_segmentation_anchor_generator(
    manifest_path: str,
    default_aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)) -> AnchorGenerator:
    """
    Builds a torchvision AnchorGenerator from the optimized manifest.
    Ensures that 5 levels are always provided to match maskrcnn_resnet50_fpn_v2.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Anchor configuration not found at {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Handle both full manifest structure and direct metadata structure
    metadata = data.get("metadata", data)
    level_ratios_map = metadata.get("level_aspect_ratios", {})
    scales = metadata.get("scales", [1.0, 2**(1/3), 2**(2/3)])
    
    standard_levels = ["P2", "P3", "P4", "P5", "P6"]
    fpn_specs = {s['level']: s for s in metadata.get("fpn_specs", [])}
    
    if level_ratios_map:
        target_num_ratios = max(len(ratios) for ratios in level_ratios_map.values())
    else:
        target_num_ratios = len(default_aspect_ratios)

    anchor_sizes = []
    aspect_ratios = []
    
    for level in standard_levels:
        spec = fpn_specs.get(level)
        if not spec:
            strides = {"P2": 4, "P3": 8, "P4": 16, "P5": 32, "P6": 64}
            base_size = strides.get(level, 8) * metadata.get("coefficient", 4)
        else:
            base_size = spec['base_size']
            
        level_sizes = tuple(base_size * s for s in scales)
        anchor_sizes.append(level_sizes)
        
        ratios = level_ratios_map.get(level)
        if ratios is None:
            if target_num_ratios == 3:
                ratios = list(default_aspect_ratios)
            elif target_num_ratios == 5:
                ratios = [0.2, 0.5, 1.0, 2.0, 5.0]
            else:
                ratios = list(default_aspect_ratios)
                while len(ratios) < target_num_ratios:
                    ratios.append(ratios[-1])
                ratios = ratios[:target_num_ratios]
        
        aspect_ratios.append(tuple(ratios))
        
    return AnchorGenerator(tuple(anchor_sizes), tuple(aspect_ratios))

def load_backbone_weights(backbone_base: nn.Module, weights_path: str):
    """Loads backbone weights from a checkpoint, handling various formats."""
    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError(f"Backbone weights not found at {weights_path}")

    print(f"Loading backbone weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    if hasattr(checkpoint, 'backbone'):
        target_state_dict = checkpoint.backbone.body.state_dict() if hasattr(checkpoint.backbone, 'body') else checkpoint.backbone.state_dict()
        backbone_base.load_state_dict(target_state_dict, strict=False)
    elif isinstance(checkpoint, dict):
        if 'backbone.body.conv1.weight' in checkpoint:
            new_state_dict = {k.replace('backbone.body.', ''): v for k, v in checkpoint.items() if k.startswith('backbone.body.')}
            backbone_base.load_state_dict(new_state_dict, strict=False)
        elif 'body.conv1.weight' in checkpoint:
                new_state_dict = {k.replace('body.', ''): v for k, v in checkpoint.items() if k.startswith('body.')}
                backbone_base.load_state_dict(new_state_dict, strict=False)
        else:
            backbone_base.load_state_dict(checkpoint, strict=False)
    else:
        backbone_base.load_state_dict(checkpoint, strict=False)
    del checkpoint

def build_welding_mask_rcnn(
    num_classes: int,
    manifest_path: str,
    weights_path: str = None,
    trainable_backbone_layers: int = 0,
    mean: list = [0.10013955, 0.10013955, 0.10013955],
    std: list = [0.11381042, 0.11381042, 0.11381042],
    device: torch.device = torch.device('cpu')):
    """Builds a Mask R-CNN V2 model for inference."""
    backbone_base = resnet50(weights=None)
    
    load_backbone_weights(backbone_base, weights_path)

    backbone = _resnet_fpn_extractor(
        backbone_base, 
        trainable_layers=trainable_backbone_layers,
        norm_layer=nn.BatchNorm2d
    )

    rpn_anchor_generator = build_segmentation_anchor_generator(manifest_path)

    rpn_head = RPNHead(
        backbone.out_channels, 
        rpn_anchor_generator.num_anchors_per_location()[0],
        conv_depth=2
    )
    
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), 
        [256, 256, 256, 256], [1024], 
        norm_layer=nn.BatchNorm2d
    )
    
    mask_head = MaskRCNNHeads(
        backbone.out_channels, 
        [256, 256, 256, 256], 1, 
        norm_layer=nn.BatchNorm2d
    )

    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_head=box_head,
        mask_head=mask_head,
        min_size=DEFAULT_IMG_RESOLUTION,  
        max_size=DEFAULT_IMG_RESOLUTION,
        image_mean=mean,
        image_std=std
    )

    return model.to(device)

def filter_predictions(
    outputs: Dict[str, torch.Tensor], 
    conf_threshold: float = 0.5, 
    nms_threshold: Optional[float] = None) -> Dict[str, torch.Tensor]:
    """Converts raw model outputs into filtered predictions."""
    scores = outputs["scores"]
    mask = scores >= conf_threshold
    
    filtered = {
        "boxes": outputs["boxes"][mask],
        "labels": outputs["labels"][mask],
        "scores": outputs["scores"][mask],
        "masks": outputs["masks"][mask]
    }
    
    if filtered["masks"].dtype != torch.bool and filtered["masks"].dtype != torch.uint8:
        filtered["masks"] = filtered["masks"] > 0.5
    
    if filtered["masks"].ndim == 4:
        filtered["masks"] = filtered["masks"].squeeze(1)
    
    if nms_threshold is not None and filtered["boxes"].shape[0] > 0:
        max_coordinate = filtered["boxes"].max()
        offsets = filtered["labels"].to(filtered["boxes"]) * (max_coordinate + 1)
        boxes_for_nms = filtered["boxes"] + offsets[:, None]
        keep = nms(boxes_for_nms.float(), filtered["scores"], nms_threshold)

        for k in filtered:
            filtered[k] = filtered[k][keep]
            
    return filtered

 