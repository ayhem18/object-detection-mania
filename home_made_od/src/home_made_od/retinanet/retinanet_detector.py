"""
Build torchvision RetinaNet detectors from a finalized anchor spec and optional checkpoint.

The detector does **not** transform FPN levels or layers. Callers must pass an
:class:`~home_made_od.retinanet.retinanet_anchors.AnchorTrainingSpec` whose
``used_fpn_levels`` were already finalized by ``build_anchor_generator``.

Backbone wiring follows the layer model documented in
``home_made_od.retinanet.retinanet_levels`` and validated by
``tests/retinanet/test_retinanet_internal_layer_map.py``.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.retinanet import LastLevelP6P7, RetinaNetHead

from home_made_od.retinanet.retinanet_anchors import (
    AnchorTrainingSpec,
    RESNET_OUT_CHANNELS,
    build_anchor_generator,
    load_anchor_config,
)
from home_made_od.retinanet.retinanet_levels import (
    BODY_LAYERS,
    EXTRA_LAYERS,
    levels_to_layers,
    validate_anchor_aligned_layers,
)

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

FULLY_TRAINABLE_BACKBONE_LAYERS = 5
FPN_OUT_CHANNELS = 256
LAST_LEVEL_P6P7_IN_CHANNELS = RESNET_OUT_CHANNELS[4]


def _normalize_img_size(img_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(img_size, int):
        return (img_size, img_size)
    if len(img_size) != 2:
        raise ValueError(f"img_size must be (height, width), got {img_size!r}")
    return int(img_size[0]), int(img_size[1])


def _resolve_used_layers(anchor_spec: AnchorTrainingSpec) -> List[int]:
    """Read finalized layers from the anchor spec and reject invalid wiring."""
    config = load_anchor_config(anchor_spec.config_path)
    if "used_layers" in config:
        used_layers = [int(layer) for layer in config["used_layers"]]
    elif anchor_spec.used_fpn_levels is not None:
        used_layers = levels_to_layers(anchor_spec.used_fpn_levels)
    else:
        raise ValueError(
            "AnchorTrainingSpec is not finalized; call build_anchor_generator() "
            f"before building the detector (config={anchor_spec.config_path})."
        )

    validate_anchor_aligned_layers(used_layers)
    return used_layers


def _build_fpn_backbone_from_layers(
    used_layers: Sequence[int],
    trainable_backbone_layers: int,
) -> nn.Module:
    """
    ResNet50 + FPN wired exactly from unified layer indices 1..6.

    Body layers (1–4) are passed verbatim to ``returned_layers``.
    Layers 5–6 attach ``LastLevelP6P7``; they are not ResNet stages.
    """
    body_returned_layers = [layer for layer in used_layers if layer in BODY_LAYERS]
    uses_extra_block = any(layer in EXTRA_LAYERS for layer in used_layers)

    body = resnet50(weights=None)
    if uses_extra_block:
        extra_blocks = LastLevelP6P7(LAST_LEVEL_P6P7_IN_CHANNELS, FPN_OUT_CHANNELS)
        return _resnet_fpn_extractor(
            body,
            trainable_layers=trainable_backbone_layers,
            returned_layers=body_returned_layers,
            extra_blocks=extra_blocks,
        )

    return _resnet_fpn_extractor(
        body,
        trainable_layers=trainable_backbone_layers,
        returned_layers=body_returned_layers,
    )


def _assemble_retinanet(
    fpn_backbone: nn.Module,
    anchor_generator: AnchorGenerator,
    num_classes: int,
    img_size: Tuple[int, int],
    mean: Sequence[float],
    std: Sequence[float],
) -> RetinaNet:
    """
    Wire FPN backbone, anchor generator, and detection head.

    ``img_size`` is ``(height, width)``; torchvision ``fixed_size`` expects ``(width, height)``.
    ``_skip_resize=True`` assumes callers already resize inputs to ``img_size``.
    """
    head = RetinaNetHead(
        in_channels=fpn_backbone.out_channels,
        num_anchors=anchor_generator.num_anchors_per_location()[0],
        num_classes=num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    head.regression_head._loss_type = "giou"

    y_dim, x_dim = img_size
    return RetinaNet(
        fpn_backbone,
        num_classes,
        anchor_generator=anchor_generator,
        head=head,
        image_mean=list(mean),
        image_std=list(std),
        fixed_size=(x_dim, y_dim),
        _skip_resize=True,
    )


def _freeze_backbone_body(model: RetinaNet, freeze_layers: int) -> None:
    """Freeze the bottom ``freeze_layers`` ResNet body stages (1 = layer1, …)."""
    if freeze_layers <= 0:
        return
    for param in model.backbone.body.conv1.parameters():
        param.requires_grad = False
    for param in model.backbone.body.bn1.parameters():
        param.requires_grad = False
    for i in range(1, freeze_layers + 1):
        layer = getattr(model.backbone.body, f"layer{i}", None)
        if layer is not None:
            for param in layer.parameters():
                param.requires_grad = False


def build_retinanet(
    *,
    anchor_spec: AnchorTrainingSpec,
    num_classes: int,
    img_size: Union[int, Tuple[int, int]],
    device: torch.device,
    checkpoint_path: str | Path | None = None,
    freeze_backbone_layers: int = 0,
    trainable_backbone_layers: int = FULLY_TRAINABLE_BACKBONE_LAYERS,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    anchor_generator: AnchorGenerator | None = None,
) -> RetinaNet:
    """
    Build RetinaNet with anchors from a finalized ``anchor_spec``.

    Parameters
    ----------
    anchor_spec :
        Finalized ``anchor_config.json`` (``used_fpn_levels`` / ``used_layers``).
    num_classes :
        Foreground classes plus background (torchvision convention).
    img_size :
        Fixed input ``(height, width)`` for training/inference.
    checkpoint_path :
        If set, load a full detector ``state_dict`` (architecture must match anchors).
    freeze_backbone_layers :
        After build/load, freeze the bottom N ResNet stages in the backbone body.
    """
    image_mean = list(mean) if mean is not None else list(DEFAULT_IMAGE_MEAN)
    image_std = list(std) if std is not None else list(DEFAULT_IMAGE_STD)
    used_layers = _resolve_used_layers(anchor_spec)
    size_hw = _normalize_img_size(img_size)

    if anchor_generator is None:
        anchor_generator = build_anchor_generator(anchor_spec)
        logger.info("AnchorGenerator from %s", anchor_spec.config_path)
    else:
        logger.info("Using provided AnchorGenerator")

    fpn_backbone = _build_fpn_backbone_from_layers(used_layers, trainable_backbone_layers)
    model = _assemble_retinanet(
        fpn_backbone,
        anchor_generator,
        num_classes,
        size_hw,
        image_mean,
        image_std,
    )

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        logger.info("Loading detector checkpoint: %s", path)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    _freeze_backbone_body(model, freeze_backbone_layers)
    return model.to(device)
