"""
Build torchvision RetinaNet detectors from an anchor config and optional checkpoint.

FPN / backbone gotchas (torchvision + anchor manifest coupling)
-----------------------------------------------------------------
The anchor manifest ``used_fpn_levels`` drives which FPN levels the detector
uses, but several torchvision constraints force implicit backbone changes.
See :func:`dl_lib.etalon_object_detection.modules.retinanet.retinanet_anchors.retinanet_returned_layers`
and :func:`_build_fpn_backbone` for the mapping logic; regression coverage lives in ``tests/retinanet/test_retinanet_building.py``.

**Level naming and strides**

- P2 → stride 4, P3 → 8, P4 → 16, P5 → 32, P6 → 64, P7 → 128 (relative to
  the input image, assuming square tensors).
- P2–P5 map to ResNet50 ``layer1``–``layer4`` (indices 1–4). P6/P7 are not
  ResNet stages; they are produced by ``LastLevelP6P7`` on top of the P5
  backbone feature (2048 channels, stride 32).

**P3 implies P2**

If P3 is requested, ``layer2`` must be extracted from the backbone. Torchvision
FPN cannot build a valid pyramid from ``layer2`` alone — ``layer1`` (P2) is
always added to ``returned_layers`` even when P2 is absent from the manifest.
The backbone therefore emits a P2 feature map that may not have matching anchors.

**P6 and P7 are inseparable**

``validate_retinanet_fpn_levels`` requires P6 and P7 together. Either both
appear in ``used_fpn_levels`` or neither does.

**P6/P7 imply P5 in the backbone (even if P5 is not in the manifest)**

``LastLevelP6P7`` branches from the raw ResNet ``layer4`` output (stride 32,
2048 channels). If P6 or P7 is requested but P5 is not, ``layer4`` is still
added to ``returned_layers``. The P5 FPN map is computed but may be discarded
downstream when feature maps are aligned to ``used_fpn_levels``.

**``LastLevelP6P7`` input channels**

Always pass ``in_channels=2048`` (ResNet ``layer4``), never the channel width
of the highest *requested* level. Using e.g. 256 when only P2 + P6 + P7 are
requested makes ``LastLevelP6P7`` treat the finest FPN output as its source,
placing P6/P7 at strides 8/16 instead of 64/128.

**Extra ``pool`` level when P6/P7 are off**

When ``LastLevelP6P7`` is not attached, torchvision defaults to
``LastLevelMaxPool``, appending one extra feature map (key ``"pool"``) at
2× the stride of the finest returned level. The backbone can therefore return
*more* maps than ``len(used_fpn_levels)``; downstream code truncates to the
manifest length (see ``anchors_matching_sanity_check.py``).

**Non-contiguous ResNet layers**

``returned_layers`` must form a contiguous range ``[min, …, max]``. Gaps (e.g.
P2 + P4 without P3) break FPN lateral connections and cause channel mismatches
at runtime. Anchor optimization should not produce such manifests.

**Empty / P6-only manifest edge case**

If ``used_fpn_levels`` contains no P2–P5 entries (e.g. only ``["P6", "P7"]``),
``returned_layers`` falls back to ``[4]`` (P5 backbone only) before P6/P7 are
attached.
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
    build_anchor_generator_from_spec,
    retinanet_returned_layers,
    validate_retinanet_fpn_levels,
)

logger = logging.getLogger(__name__)

# ImageNet normalization (torchvision default for detection models).
DEFAULT_IMAGE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

FULLY_TRAINABLE_BACKBONE_LAYERS = 5

_OUT_CHANNELS = RESNET_OUT_CHANNELS
_returned_layers = retinanet_returned_layers


def _normalize_img_size(img_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(img_size, int):
        return (img_size, img_size)
    if len(img_size) != 2:
        raise ValueError(f"img_size must be (height, width), got {img_size!r}")
    return int(img_size[0]), int(img_size[1])


def _build_fpn_backbone(
    used_levels: List[str],
    trainable_backbone_layers: int,
) -> nn.Module:
    """ResNet50 + FPN for the FPN levels listed in the anchor config."""
    body = resnet50(weights=None)
    returned_layers = _returned_layers(used_levels)
    needs_p6p7 = "P6" in used_levels or "P7" in used_levels

    if needs_p6p7:
        validate_retinanet_fpn_levels(used_levels)
        extra_blocks = LastLevelP6P7(_OUT_CHANNELS[4], 256)
        return _resnet_fpn_extractor(
            body,
            trainable_layers=trainable_backbone_layers,
            returned_layers=returned_layers,
            extra_blocks=extra_blocks,
        )

    return _resnet_fpn_extractor(
        body,
        trainable_layers=trainable_backbone_layers,
        returned_layers=returned_layers,
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
    Build RetinaNet with anchors from ``anchor_spec`` and optional weights.

    Parameters
    ----------
    anchor_spec :
        Resolved ``anchor_config.json`` (FPN levels + per-level sizes/ratios).
    num_classes :
        Foreground classes plus background (torchvision convention).
    img_size :
        Fixed input ``(height, width)`` for training/inference (independent of anchor clustering).
    checkpoint_path :
        If set, load a full detector ``state_dict`` (architecture must match anchors).
    freeze_backbone_layers :
        After build/load, freeze the bottom N ResNet stages in the backbone body.
    """
    image_mean = list(mean) if mean is not None else list(DEFAULT_IMAGE_MEAN)
    image_std = list(std) if std is not None else list(DEFAULT_IMAGE_STD)
    used_levels = anchor_spec.used_fpn_levels
    size_hw = _normalize_img_size(img_size)

    if anchor_generator is None:
        anchor_generator = build_anchor_generator_from_spec(anchor_spec)
        logger.info("AnchorGenerator from %s", anchor_spec.config_path)
    else:
        logger.info("Using provided AnchorGenerator")

    fpn_backbone = _build_fpn_backbone(used_levels, trainable_backbone_layers)
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
