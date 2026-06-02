"""
Build dl-lab Etalon RetinaNet detectors (shared layout, two weight sources).

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
from typing import List, Literal, Tuple, Union

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
# from dl_lib.etalon_object_detection.scripts.labeling.segmentation_model_predictions.seg_model import load_backbone_weights

logger = logging.getLogger(__name__)

WeightSource = Literal["segmentation", "trained_detector"]

DL_LIB_ETALON_NUM_CLASSES = 2
DL_LIB_ETALON_MEAN_VECTOR = [0.10013955, 0.10013955, 0.10013955]
DL_LIB_ETALON_STD_VECTOR = [0.11381042, 0.11381042, 0.11381042]
# All ResNet body blocks start trainable; use ``freeze_layers`` to restrict from the bottom.
FULLY_TRAINABLE_BACKBONE_LAYERS = 5

_OUT_CHANNELS = RESNET_OUT_CHANNELS
_returned_layers = retinanet_returned_layers


def _build_fpn_backbone(
    used_levels: List[str],
    trainable_backbone_layers: int,
) -> tuple[nn.Module, nn.Module]:
    """
    Build ResNet50 body and FPN backbone for the requested manifest levels.

    Parameters
    ----------
    used_levels :
        FPN level names from the anchor manifest.
    trainable_backbone_layers :
        Count of unfrozen ResNet stages (0–5); see ``FULLY_TRAINABLE_BACKBONE_LAYERS``.

    Returns
    -------
    tuple[nn.Module, nn.Module]
        ``(body, fpn_backbone)`` where ``body`` is the raw ResNet50 used for
        segmentation weight loading and ``fpn_backbone`` is the torchvision
        ``BackboneWithFPN`` wrapper.

    Notes
    -----
    **P6/P7 path.** When either level is requested, ``validate_retinanet_fpn_levels``
    runs first, then ``LastLevelP6P7(_OUT_CHANNELS[4], 256)`` is attached.
    ``_OUT_CHANNELS[4]`` is always 2048 (ResNet ``layer4``); do not derive this
    from the last entry of :func:`_returned_layers` — see module docstring.

    **No P6/P7 path.** ``extra_blocks`` is omitted, so torchvision inserts
    ``LastLevelMaxPool`` and appends a ``"pool"`` map at 2× the finest stride.
    Expect ``len(backbone(...)) > len(used_fpn_levels)`` in that case.

    **Trainability vs freezing.** ``trainable_backbone_layers`` controls which
    ResNet stages receive gradients at build time; :func:`_freeze_backbone_body`
    may further freeze stages after weight loading.
    """
    body = resnet50(weights=None)

    returned_layers = _returned_layers(used_levels)
    needs_p6p7 = "P6" in used_levels or "P7" in used_levels

    if needs_p6p7:
        validate_retinanet_fpn_levels(used_levels)
        extra_blocks = LastLevelP6P7(_OUT_CHANNELS[4], 256)
        fpn = _resnet_fpn_extractor(
            body,
            trainable_layers=trainable_backbone_layers,
            returned_layers=returned_layers,
            extra_blocks=extra_blocks,
        )
        return body, fpn

    fpn = _resnet_fpn_extractor(
        body,
        trainable_layers=trainable_backbone_layers,
        returned_layers=returned_layers,
    )
    return body, fpn


def _assemble_retinanet(
    fpn_backbone: nn.Module,
    anchor_generator: AnchorGenerator,
    num_classes: int,
    img_size: Union[int, Tuple[int, int]],
    mean: List[float],
    std: List[float],
) -> RetinaNet:
    """
    Wire FPN backbone, anchor generator, and detection head into a RetinaNet.

    Notes
    -----
    **Head.** Uses ``GroupNorm(32)`` and GIoU box regression
    (``head.regression_head._loss_type = "giou"``).

    **Image size convention.** ``img_size`` is ``(height, width)`` everywhere
    else in dl-lab (tensors shaped ``[C, H, W]``). Torchvision ``fixed_size``
    expects ``(width, height)``, so the tuple is swapped here. ``_skip_resize=True``
    assumes callers already feed images at the target resolution.
    """
    head = RetinaNetHead(
        in_channels=fpn_backbone.out_channels,
        num_anchors=anchor_generator.num_anchors_per_location()[0],
        num_classes=num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    head.regression_head._loss_type = "giou"

    flipped_transform_fixed_size = (img_size[1], img_size[0])
    return RetinaNet(
        fpn_backbone,
        num_classes,
        anchor_generator=anchor_generator,
        head=head,
        image_mean=mean,
        image_std=std,
        fixed_size=flipped_transform_fixed_size,
        _skip_resize=True,
    )


def _freeze_backbone_body(model: RetinaNet, freeze_layers: int) -> None:
    """
    Freeze the bottom ``freeze_layers`` ResNet body stages after weight loading.

    ``freeze_layers=2`` freezes ``conv1``, ``bn1``, ``layer1``, and ``layer2``.
    This is independent of ``trainable_backbone_layers`` passed at FPN build time.
    """
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


def build_dl_lab_etalon_retinanet(
    *,
    anchor_spec: AnchorTrainingSpec,
    img_size: Union[int, Tuple[int, int]],
    device: torch.device,
    weights_path: str,
    weight_source: WeightSource,
    num_classes: int = DL_LIB_ETALON_NUM_CLASSES,
    freeze_layers: int = 2,
    mean: List[float] | None = None,
    std: List[float] | None = None,
    anchor_generator: AnchorGenerator | None = None,
) -> RetinaNet:
    """
    Build Etalon RetinaNet with optimized anchors.

    Parameters
    ----------
    anchor_spec :
        Resolved anchor manifest (FPN levels, ``anchor_config.json`` path).
    weights_path :
        Checkpoint file path; meaning depends on ``weight_source``.
    weight_source :
        ``"segmentation"`` — load ResNet body weights from the segmentation model,
        then train detection head + fine-tune FPN.

        ``"trained_detector"`` — load a full RetinaNet ``state_dict`` from a
        prior detection training run (same architecture / anchors).
    anchor_generator :
        Optional pre-built generator; if omitted, built from ``anchor_spec``.

    Notes
    -----
    **Backbone vs manifest.** ``anchor_spec.used_fpn_levels`` drives
    :func:`_build_fpn_backbone`. Implicit level coupling (P3→P2, P6/P7→P5,
    extra ``pool`` map) is documented in the module docstring; align downstream
    feature slicing with ``len(used_fpn_levels)``.

    **Weight loading.**

    - ``"segmentation"`` — only the ResNet ``body`` weights are loaded; FPN and
      head start from scratch. ``freeze_layers`` then freezes the bottom stages.
    - ``"trained_detector"`` — full ``state_dict``; architecture and anchors
      must match the checkpoint exactly.
    """
    if weight_source not in ("segmentation", "trained_detector"):
        raise ValueError(
            f"weight_source must be 'segmentation' or 'trained_detector', got {weight_source!r}."
        )

    image_mean = mean if mean is not None else DL_LIB_ETALON_MEAN_VECTOR
    image_std = std if std is not None else DL_LIB_ETALON_STD_VECTOR
    used_levels = anchor_spec.used_fpn_levels

    if anchor_generator is None:
        anchor_generator = build_anchor_generator_from_spec(anchor_spec)
        logger.info("AnchorGenerator from manifest: %s", anchor_spec.manifest_path)
    else:
        logger.info("Using provided AnchorGenerator")

    body, fpn_backbone = _build_fpn_backbone(used_levels, FULLY_TRAINABLE_BACKBONE_LAYERS)
    model = _assemble_retinanet(
        fpn_backbone,
        anchor_generator,
        num_classes,
        img_size,
        image_mean,
        image_std,
    )

    if weight_source == "segmentation":
        logger.info("Loading segmentation backbone weights: %s", weights_path)
        load_backbone_weights(body, weights_path)
        _freeze_backbone_body(model, freeze_layers)
    else:
        logger.info("Loading trained detector weights: %s", weights_path)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        _freeze_backbone_body(model, freeze_layers)

    return model.to(device)


# ---------------------------------------------------------------------------
# Convenience aliases (same interface, explicit weight source)
# ---------------------------------------------------------------------------


def build_dl_lab_etalon_retinanet_from_segmentation_weights(
    *,
    anchor_spec: AnchorTrainingSpec,
    segmentation_weights_path: str,
    img_size: Union[int, Tuple[int, int]],
    device: torch.device,
    num_classes: int = DL_LIB_ETALON_NUM_CLASSES,
    freeze_layers: int = 2,
    mean: List[float] | None = None,
    std: List[float] | None = None,
    anchor_generator: AnchorGenerator | None = None,
) -> RetinaNet:
    """Train / eval from scratch using segmentation-pretrained backbone."""
    return build_dl_lab_etalon_retinanet(
        anchor_spec=anchor_spec,
        img_size=img_size,
        device=device,
        weights_path=segmentation_weights_path,
        weight_source="segmentation",
        num_classes=num_classes,
        freeze_layers=freeze_layers,
        mean=mean,
        std=std,
        anchor_generator=anchor_generator,
    )


def build_dl_lab_etalon_retinanet_from_trained_weights(
    *,
    anchor_spec: AnchorTrainingSpec,
    trained_weights_path: str,
    img_size: Union[int, Tuple[int, int]],
    device: torch.device,
    num_classes: int = DL_LIB_ETALON_NUM_CLASSES,
    freeze_layers: int = 0,
    mean: List[float] | None = None,
    std: List[float] | None = None,
    anchor_generator: AnchorGenerator | None = None,
) -> RetinaNet:
    """Resume or evaluate from a full detector checkpoint."""
    return build_dl_lab_etalon_retinanet(
        anchor_spec=anchor_spec,
        img_size=img_size,
        device=device,
        weights_path=trained_weights_path,
        weight_source="trained_detector",
        num_classes=num_classes,
        freeze_layers=freeze_layers,
        mean=mean,
        std=std,
        anchor_generator=anchor_generator,
    )
