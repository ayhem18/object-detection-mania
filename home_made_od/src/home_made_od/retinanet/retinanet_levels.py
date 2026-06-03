"""
RetinaNet FPN layer model (layers 1–6 only, no P-level naming).

Unified layer indices
---------------------
Layers 1–4 map to ResNet50 body stages (strides 4 / 8 / 16 / 32).
Layers 5–6 map to ``LastLevelP6P7`` outputs (strides 64 / 128).

The mapping from input layers → output layers is derived from empirical
torchvision ground truth (see ``tests/retinanet/layer_combination_map.json``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

ALL_LAYERS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
BODY_LAYERS: frozenset[int] = frozenset({1, 2, 3, 4})
EXTRA_LAYERS: frozenset[int] = frozenset({5, 6})

LAYER_STRIDES: Dict[int, int] = {
    1: 4,
    2: 8,
    3: 16,
    4: 32,
    5: 64,
    6: 128,
}
STRIDE_TO_LAYER: Dict[int, int] = {stride: layer for layer, stride in LAYER_STRIDES.items()}

# LastLevelP6P7 always reads the raw ResNet layer4 tensor (2048 channels).
LAST_LEVEL_P6P7_SOURCE_LAYER = 4


@dataclass(frozen=True)
class RetinanetLayerMap:
    """Predicted backbone wiring and FPN outputs for a requested layer set."""

    input_layers: Tuple[int, ...]
    body_returned_layers: Tuple[int, ...]
    uses_extra_block: bool
    output_layers: Tuple[int, ...]
    is_valid: bool
    error: str | None = None


def normalize_input_layers(layers: Sequence[int]) -> List[int]:
    normalized = sorted({int(layer) for layer in layers})
    invalid = [layer for layer in normalized if layer not in ALL_LAYERS]
    if invalid:
        raise ValueError(f"Layers must be in {ALL_LAYERS}, got invalid {invalid}")
    return normalized


def _body_layers_from_input(layers: Sequence[int]) -> List[int]:
    return [layer for layer in layers if layer in BODY_LAYERS]


def _uses_extra_block(layers: Sequence[int]) -> bool:
    return any(layer in EXTRA_LAYERS for layer in layers)


def input_output_retinanet_layer_map(layers: Sequence[int]) -> RetinanetLayerMap:
    """
    Predict FPN output layers and backbone validity for a requested layer set.

    Build semantics (matching the empirical torchvision probe):
      - Body layers (1–4) are passed verbatim to ``returned_layers``.
      - If any of {5, 6} is requested, ``LastLevelP6P7`` is attached.
      - Otherwise torchvision uses ``LastLevelMaxPool``, appending one extra map.

    Validity rules inferred from ``layer_combination_map.json``:
      - At least one body layer (1–4) must be requested.
      - When layers 5 or 6 are requested, body layer 4 must also be requested
        because ``LastLevelP6P7`` branches from the raw stride-32 ResNet stage.

    Output rules:
      - Without extra block: ``output = body + [max(body) + 1]`` (pool level).
      - With extra block (and valid input): ``output = body + [5, 6]`` always;
        P6 and P7 are produced together regardless of whether 5, 6, or both
        were requested.
    """
    requested = normalize_input_layers(layers) if layers else []
    body_layers = _body_layers_from_input(requested)
    uses_extra_block = _uses_extra_block(requested)
    body_tuple = tuple(body_layers)

    if not requested:
        return RetinanetLayerMap(
            input_layers=(),
            body_returned_layers=(),
            uses_extra_block=False,
            output_layers=(),
            is_valid=False,
            error="At least one layer in {1, 2, 3, 4, 5, 6} is required.",
        )

    if not body_layers:
        return RetinanetLayerMap(
            input_layers=tuple(requested),
            body_returned_layers=(),
            uses_extra_block=uses_extra_block,
            output_layers=(),
            is_valid=False,
            error=(
                "Layers 5 and 6 require at least one body layer (1–4); "
                f"got input_layers={requested}."
            ),
        )

    if uses_extra_block and LAST_LEVEL_P6P7_SOURCE_LAYER not in body_layers:
        return RetinanetLayerMap(
            input_layers=tuple(requested),
            body_returned_layers=body_tuple,
            uses_extra_block=True,
            output_layers=(),
            is_valid=False,
            error=(
                f"Layers 5 and 6 require body layer {LAST_LEVEL_P6P7_SOURCE_LAYER} "
                f"(LastLevelP6P7 source); got input_layers={requested}."
            ),
        )

    if uses_extra_block:
        output_layers = tuple(body_layers + [5, 6])
    else:
        output_layers = tuple(body_layers + [max(body_layers) + 1])

    return RetinanetLayerMap(
        input_layers=tuple(requested),
        body_returned_layers=body_tuple,
        uses_extra_block=uses_extra_block,
        output_layers=output_layers,
        is_valid=True,
    )
