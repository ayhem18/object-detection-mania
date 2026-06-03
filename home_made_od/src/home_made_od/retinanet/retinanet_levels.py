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
from enum import Enum
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

# Minimum anchor-aligned layer set when the caller provides nothing.
DEFAULT_ANCHOR_ALIGNED_LAYERS: Tuple[int, ...] = (4, 5, 6)

# P-level naming (P2..P7 ↔ layers 1..6).
LEVEL_TO_LAYER: Dict[str, int] = {
    "P2": 1,
    "P3": 2,
    "P4": 3,
    "P5": 4,
    "P6": 5,
    "P7": 6,
}
LAYER_TO_LEVEL: Dict[int, str] = {layer: level for level, layer in LEVEL_TO_LAYER.items()}
ALL_FPN_LEVELS: Tuple[str, ...] = tuple(LEVEL_TO_LAYER.keys())


class InputLayerClassification(str, Enum):
    """How a requested layer set relates to backbone build and anchor wiring."""

    ANCHOR_ALIGNED = "anchor_aligned"
    COUNT_MISMATCH = "count_mismatch"
    BUILD_ERROR = "build_error"


@dataclass(frozen=True)
class RetinanetLayerMap:
    """Predicted backbone wiring and FPN outputs for a requested layer set."""

    input_layers: Tuple[int, ...]
    body_returned_layers: Tuple[int, ...]
    uses_extra_block: bool
    output_layers: Tuple[int, ...]
    is_valid: bool
    error: str | None = None


@dataclass(frozen=True)
class RetinanetLayerTransform:
    """Original layer request and its anchor-aligned superset."""

    input_layers: Tuple[int, ...]
    transformed_layers: Tuple[int, ...]
    classification: InputLayerClassification
    modified: bool


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


def classify_input_layers(layers: Sequence[int]) -> InputLayerClassification:
    """
    Classify a layer request before anchor/backbone wiring.

    - ``BUILD_ERROR``: torchvision build would fail.
    - ``COUNT_MISMATCH``: build succeeds but ``len(output) != len(input)``.
    - ``ANCHOR_ALIGNED``: build succeeds and feature-map count matches input count.
    """
    requested = normalize_input_layers(layers) if layers else []
    layer_map = input_output_retinanet_layer_map(requested)

    if not layer_map.is_valid:
        return InputLayerClassification.BUILD_ERROR

    if len(layer_map.output_layers) != len(requested):
        return InputLayerClassification.COUNT_MISMATCH

    return InputLayerClassification.ANCHOR_ALIGNED


def is_anchor_aligned_layers(layers: Sequence[int]) -> bool:
    return classify_input_layers(layers) == InputLayerClassification.ANCHOR_ALIGNED


def levels_to_layers(levels: Sequence[str]) -> List[int]:
    """Map FPN level names (P2..P7) to unified layer indices (1..6)."""
    unique_levels = sorted(set(levels))
    unknown = [level for level in unique_levels if level not in LEVEL_TO_LAYER]
    if unknown:
        raise ValueError(f"Unknown FPN levels {unknown}; expected one of {ALL_FPN_LEVELS}.")
    return sorted(LEVEL_TO_LAYER[level] for level in unique_levels)


def layers_to_levels(layers: Sequence[int]) -> List[str]:
    """Map unified layer indices (1..6) back to FPN level names (P2..P7)."""
    normalized = normalize_input_layers(layers)
    return [LAYER_TO_LEVEL[layer] for layer in normalized]


def validate_anchor_aligned_layers(layers: Sequence[int]) -> None:
    """Raise if *layers* cannot be wired to matching feature-map and anchor counts."""
    classification = classify_input_layers(layers)
    layer_map = input_output_retinanet_layer_map(layers)

    if classification == InputLayerClassification.BUILD_ERROR:
        raise ValueError(layer_map.error or f"Invalid RetinaNet layers: {list(layers)}")
    if classification == InputLayerClassification.COUNT_MISMATCH:
        raise ValueError(
            "RetinaNet layer count must match FPN output count; "
            f"input_layers={list(layers)}, output_layers={list(layer_map.output_layers)}."
        )


def validate_anchor_aligned_levels(levels: Sequence[str]) -> None:
    """Level-name wrapper around :func:`validate_anchor_aligned_layers`."""
    validate_anchor_aligned_layers(levels_to_layers(levels))


def _apply_empty_input_fallback() -> set[int]:
    # --- Section 1: empty input fallback ---------------------------------
    # An empty request cannot build a backbone. Default to the smallest
    # anchor-aligned set observed in the empirical map: {4, 5, 6}.
    return set(DEFAULT_ANCHOR_ALIGNED_LAYERS)


def _ensure_last_level_p6p7_source_layer(transformed: set[int]) -> None:
    # --- Section 2: LastLevelP6P7 requires body layer 4 ------------------
    # Any request for layer 5 or 6 attaches LastLevelP6P7, which reads the
    # raw stride-32 ResNet stage. Body layer 4 must therefore be present.
    if transformed & EXTRA_LAYERS:
        transformed.add(LAST_LEVEL_P6P7_SOURCE_LAYER)


def _ensure_inseparable_extra_layers(transformed: set[int]) -> None:
    # --- Section 3: layers 5 and 6 are inseparable -----------------------
    # LastLevelP6P7 always emits both P6 and P7 feature maps. If either
    # high layer is requested, both must appear in the transformed input.
    if transformed & EXTRA_LAYERS:
        transformed.update(EXTRA_LAYERS)


def _ensure_anchor_count_alignment(transformed: set[int]) -> None:
    # --- Section 4: anchor / feature-map count alignment -------------------
    # Body-only builds append one extra pool map, so len(output) = len(body)+1.
    # Anchor wiring requires len(input) == len(output). Empirically, every
    # anchor-aligned combination includes {4, 5, 6}; adding that set switches
    # the build to LastLevelP6P7 and makes len(output) == len(transformed).
    if not is_anchor_aligned_layers(sorted(transformed)):
        transformed.update({LAST_LEVEL_P6P7_SOURCE_LAYER, *EXTRA_LAYERS})


def transform_input_layers(layers: Sequence[int]) -> RetinanetLayerTransform:
    """
    Return an anchor-aligned superset of the requested layers.

    If the input is already anchor-aligned, it is returned unchanged.
    Otherwise layers are added in the section order documented above until the
    transformed set satisfies:

      a. original input layers are preserved
      b. the transformed set is build-valid
      c. ``len(output_layers) == len(transformed_layers)``
    """
    original = normalize_input_layers(layers) if layers else []
    classification = classify_input_layers(original)

    if classification == InputLayerClassification.ANCHOR_ALIGNED:
        original_tuple = tuple(original)
        return RetinanetLayerTransform(
            input_layers=original_tuple,
            transformed_layers=original_tuple,
            classification=classification,
            modified=False,
        )

    transformed = set(original)

    if not transformed:
        transformed = _apply_empty_input_fallback()
    else:
        _ensure_last_level_p6p7_source_layer(transformed)
        _ensure_inseparable_extra_layers(transformed)
        _ensure_anchor_count_alignment(transformed)

    transformed_layers = tuple(sorted(transformed))
    transformed_map = input_output_retinanet_layer_map(transformed_layers)

    if not transformed_map.is_valid:
        raise RuntimeError(
            f"Transformed layers {list(transformed_layers)} are not build-valid; "
            f"error={transformed_map.error!r}"
        )
    if len(transformed_map.output_layers) != len(transformed_layers):
        raise RuntimeError(
            f"Transformed layers {list(transformed_layers)} are not anchor-aligned; "
            f"output_layers={list(transformed_map.output_layers)}"
        )
    if not set(original).issubset(set(transformed_layers)):
        raise RuntimeError(
            f"Transformed layers {list(transformed_layers)} dropped original "
            f"input layers {original}."
        )

    return RetinanetLayerTransform(
        input_layers=tuple(original),
        transformed_layers=transformed_layers,
        classification=classification,
        modified=True,
    )
