"""
Layer-based FPN backbone ground-truth map for RetinaNet.

Testing methodology
===================

Goal
----
Derive and cache the exact relationship between *requested* FPN layers (1–6)
and *produced* FPN layers when wiring torchvision's ResNet50+FPN backbone the
same way ``retinanet_detector`` does. The cached map is the empirical ground
truth; ``home_made_od.retinanet.retinanet_levels.input_output_retinanet_layer_map``
is the analytical model we validate against it.

Layer numbering
---------------
We work exclusively in layer indices (levels discarded for now):

  layer 1..4  — ResNet50 body stages ``layer1``..``layer4`` (strides 4/8/16/32)
  layer 5..6  — ``LastLevelP6P7`` outputs (strides 64/128)

Build probe (synthetic backbone)
--------------------------------
For each subset of ``{1, 2, 3, 4, 5, 6}`` we call ``build_backbone_from_layers``:

1. Split the request into body layers (``input ∩ {1,2,3,4}``) and high layers
   (``input ∩ {5,6}``).
2. If any high layer is requested → attach ``LastLevelP6P7(in_channels=2048)``.
   Otherwise torchvision defaults to ``LastLevelMaxPool``.
3. Pass body layers directly as ``returned_layers`` to
   ``_resnet_fpn_extractor`` — no extrapolation inside the probe.

Ground-truth extraction (torchvision as oracle)
-----------------------------------------------
When the build succeeds we treat the live module as source of truth:

1. **Structural introspection** — read ``backbone.body.return_layers`` keys
   (``"layer1"`` → 1, …) to record which ResNet stages were hooked.
2. **Functional introspection** — run a square dummy forward pass and map each
   output tensor's stride back to a unified layer index via ``LAYER_STRIDES``.
   Strides that do not correspond to layers 1–6 (e.g. the internal ``pool``
   key at ``2 × max(body stride)`` when ``LastLevelMaxPool`` is active) are
   mapped to layer ``max(body) + 1`` because they occupy the same stride slot
   in our numbering (e.g. body ``[4]`` → output strides 32 and 64 → layers 4
   and 5).

Each combination is recorded as either:

  - ``status: ok`` with ``input_layers``, ``body_returned_layers``,
    ``uses_extra_block``, ``output_layers``, ``output_strides``
  - ``status: error`` with ``error_type`` and ``error`` message

Caching
-------
All 2^6 = 64 combinations are stored in ``layer_combination_map.json`` next to
this file. Regenerate with::

    uv run --project home_made_od python home_made_od/tests/retinanet/test_architecture_building.py --refresh-cache

Analytical model validation
---------------------------
Every one of the 64 combinations is exercised by ``LayerCombinationTestCase``
(three assertions per combination):

  - analytical map matches the cached ground truth
  - live torchvision probe matches the cached ground truth
  - analytical map matches the live probe

Run all tests::

    uv run --project home_made_od python -m unittest home_made_od.tests.retinanet.test_architecture_building -v
"""

from __future__ import annotations

import itertools
import json
import sys
import unittest

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Sequence, TypedDict
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.retinanet import LastLevelP6P7

from home_made_od.retinanet.retinanet_levels import (
    ALL_LAYERS,
    BODY_LAYERS,
    EXTRA_LAYERS,
    STRIDE_TO_LAYER,
    input_output_retinanet_layer_map,
)

TEST_DIR = Path(__file__).parent
LAYER_COMBINATION_MAP_PATH = TEST_DIR / "layer_combination_map.json"
EXPECTED_COMBINATION_COUNT = 2 ** len(ALL_LAYERS)

IMAGE_SIZE = 512
TRAINABLE_BACKBONE_LAYERS = 5
LAST_LEVEL_P6P7_IN_CHANNELS = 2048
FPN_OUT_CHANNELS = 256

LAYER_COMBINATION_TEST_METHODS: tuple[str, ...] = (
    "test_analytical_map_matches_ground_truth_cache",
    "test_live_probe_matches_ground_truth_cache",
    "test_analytical_map_matches_live_probe",
)


class LayerCombinationOk(TypedDict):
    status: str
    input_layers: List[int]
    body_returned_layers: List[int]
    uses_extra_block: bool
    output_layers: List[int]
    output_strides: List[int]


class LayerCombinationError(TypedDict):
    status: str
    input_layers: List[int]
    uses_extra_block: bool
    error_type: str
    error: str


LayerCombinationResult = LayerCombinationOk | LayerCombinationError


def all_layer_combinations() -> List[List[int]]:
    """Every subset of {1, 2, 3, 4, 5, 6}, including the empty set."""
    combinations: List[List[int]] = []
    for size in range(len(ALL_LAYERS) + 1):
        for combo in itertools.combinations(ALL_LAYERS, size):
            combinations.append(list(combo))
    return combinations


def _normalize_layers(layers: Sequence[int]) -> List[int]:
    normalized = sorted(set(int(layer) for layer in layers))
    invalid = [layer for layer in normalized if layer not in ALL_LAYERS]
    if invalid:
        raise ValueError(f"Layers must be in {ALL_LAYERS}, got invalid {invalid}")
    return normalized


def build_backbone_from_layers(layers: Sequence[int]) -> nn.Module:
    """
    Build a ResNet50+FPN backbone for the requested layer set.

    Body layers (1–4) are passed directly to ``returned_layers``.
    Layers 5–6 trigger ``LastLevelP6P7``; they are not ResNet stages.
    """
    requested = _normalize_layers(layers)
    body_returned_layers = [layer for layer in requested if layer in BODY_LAYERS]
    uses_extra_block = any(layer in EXTRA_LAYERS for layer in requested)

    body = resnet50(weights=None)
    if uses_extra_block:
        extra_blocks = LastLevelP6P7(LAST_LEVEL_P6P7_IN_CHANNELS, FPN_OUT_CHANNELS)
        return _resnet_fpn_extractor(
            body,
            trainable_layers=TRAINABLE_BACKBONE_LAYERS,
            returned_layers=body_returned_layers,
            extra_blocks=extra_blocks,
        )

    return _resnet_fpn_extractor(
        body,
        trainable_layers=TRAINABLE_BACKBONE_LAYERS,
        returned_layers=body_returned_layers,
    )


def introspect_body_returned_layers(backbone: nn.Module) -> List[int]:
    return sorted(int(name.removeprefix("layer")) for name in backbone.body.return_layers)


def introspect_output_layers(
    backbone: nn.Module,
    image_size: int = IMAGE_SIZE,
) -> tuple[List[int], List[int]]:
    """Map forward-pass feature strides back to unified layer indices 1..6."""
    features = backbone(torch.zeros(1, 3, image_size, image_size))
    output_layers: List[int] = []
    output_strides: List[int] = []

    for tensor in features.values():
        stride_h = image_size // tensor.shape[-2]
        stride_w = image_size // tensor.shape[-1]
        if stride_h != stride_w:
            raise ValueError(
                f"Non-square stride: h={stride_h}, w={stride_w} for shape={tuple(tensor.shape)}"
            )
        if stride_h not in STRIDE_TO_LAYER:
            continue
        layer = STRIDE_TO_LAYER[stride_h]
        if layer not in output_layers:
            output_layers.append(layer)
            output_strides.append(stride_h)

    paired = sorted(zip(output_layers, output_strides), key=lambda item: item[0])
    if not paired:
        return [], []
    layers, strides = zip(*paired)
    return list(layers), list(strides)


def layer_combination_key(layers: Sequence[int]) -> str:
    normalized = _normalize_layers(layers) if layers else []
    return ",".join(str(layer) for layer in normalized) if normalized else "none"


def evaluate_layer_combination(layers: Sequence[int]) -> LayerCombinationResult:
    requested = [] if not layers else _normalize_layers(layers)
    uses_extra_block = any(layer in EXTRA_LAYERS for layer in requested)

    try:
        backbone = build_backbone_from_layers(requested)
        output_layers, output_strides = introspect_output_layers(backbone)
        return LayerCombinationOk(
            status="ok",
            input_layers=requested,
            body_returned_layers=introspect_body_returned_layers(backbone),
            uses_extra_block=uses_extra_block,
            output_layers=output_layers,
            output_strides=output_strides,
        )
    except Exception as exc:
        return LayerCombinationError(
            status="error",
            input_layers=requested,
            uses_extra_block=uses_extra_block,
            error_type=type(exc).__name__,
            error=str(exc),
        )


def build_layer_combination_map(*, force_refresh: bool = False) -> Dict[str, LayerCombinationResult]:
    """
    Evaluate every subset of {1, 2, 3, 4, 5, 6} and map it to backbone output or error.

    Results are cached at ``layer_combination_map.json`` next to this test file.
    """
    if not force_refresh and LAYER_COMBINATION_MAP_PATH.exists():
        cached = json.loads(LAYER_COMBINATION_MAP_PATH.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and len(cached) == EXPECTED_COMBINATION_COUNT:
            return cached

    combination_map: Dict[str, LayerCombinationResult] = {}
    for size in tqdm(range(len(ALL_LAYERS) + 1), desc="Building layer combination map"):
        for combo in itertools.combinations(ALL_LAYERS, size):
            key = layer_combination_key(combo)
            combination_map[key] = evaluate_layer_combination(combo)

    LAYER_COMBINATION_MAP_PATH.write_text(
        json.dumps(combination_map, indent=2),
        encoding="utf-8",
    )
    return combination_map


def load_layer_combination_map(*, force_refresh: bool = False) -> Dict[str, LayerCombinationResult]:
    return build_layer_combination_map(force_refresh=force_refresh)


def _assert_analytical_matches_cache(
    layers: List[int],
    cached: LayerCombinationResult,
) -> None:
    predicted = input_output_retinanet_layer_map(layers)

    assert predicted.input_layers == tuple(cached["input_layers"])
    assert predicted.uses_extra_block == cached["uses_extra_block"]

    if cached["status"] == "ok":
        assert predicted.is_valid
        assert predicted.error is None
        assert list(predicted.body_returned_layers) == cached["body_returned_layers"]
        assert list(predicted.output_layers) == cached["output_layers"]
    else:
        assert not predicted.is_valid
        assert predicted.error is not None
        assert predicted.output_layers == ()


def _assert_live_probe_matches_cache(
    layers: List[int],
    cached: LayerCombinationResult,
) -> None:
    live = evaluate_layer_combination(layers)

    assert live["input_layers"] == cached["input_layers"]
    assert live["uses_extra_block"] == cached["uses_extra_block"]
    assert live["status"] == cached["status"]

    if cached["status"] == "ok":
        assert live["body_returned_layers"] == cached["body_returned_layers"]
        assert live["output_layers"] == cached["output_layers"]
        assert live["output_strides"] == cached["output_strides"]
    else:
        assert live["error_type"] == cached["error_type"]
        assert live["error"] == cached["error"]


def _assert_analytical_matches_live_probe(layers: List[int]) -> None:
    predicted = input_output_retinanet_layer_map(layers)
    live = evaluate_layer_combination(layers)

    assert predicted.input_layers == tuple(live["input_layers"])
    assert predicted.uses_extra_block == live["uses_extra_block"]

    if live["status"] == "ok":
        assert predicted.is_valid
        assert list(predicted.body_returned_layers) == live["body_returned_layers"]
        assert list(predicted.output_layers) == live["output_layers"]
    else:
        assert not predicted.is_valid
        assert predicted.output_layers == ()


class TestLayerCombinationMapCache(unittest.TestCase):
    """Sanity checks on the cached ground-truth artifact."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.combination_map = load_layer_combination_map()

    def test_cache_file_exists(self) -> None:
        self.assertTrue(LAYER_COMBINATION_MAP_PATH.exists())

    def test_cache_covers_every_combination(self) -> None:
        self.assertEqual(len(self.combination_map), EXPECTED_COMBINATION_COUNT)
        expected_keys = {layer_combination_key(combo) for combo in all_layer_combinations()}
        self.assertEqual(set(self.combination_map.keys()), expected_keys)


def _build_layer_combination_test_suite() -> unittest.TestSuite:
    """
    Build one unittest instance per (combination, test method) pair.

    64 combinations × 3 test methods = 192 tests.
    """
    combination_map = load_layer_combination_map()

    class LayerCombinationTestCase(unittest.TestCase):
        layers: List[int]
        combo_key: str
        cached: LayerCombinationResult

        def __init__(
            self,
            methodName: str,
            layers: List[int],
            cached: LayerCombinationResult,
        ) -> None:
            super().__init__(methodName)
            self.layers = layers
            self.combo_key = layer_combination_key(layers)
            self.cached = cached

        def __str__(self) -> str:
            return f"{self.combo_key} ({self._testMethodName})"

        def test_analytical_map_matches_ground_truth_cache(self) -> None:
            _assert_analytical_matches_cache(self.layers, self.cached)

        def test_live_probe_matches_ground_truth_cache(self) -> None:
            _assert_live_probe_matches_cache(self.layers, self.cached)

        def test_analytical_map_matches_live_probe(self) -> None:
            _assert_analytical_matches_live_probe(self.layers)

    suite = unittest.TestSuite()
    for layers in all_layer_combinations():
        combo_key = layer_combination_key(layers)
        cached = combination_map[combo_key]
        for method_name in LAYER_COMBINATION_TEST_METHODS:
            suite.addTest(LayerCombinationTestCase(method_name, layers, cached))
    return suite


def suite() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    combined = unittest.TestSuite()
    combined.addTests(loader.loadTestsFromTestCase(TestLayerCombinationMapCache))
    combined.addTests(_build_layer_combination_test_suite())
    return combined


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    return suite()


if __name__ == "__main__":
    if "--refresh-cache" in sys.argv:
        load_layer_combination_map(force_refresh=True)
        print(f"Wrote {LAYER_COMBINATION_MAP_PATH}")
    else:
        argv = [arg for arg in sys.argv if arg != "--refresh-cache"]
        unittest.main(defaultTest="suite", argv=argv, verbosity=2)
