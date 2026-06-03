"""
Tests for RetinaNet layer-request classification and transformation.

Classification cache
--------------------
``layer_transform_classification.json`` stores, for every subset of
``{1, 2, 3, 4, 5, 6}``, whether the request is:

  - ``anchor_aligned``
  - ``count_mismatch``
  - ``build_error``

Transformation tests
--------------------
For every combination, ``transform_input_layers`` must satisfy:

  a. original input layers are preserved in the transformed set
  b. transformed layers are build-valid
  c. ``len(output_layers) == len(transformed_layers)``

Run all tests::

    uv run --project home_made_od python home_made_od/tests/retinanet/test_retinanet_layer_transform.py
"""

from __future__ import annotations

import itertools
import json
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Sequence, TypedDict

from home_made_od.retinanet.retinanet_levels import (
    ALL_LAYERS,
    InputLayerClassification,
    classify_input_layers,
    input_output_retinanet_layer_map,
    transform_input_layers,
)

TEST_DIR = Path(__file__).parent
LAYER_TRANSFORM_CLASSIFICATION_PATH = TEST_DIR / "layer_transform_classification.json"
EXPECTED_COMBINATION_COUNT = 2 ** len(ALL_LAYERS)

TRANSFORM_TEST_METHODS: tuple[str, ...] = (
    "test_classification_matches_cache",
    "test_transform_preserves_original_layers",
    "test_transform_is_build_valid",
    "test_transform_is_anchor_aligned",
    "test_transform_idempotent_for_anchor_aligned_inputs",
)


class LayerTransformClassificationEntry(TypedDict):
    input_layers: List[int]
    classification: str


def all_layer_combinations() -> List[List[int]]:
    combinations: List[List[int]] = []
    for size in range(len(ALL_LAYERS) + 1):
        for combo in itertools.combinations(ALL_LAYERS, size):
            combinations.append(list(combo))
    return combinations


def layer_combination_key(layers: Sequence[int]) -> str:
    if not layers:
        return "none"
    return ",".join(str(layer) for layer in sorted(set(int(layer) for layer in layers)))


def build_layer_transform_classification_map() -> Dict[str, LayerTransformClassificationEntry]:
    classification_map: Dict[str, LayerTransformClassificationEntry] = {}
    for layers in all_layer_combinations():
        key = layer_combination_key(layers)
        classification_map[key] = LayerTransformClassificationEntry(
            input_layers=layers,
            classification=classify_input_layers(layers).value,
        )
    return classification_map


def load_layer_transform_classification_map(*, force_refresh: bool = False) -> Dict[str, LayerTransformClassificationEntry]:
    if not force_refresh and LAYER_TRANSFORM_CLASSIFICATION_PATH.exists():
        cached = json.loads(LAYER_TRANSFORM_CLASSIFICATION_PATH.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and len(cached) == EXPECTED_COMBINATION_COUNT:
            return cached

    classification_map = build_layer_transform_classification_map()
    LAYER_TRANSFORM_CLASSIFICATION_PATH.write_text(
        json.dumps(classification_map, indent=2),
        encoding="utf-8",
    )
    return classification_map


def _assert_transform_satisfies_contract(layers: List[int]) -> None:
    result = transform_input_layers(layers)
    transformed_map = input_output_retinanet_layer_map(result.transformed_layers)

    assert set(result.input_layers).issubset(set(result.transformed_layers))
    assert transformed_map.is_valid
    assert len(transformed_map.output_layers) == len(result.transformed_layers)

    if result.classification == InputLayerClassification.ANCHOR_ALIGNED:
        assert not result.modified
        assert result.transformed_layers == result.input_layers
    else:
        assert result.modified


class TestLayerTransformClassificationCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classification_map = load_layer_transform_classification_map()

    def test_cache_file_exists(self) -> None:
        self.assertTrue(LAYER_TRANSFORM_CLASSIFICATION_PATH.exists())

    def test_cache_covers_every_combination(self) -> None:
        self.assertEqual(len(self.classification_map), EXPECTED_COMBINATION_COUNT)
        expected_keys = {layer_combination_key(combo) for combo in all_layer_combinations()}
        self.assertEqual(set(self.classification_map.keys()), expected_keys)


def _build_layer_transform_test_suite() -> unittest.TestSuite:
    classification_map = load_layer_transform_classification_map()

    class LayerTransformTestCase(unittest.TestCase):
        layers: List[int]
        combo_key: str
        cached: LayerTransformClassificationEntry

        def __init__(
            self,
            methodName: str,
            layers: List[int],
            cached: LayerTransformClassificationEntry,
        ) -> None:
            super().__init__(methodName)
            self.layers = layers
            self.combo_key = layer_combination_key(layers)
            self.cached = cached

        def __str__(self) -> str:
            return f"{self.combo_key} ({self._testMethodName})"

        def test_classification_matches_cache(self) -> None:
            self.assertEqual(
                classify_input_layers(self.layers).value,
                self.cached["classification"],
            )

        def test_transform_preserves_original_layers(self) -> None:
            result = transform_input_layers(self.layers)
            self.assertTrue(set(result.input_layers).issubset(set(result.transformed_layers)))

        def test_transform_is_build_valid(self) -> None:
            result = transform_input_layers(self.layers)
            transformed_map = input_output_retinanet_layer_map(result.transformed_layers)
            self.assertTrue(transformed_map.is_valid, transformed_map.error)

        def test_transform_is_anchor_aligned(self) -> None:
            result = transform_input_layers(self.layers)
            transformed_map = input_output_retinanet_layer_map(result.transformed_layers)
            self.assertEqual(len(transformed_map.output_layers), len(result.transformed_layers))

        def test_transform_idempotent_for_anchor_aligned_inputs(self) -> None:
            first = transform_input_layers(self.layers)
            second = transform_input_layers(list(first.transformed_layers))
            self.assertEqual(first.transformed_layers, second.transformed_layers)
            self.assertFalse(second.modified)

    suite = unittest.TestSuite()
    for layers in all_layer_combinations():
        combo_key = layer_combination_key(layers)
        cached = classification_map[combo_key]
        for method_name in TRANSFORM_TEST_METHODS:
            suite.addTest(LayerTransformTestCase(method_name, layers, cached))
    return suite


def suite() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    combined = unittest.TestSuite()
    combined.addTests(loader.loadTestsFromTestCase(TestLayerTransformClassificationCache))
    combined.addTests(_build_layer_transform_test_suite())
    return combined


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    return suite()


if __name__ == "__main__":
    if "--refresh-cache" in sys.argv:
        load_layer_transform_classification_map(force_refresh=True)
        print(f"Wrote {LAYER_TRANSFORM_CLASSIFICATION_PATH}")
    else:
        argv = [arg for arg in sys.argv if arg != "--refresh-cache"]
        unittest.main(defaultTest="suite", argv=argv, verbosity=2)
