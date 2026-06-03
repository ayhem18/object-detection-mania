"""
End-to-end RetinaNet build integration tests.

For every subset of ``{P2, P3, P4, P5, P6, P7}``:

1. **Strict detector path** — a finalized config whose ``used_layers`` match the
   raw requested subset (no anchor-module transformation) must raise when the
   subset is not anchor-aligned.
2. **Full pipeline** — ``build_anchor_generator`` → ``build_retinanet`` → eval
   forward pass must succeed for every subset (transformation fixes invalid input).

Run all tests::

    uv run --project home_made_od python home_made_od/tests/retinanet/test_retinanet_full_build.py
"""

from __future__ import annotations

import itertools
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List, Sequence

import torch

from home_made_od.general.path_utils import write_json_file
from home_made_od.retinanet.retinanet_anchors import (
    ANCHOR_CONFIG_FILENAME,
    AnchorTrainingSpec,
    anchor_spec_from_path,
    build_anchor_generator,
)
from home_made_od.retinanet.retinanet_detector import build_retinanet
from home_made_od.retinanet.retinanet_levels import (
    ALL_FPN_LEVELS,
    is_anchor_aligned_layers,
    levels_to_layers,
)

ALL_LEVELS: tuple[str, ...] = ALL_FPN_LEVELS
EXPECTED_SUBSET_COUNT = 2 ** len(ALL_LEVELS)
IMAGE_SIZE = 512
NUM_CLASSES = 2
FPN_COEFFICIENT = 4

METHOD_PARAMETERS = {
    "fpn_coefficient": FPN_COEFFICIENT,
    "num_aspect_ratios": 2,
    "num_base_sizes": 1,
    "min_etalons_per_level": 1,
    "scales": [1.0],
    "seed": 0,
    "fallback_level": "P3",
    "max_iters": 10,
}

LEVEL_STRIDES = {
    "P2": 4,
    "P3": 8,
    "P4": 16,
    "P5": 32,
    "P6": 64,
    "P7": 128,
}

FULL_BUILD_TEST_METHODS: tuple[str, ...] = (
    "test_raw_finalized_config_raises_when_not_anchor_aligned",
    "test_full_pipeline_forward_pass",
)


def all_level_subsets() -> List[List[str]]:
    subsets: List[List[str]] = []
    for size in range(len(ALL_LEVELS) + 1):
        for combo in itertools.combinations(ALL_LEVELS, size):
            subsets.append(list(combo))
    return subsets


def subset_key(levels: Sequence[str]) -> str:
    return "+".join(levels) if levels else "none"


def _fpn_specs() -> List[dict]:
    return [
        {
            "level": level,
            "stride": LEVEL_STRIDES[level],
            "base_size": LEVEL_STRIDES[level] * FPN_COEFFICIENT,
            "nominal_area": (LEVEL_STRIDES[level] * FPN_COEFFICIENT) ** 2,
        }
        for level in ALL_LEVELS
    ]


def _bootstrap_level_params(requested_levels: Sequence[str]) -> tuple[dict, dict]:
    ratios = {level: [1.0, 1.5] for level in requested_levels}
    sizes = {
        level: [float(LEVEL_STRIDES[level] * FPN_COEFFICIENT)] for level in requested_levels
    }
    return ratios, sizes


def _make_anchor_training_spec(
    requested_levels: Sequence[str],
    config_path: Path,
) -> AnchorTrainingSpec:
    ratios, sizes = _bootstrap_level_params(requested_levels)
    return AnchorTrainingSpec(
        method="dimension_based",
        method_parameters=dict(METHOD_PARAMETERS),
        config_hash=f"integration-{subset_key(requested_levels)}",
        config_path=config_path,
        requested_fpn_levels=list(requested_levels),
        fpn_specs=_fpn_specs(),
        level_aspect_ratios=ratios,
        level_base_sizes=sizes,
        scales=[1.0],
        fpn_coefficient=FPN_COEFFICIENT,
        seed=0,
        box_count=1,
    )


def _write_raw_untransformed_config(
    config_path: Path,
    requested_levels: Sequence[str],
) -> None:
    """Persist a config that skips anchor-module layer transformation."""
    ratios, sizes = _bootstrap_level_params(requested_levels)
    raw_layers = levels_to_layers(requested_levels) if requested_levels else []
    write_json_file(
        config_path,
        {
            "config_hash": f"raw-{subset_key(requested_levels)}",
            "assignment_method": "dimension_based",
            "method_parameters": METHOD_PARAMETERS,
            "requested_fpn_levels": list(requested_levels),
            "used_fpn_levels": list(requested_levels),
            "used_layers": raw_layers,
            "level_provenance": {level: "original" for level in requested_levels},
            "fpn_specs": _fpn_specs(),
            "used_base_sizes": [
                LEVEL_STRIDES[level] * FPN_COEFFICIENT for level in requested_levels
            ],
            "coefficient": FPN_COEFFICIENT,
            "scales": [1.0],
            "level_aspect_ratios": ratios,
            "level_base_sizes": sizes,
            "seed": 0,
            "box_count": 1,
        },
    )



def _forward_pass(model: torch.nn.Module) -> None:
    image = torch.rand(3, IMAGE_SIZE, IMAGE_SIZE)
    model.eval()
    with torch.no_grad():
        outputs = model([image])
    assert isinstance(outputs, list)
    assert len(outputs) == 1


class TestFullBuildSubsetCount(unittest.TestCase):
    def test_subset_enumeration_count(self) -> None:
        self.assertEqual(len(all_level_subsets()), EXPECTED_SUBSET_COUNT)


def _build_full_pipeline_test_suite() -> unittest.TestSuite:
    class FullBuildTestCase(unittest.TestCase):
        requested_levels: List[str]
        combo_key: str
        tmp_dir: str

        def __init__(self, methodName: str, requested_levels: List[str]) -> None:
            super().__init__(methodName)
            self.requested_levels = requested_levels
            self.combo_key = subset_key(requested_levels)

        def __str__(self) -> str:
            return f"{self.combo_key} ({self._testMethodName})"

        def setUp(self) -> None:
            self._temp_dir = tempfile.TemporaryDirectory()
            self.tmp_dir = self._temp_dir.name
            self.config_path = Path(self.tmp_dir) / ANCHOR_CONFIG_FILENAME

        def tearDown(self) -> None:
            self._temp_dir.cleanup()

        def test_raw_finalized_config_raises_when_not_anchor_aligned(self) -> None:
            if is_anchor_aligned_layers(levels_to_layers(self.requested_levels)) if self.requested_levels else False:
                self.skipTest(f"{self.combo_key} is anchor-aligned without transformation.")

            _write_raw_untransformed_config(self.config_path, self.requested_levels)
            raw_spec = anchor_spec_from_path(self.config_path)

            good_config_path = Path(self.tmp_dir) / "good_anchor_config.json"
            good_spec = _make_anchor_training_spec(self.requested_levels, good_config_path)
            anchor_generator = build_anchor_generator(good_spec)

            with self.assertRaises(ValueError):
                build_retinanet(
                    anchor_spec=raw_spec,
                    num_classes=NUM_CLASSES,
                    img_size=IMAGE_SIZE,
                    device=torch.device("cpu"),
                    anchor_generator=anchor_generator,
                )

        def test_full_pipeline_forward_pass(self) -> None:
            spec = _make_anchor_training_spec(self.requested_levels, self.config_path)
            anchor_generator = build_anchor_generator(spec)
            finalized_spec = anchor_spec_from_path(self.config_path)

            model = build_retinanet(
                anchor_spec=finalized_spec,
                num_classes=NUM_CLASSES,
                img_size=IMAGE_SIZE,
                device=torch.device("cpu"),
                anchor_generator=anchor_generator,
            )
            _forward_pass(model)

    suite = unittest.TestSuite()
    for requested_levels in all_level_subsets():
        for method_name in FULL_BUILD_TEST_METHODS:
            suite.addTest(FullBuildTestCase(method_name, requested_levels))
    return suite


def suite() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    combined = unittest.TestSuite()
    combined.addTests(loader.loadTestsFromTestCase(TestFullBuildSubsetCount))
    combined.addTests(_build_full_pipeline_test_suite())
    return combined


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    return suite()


if __name__ == "__main__":
    argv = [arg for arg in sys.argv if arg != "--refresh-cache"]
    unittest.main(defaultTest="suite", argv=argv, verbosity=2)
