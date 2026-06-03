"""
RetinaNet anchor optimization, config I/O, and torchvision AnchorGenerator wiring.

Clustering is detector-agnostic (``anchor_computation_strategies``). This module
finalizes FPN levels through ``retinanet_levels``, derives anchor parameters for
added levels, persists ``anchor_config.json``, and builds ``AnchorGenerator``.

Public entry point: :func:`build_anchor_generator`.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

import numpy as np
from torchvision.models.detection.anchor_utils import AnchorGenerator

from home_made_od.anchors.anchor_computation_strategies import (
    AnchorOptimizationResult,
    FPN_LEVEL_ORDER,
    BoxDimensions,
    compute_optimized_anchors,
    validate_method_parameters,
)
from home_made_od.general.path_utils import read_json_file, write_json_file
from home_made_od.retinanet.retinanet_levels import (
    LAYER_STRIDES,
    layers_to_levels,
    levels_to_layers,
    transform_input_layers,
)

logger = logging.getLogger(__name__)

ANCHOR_CONFIG_FILENAME = "anchor_config.json"
RETINANET_FPN_LEVEL_ORDER: tuple[str, ...] = FPN_LEVEL_ORDER
RETINANET_FPN_LEVEL_STRIDES: Dict[str, int] = {
    f"P{layer + 1}": LAYER_STRIDES[layer] for layer in (1, 2, 3, 4, 5, 6)
}
RESNET_OUT_CHANNELS: Dict[int, int] = {1: 256, 2: 512, 3: 1024, 4: 2048}

LevelSource = Literal["original", "added"]


@dataclass(frozen=True)
class FinalizedFpnLevels:
    requested_fpn_levels: tuple[str, ...]
    used_fpn_levels: tuple[str, ...]
    used_layers: tuple[int, ...]
    level_provenance: Dict[str, LevelSource]


@dataclass(frozen=True)
class AnchorTrainingSpec:
    """Anchor computation strategy plus clustered per-level parameters."""

    method: str
    method_parameters: Dict[str, Any]
    config_hash: str
    config_path: Path
    requested_fpn_levels: List[str]
    fpn_specs: List[Dict[str, Any]]
    level_aspect_ratios: Dict[str, List[float]]
    level_base_sizes: Dict[str, List[float]]
    scales: List[float]
    fpn_coefficient: int
    seed: int
    box_count: int
    used_fpn_levels: List[str] | None = None
    level_provenance: Dict[str, LevelSource] | None = None

    @classmethod
    def from_optimization(
        cls,
        result: AnchorOptimizationResult,
        config_path: str | Path,
    ) -> AnchorTrainingSpec:
        return cls(
            method=result.method,
            method_parameters=dict(result.method_parameters),
            config_hash=result.config_hash,
            config_path=Path(config_path),
            requested_fpn_levels=list(result.used_fpn_levels),
            fpn_specs=list(result.fpn_specs),
            level_aspect_ratios=dict(result.level_aspect_ratios),
            level_base_sizes=dict(result.level_base_sizes),
            scales=list(result.scales),
            fpn_coefficient=result.fpn_coefficient,
            seed=result.seed,
            box_count=result.box_count,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any], config_path: str | Path) -> AnchorTrainingSpec:
        path = Path(config_path)
        requested = list(config.get("requested_fpn_levels", config["used_fpn_levels"]))
        return cls(
            method=config["assignment_method"],
            method_parameters=dict(config["method_parameters"]),
            config_hash=config["config_hash"],
            config_path=path.resolve(),
            requested_fpn_levels=requested,
            fpn_specs=list(config["fpn_specs"]),
            level_aspect_ratios={
                level: list(ratios) for level, ratios in config["level_aspect_ratios"].items()
            },
            level_base_sizes={
                level: list(sizes) for level, sizes in config["level_base_sizes"].items()
            },
            scales=list(config["scales"]),
            fpn_coefficient=int(config["coefficient"]),
            seed=int(config["seed"]),
            box_count=int(config["box_count"]),
            used_fpn_levels=list(config["used_fpn_levels"]),
            level_provenance=dict(config.get("level_provenance", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "method_parameters": self.method_parameters,
            "config_hash": self.config_hash,
            "config_path": str(self.config_path),
            "requested_fpn_levels": self.requested_fpn_levels,
            "used_fpn_levels": self.used_fpn_levels or self.requested_fpn_levels,
            "level_provenance": self.level_provenance or {},
            "fpn_coefficient": self.fpn_coefficient,
            "seed": self.seed,
            "box_count": self.box_count,
        }


def _sort_fpn_levels(levels: Iterable[str]) -> List[str]:
    order_index = {name: index for index, name in enumerate(RETINANET_FPN_LEVEL_ORDER)}
    return sorted(levels, key=lambda level: order_index.get(level, len(RETINANET_FPN_LEVEL_ORDER)))


def _spec_by_level(fpn_specs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {spec["level"]: spec for spec in fpn_specs}


def _num_aspect_ratios(level_aspect_ratios: Dict[str, List[float]], *, default: int = 2) -> int:
    if not level_aspect_ratios:
        return default
    return len(next(iter(level_aspect_ratios.values())))


def _bootstrap_aspect_ratios(num_aspect_ratios: int) -> List[float]:
    return np.linspace(0.25, 1.75, num_aspect_ratios).tolist()


def _bootstrap_base_sizes_from_spec(
    level: str,
    fpn_specs_by_level: Dict[str, Dict[str, Any]],
    scales: List[float],
) -> List[float]:
    spec = fpn_specs_by_level[level]
    return [spec["base_size"] * scale for scale in scales]


def _derive_level_anchors_from_reference(
    target_level: str,
    reference_level: str,
    level_aspect_ratios: Dict[str, List[float]],
    level_base_sizes: Dict[str, List[float]],
    fpn_specs_by_level: Dict[str, Dict[str, Any]],
) -> None:
    ref_spec = fpn_specs_by_level[reference_level]
    tgt_spec = fpn_specs_by_level[target_level]
    size_scale = tgt_spec["base_size"] / ref_spec["base_size"]

    level_aspect_ratios[target_level] = list(level_aspect_ratios[reference_level])
    level_base_sizes[target_level] = [
        float(size) * size_scale for size in level_base_sizes[reference_level]
    ]


def _nearest_original_level_with_params(
    target_level: str,
    original_levels: Iterable[str],
    level_aspect_ratios: Dict[str, List[float]],
) -> Optional[str]:
    order_index = {name: index for index, name in enumerate(RETINANET_FPN_LEVEL_ORDER)}
    target_index = order_index[target_level]
    available = [
        level
        for level in original_levels
        if level in level_aspect_ratios and level in order_index
    ]
    if not available:
        return None
    return min(available, key=lambda level: abs(order_index[level] - target_index))


def _derive_params_for_added_level(
    target_level: str,
    original_levels: set[str],
    level_aspect_ratios: Dict[str, List[float]],
    level_base_sizes: Dict[str, List[float]],
    fpn_specs_by_level: Dict[str, Dict[str, Any]],
    *,
    num_aspect_ratios: int,
    scales: List[float],
) -> None:
    if target_level in level_aspect_ratios and target_level in level_base_sizes:
        return

    reference_level = _nearest_original_level_with_params(
        target_level,
        original_levels,
        level_aspect_ratios,
    )
    if reference_level is None:
        logger.warning(
            "No reference anchors for added level %s — bootstrapping from fpn_specs.",
            target_level,
        )
        level_aspect_ratios[target_level] = _bootstrap_aspect_ratios(num_aspect_ratios)
        level_base_sizes[target_level] = _bootstrap_base_sizes_from_spec(
            target_level,
            fpn_specs_by_level,
            scales,
        )
        return

    logger.warning(
        "Added level %s — deriving anchor params from %s.",
        target_level,
        reference_level,
    )
    _derive_level_anchors_from_reference(
        target_level,
        reference_level,
        level_aspect_ratios,
        level_base_sizes,
        fpn_specs_by_level,
    )


def _finalize_fpn_levels(
    requested_fpn_levels: List[str],
    level_aspect_ratios: Dict[str, List[float]],
    level_base_sizes: Dict[str, List[float]],
    fpn_specs: List[Dict[str, Any]],
    *,
    scales: List[float],
) -> tuple[FinalizedFpnLevels, Dict[str, List[float]], Dict[str, List[float]]]:
    requested = _sort_fpn_levels(requested_fpn_levels)
    original_layers = levels_to_layers(requested)
    transform = transform_input_layers(original_layers)
    final_layers = transform.transformed_layers
    final_levels = layers_to_levels(final_layers)

    original_set = set(requested)
    ratios = dict(level_aspect_ratios)
    sizes = dict(level_base_sizes)
    by_level = _spec_by_level(fpn_specs)
    num_aspect_ratios = _num_aspect_ratios(ratios)

    for level in final_levels:
        if level in original_set:
            continue
        _derive_params_for_added_level(
            level,
            original_set,
            ratios,
            sizes,
            by_level,
            num_aspect_ratios=num_aspect_ratios,
            scales=scales,
        )

    provenance: Dict[str, LevelSource] = {
        level: ("original" if level in original_set else "added") for level in final_levels
    }
    finalized = FinalizedFpnLevels(
        requested_fpn_levels=tuple(requested),
        used_fpn_levels=tuple(final_levels),
        used_layers=final_layers,
        level_provenance=provenance,
    )
    return finalized, ratios, sizes


def _build_anchor_config(
    spec: AnchorTrainingSpec,
    finalized: FinalizedFpnLevels,
    level_aspect_ratios: Dict[str, List[float]],
    level_base_sizes: Dict[str, List[float]],
) -> Dict[str, Any]:
    used_specs = [entry for entry in spec.fpn_specs if entry["level"] in finalized.used_fpn_levels]
    return {
        "config_hash": spec.config_hash,
        "assignment_method": spec.method,
        "method_parameters": spec.method_parameters,
        "requested_fpn_levels": list(finalized.requested_fpn_levels),
        "used_fpn_levels": list(finalized.used_fpn_levels),
        "level_provenance": dict(finalized.level_provenance),
        "used_layers": list(finalized.used_layers),
        "fpn_specs": spec.fpn_specs,
        "used_base_sizes": [entry["base_size"] for entry in used_specs],
        "coefficient": spec.fpn_coefficient,
        "scales": spec.scales,
        "level_aspect_ratios": level_aspect_ratios,
        "level_base_sizes": level_base_sizes,
        "seed": spec.seed,
        "box_count": spec.box_count,
    }


def _anchor_generator_from_config(config: Dict[str, Any]) -> AnchorGenerator:
    used_levels = config["used_fpn_levels"]
    level_ratios_map = config["level_aspect_ratios"]
    level_base_sizes_map = config["level_base_sizes"]
    assignment_method = config["assignment_method"]
    scales = config["scales"]
    coefficient = config["coefficient"]
    fpn_specs = {spec["level"]: spec for spec in config["fpn_specs"]}

    anchor_sizes: List[tuple[float, ...]] = []
    aspect_ratios: List[tuple[float, ...]] = []

    for level in used_levels:
        ratios = level_ratios_map[level]
        aspect_ratios.append(tuple(ratios))

        if assignment_method == "dimension_based" and level in level_base_sizes_map:
            level_sizes = tuple(level_base_sizes_map[level])
        elif level in level_base_sizes_map and level_base_sizes_map[level]:
            level_sizes = tuple(level_base_sizes_map[level])
        else:
            spec = fpn_specs.get(level)
            if not spec:
                base_size = RETINANET_FPN_LEVEL_STRIDES.get(level, 8) * coefficient
            else:
                base_size = spec["base_size"]
            level_sizes = tuple(base_size * scale for scale in scales)
        anchor_sizes.append(level_sizes)

    return AnchorGenerator(tuple(anchor_sizes), tuple(aspect_ratios))


def build_anchor_generator(spec: AnchorTrainingSpec) -> AnchorGenerator:
    """
    Finalize FPN levels, persist ``anchor_config.json``, and build ``AnchorGenerator``.

    Level transformation is delegated to ``retinanet_levels``; anchor-parameter
    derivation for added levels is handled in this module.
    """
    validate_method_parameters(spec.method, spec.method_parameters)

    finalized, ratios, sizes = _finalize_fpn_levels(
        spec.requested_fpn_levels,
        spec.level_aspect_ratios,
        spec.level_base_sizes,
        spec.fpn_specs,
        scales=spec.scales,
    )
    config = _build_anchor_config(spec, finalized, ratios, sizes)
    write_json_file(spec.config_path, config)
    logger.info("RetinaNet anchor config saved to %s", spec.config_path)
    return _anchor_generator_from_config(config)


def load_anchor_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a finalized ``anchor_config.json``."""
    return read_json_file(Path(config_path))


def anchor_spec_from_path(config_path: str | Path) -> AnchorTrainingSpec:
    """Build :class:`AnchorTrainingSpec` from a finalized on-disk config."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"anchor config not found: {path}")
    config = load_anchor_config(path)
    validate_method_parameters(config["assignment_method"], config["method_parameters"])
    return AnchorTrainingSpec.from_config(config, path)


def compute_and_save_retinanet_anchors(
    box_dimensions: Sequence[BoxDimensions],
    anchor_config_path: Path,
    *,
    method: str,
    method_parameters: Dict[str, Any],
    force: bool = False,
) -> AnchorTrainingSpec:
    """
    Cluster anchors from GT box sizes, finalize levels, write config, return spec.

    When the config already exists and ``force`` is False, loads the existing spec
    without re-clustering.
    """
    anchor_config_path = Path(anchor_config_path)
    if not force and anchor_config_path.is_file():
        logger.info("Reusing anchor config: %s", anchor_config_path)
        return anchor_spec_from_path(anchor_config_path)

    logger.info(
        "Computing RetinaNet anchors: %d boxes, method=%s -> %s",
        len(box_dimensions),
        method,
        anchor_config_path,
    )
    result = compute_optimized_anchors(
        box_dimensions,
        method=method,
        method_parameters=method_parameters,
    )
    spec = AnchorTrainingSpec.from_optimization(result, anchor_config_path)
    build_anchor_generator(spec)
    return anchor_spec_from_path(anchor_config_path)


def save_anchor_spec_to_run_dir(spec: AnchorTrainingSpec, run_dir: Path) -> None:
    """Copy anchor config into a training run directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_file(run_dir / "anchor_training_spec.json", spec.to_dict(), indent=2)
    dest = run_dir / ANCHOR_CONFIG_FILENAME
    if not dest.exists():
        shutil.copy2(spec.config_path, dest)


def with_finalized_levels(spec: AnchorTrainingSpec) -> AnchorTrainingSpec:
    """Return *spec* populated with ``used_fpn_levels`` / provenance from disk."""
    config = load_anchor_config(spec.config_path)
    return replace(
        spec,
        used_fpn_levels=list(config["used_fpn_levels"]),
        level_provenance=dict(config.get("level_provenance", {})),
    )
