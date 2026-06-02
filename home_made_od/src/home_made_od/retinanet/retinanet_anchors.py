"""
RetinaNet anchor optimization, config I/O, and torchvision AnchorGenerator wiring.

Clustering is detector-agnostic (``anchor_computation_strategies``). This module
applies RetinaNet FPN constraints and reads/writes a single ``anchor_config.json``
with per-level sizes and aspect ratios — no per-box assignment files.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from torchvision.models.detection.anchor_utils import AnchorGenerator

from home_made_od.anchors.anchor_computation_strategies import (
    AnchorOptimizationResult,
    BoxDimensions,
    FPN_LEVEL_ORDER,
    compute_optimized_anchors,
    validate_method_parameters,
)
from home_made_od.anchors.anchor_config_registry import resolve_anchor_config_hash

logger = logging.getLogger(__name__)

ANCHOR_CONFIG_FILENAME = "anchor_config.json"
# Backward-compatible alias for older call sites.
ANCHOR_MANIFEST_FILENAME = ANCHOR_CONFIG_FILENAME

RETINANET_FPN_LEVEL_ORDER: Tuple[str, ...] = FPN_LEVEL_ORDER
RETINANET_EARLY_FPN_LEVELS: Tuple[str, ...] = ("P2", "P3", "P4", "P5")
RETINANET_FPN_LEVEL_STRIDES: Dict[str, int] = {
    "P2": 4,
    "P3": 8,
    "P4": 16,
    "P5": 32,
    "P6": 64,
    "P7": 128,
}
RESNET_LEVEL_MAP: Dict[str, int] = {"P2": 1, "P3": 2, "P4": 3, "P5": 4}
RESNET_OUT_CHANNELS: Dict[int, int] = {1: 256, 2: 512, 3: 1024, 4: 2048}


@dataclass(frozen=True)
class AnchorTrainingSpec:
    """Resolved per-level anchor config for training or inference."""

    method: str
    method_parameters: Dict[str, Any]
    config_hash: str
    config_path: Path
    used_fpn_levels: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "method_parameters": self.method_parameters,
            "config_hash": self.config_hash,
            "config_path": str(self.config_path),
            "used_fpn_levels": self.used_fpn_levels,
        }

    @property
    def manifest_path(self) -> Path:
        """Deprecated alias for ``config_path``."""
        return self.config_path


def _sort_fpn_levels(levels: Iterable[str]) -> List[str]:
    order_index = {name: index for index, name in enumerate(RETINANET_FPN_LEVEL_ORDER)}
    return sorted(levels, key=lambda level: order_index.get(level, len(RETINANET_FPN_LEVEL_ORDER)))


def _spec_by_level(fpn_specs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {spec["level"]: spec for spec in fpn_specs}


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


def _ensure_p2_fpn_spec(fpn_specs: List[Dict[str, Any]]) -> None:
    """Insert a P2 spec when legacy manifests only list P3–P7."""
    if any(spec["level"] == "P2" for spec in fpn_specs):
        return

    reference = next((spec for spec in fpn_specs if spec["level"] == "P3"), fpn_specs[0])
    coefficient = reference["base_size"] // reference["stride"]
    base_size = 4 * coefficient
    fpn_specs.insert(
        0,
        {
            "level": "P2",
            "stride": 4,
            "base_size": base_size,
            "nominal_area": base_size ** 2,
        },
    )


def retinanet_returned_layers(used_levels: List[str]) -> List[int]:
    """
    Map manifest FPN levels to ResNet ``returned_layers`` indices.

    Single source of truth shared with ``retinanet_detector._build_fpn_backbone``.
    """
    returned = [RESNET_LEVEL_MAP[level] for level in used_levels if level in RESNET_LEVEL_MAP]
    if not returned:
        logger.warning(
            "No mappable FPN levels in %s; defaulting returned_layers to [4] (P5).",
            used_levels,
        )
        return [4]

    if 2 in returned and 1 not in returned:
        returned.append(1)

    if ("P6" in used_levels or "P7" in used_levels) and 4 not in returned:
        returned.append(4)

    return sorted(set(returned))


def validate_retinanet_fpn_levels(used_levels: List[str]) -> None:
    """Raise if manifest FPN levels violate RetinaNet backbone / anchor alignment rules."""
    has_p6 = "P6" in used_levels
    has_p7 = "P7" in used_levels
    if has_p6 != has_p7:
        raise ValueError(
            f"RetinaNet requires P6 and P7 together; got used_fpn_levels={used_levels}. "
            "Re-run anchor optimization or load a manifest normalized with "
            "normalize_manifest_fpn_levels."
        )

    if "P3" in used_levels and "P2" not in used_levels:
        raise ValueError(
            f"RetinaNet requires P2 whenever P3 is used; got used_fpn_levels={used_levels}. "
            "Re-run anchor optimization or load a manifest normalized with "
            "normalize_manifest_fpn_levels."
        )

    if (has_p6 or has_p7) and "P5" not in used_levels:
        raise ValueError(
            f"RetinaNet requires P5 whenever P6/P7 are used; got used_fpn_levels={used_levels}. "
            "LastLevelP6P7 branches from the stride-32 backbone map."
        )

    early_present = [level for level in RETINANET_EARLY_FPN_LEVELS if level in used_levels]
    if len(early_present) >= 2:
        first_index = RETINANET_EARLY_FPN_LEVELS.index(early_present[0])
        last_index = RETINANET_EARLY_FPN_LEVELS.index(early_present[-1])
        expected = set(RETINANET_EARLY_FPN_LEVELS[first_index : last_index + 1])
        missing = expected - set(used_levels)
        if missing:
            raise ValueError(
                f"RetinaNet requires contiguous P2–P5 levels; missing {sorted(missing)} "
                f"for used_fpn_levels={used_levels}."
            )


def _couple_p6_p7_levels(
    used_fpn_levels: List[str],
    level_aspect_ratios: Dict[str, List[float]],
    level_base_sizes: Dict[str, List[float]],
    fpn_specs: List[Dict[str, Any]],
    num_aspect_ratios: int,
    scales: List[float],
) -> List[str]:
    used_set = set(used_fpn_levels)
    has_p6 = "P6" in used_set
    has_p7 = "P7" in used_set

    if has_p6 == has_p7:
        return _sort_fpn_levels(used_set)

    by_level = _spec_by_level(fpn_specs)

    if has_p6 and not has_p7:
        logger.warning(
            "P6 present without P7 — adding P7 anchors (required by LastLevelP6P7)."
        )
        if "P6" not in level_aspect_ratios:
            level_aspect_ratios["P6"] = _bootstrap_aspect_ratios(num_aspect_ratios)
            level_base_sizes["P6"] = _bootstrap_base_sizes_from_spec("P6", by_level, scales)
        _derive_level_anchors_from_reference(
            "P7", "P6", level_aspect_ratios, level_base_sizes, by_level
        )
        used_set.add("P7")
    else:
        logger.warning(
            "P7 present without P6 — adding P6 anchors (required by LastLevelP6P7)."
        )
        if "P7" not in level_aspect_ratios:
            level_aspect_ratios["P7"] = _bootstrap_aspect_ratios(num_aspect_ratios)
            level_base_sizes["P7"] = _bootstrap_base_sizes_from_spec("P7", by_level, scales)
        _derive_level_anchors_from_reference(
            "P6", "P7", level_aspect_ratios, level_base_sizes, by_level
        )
        used_set.add("P6")

    return _sort_fpn_levels(used_set)


def _couple_p5_with_p6_p7(
    used_fpn_levels: List[str],
    level_aspect_ratios: Dict[str, List[float]],
    level_base_sizes: Dict[str, List[float]],
    fpn_specs: List[Dict[str, Any]],
    num_aspect_ratios: int,
    scales: List[float],
) -> List[str]:
    used_set = set(used_fpn_levels)
    if not ({"P6", "P7"} & used_set):
        return _sort_fpn_levels(used_set)
    if "P5" in used_set:
        return _sort_fpn_levels(used_set)

    by_level = _spec_by_level(fpn_specs)
    reference_level = next(
        (level for level in reversed(RETINANET_EARLY_FPN_LEVELS) if level in used_set),
        None,
    )
    if reference_level is None:
        reference_level = "P4" if "P4" in by_level else "P3"

    logger.warning(
        "P6/P7 present without P5 — adding P5 anchors (required by LastLevelP6P7 backbone path)."
    )
    if reference_level not in level_aspect_ratios:
        level_aspect_ratios[reference_level] = _bootstrap_aspect_ratios(num_aspect_ratios)
        level_base_sizes[reference_level] = _bootstrap_base_sizes_from_spec(
            reference_level, by_level, scales
        )
    _derive_level_anchors_from_reference(
        "P5", reference_level, level_aspect_ratios, level_base_sizes, by_level
    )
    used_set.add("P5")
    return _sort_fpn_levels(used_set)


def _couple_p2_with_p3(
    used_fpn_levels: List[str],
    level_aspect_ratios: Dict[str, List[float]],
    level_base_sizes: Dict[str, List[float]],
    fpn_specs: List[Dict[str, Any]],
    num_aspect_ratios: int,
    scales: List[float],
) -> List[str]:
    used_set = set(used_fpn_levels)
    if "P3" not in used_set or "P2" in used_set:
        return _sort_fpn_levels(used_set)

    _ensure_p2_fpn_spec(fpn_specs)
    by_level = _spec_by_level(fpn_specs)

    logger.warning(
        "P3 present without P2 — adding P2 anchors (required when P3 is in returned_layers)."
    )
    if "P3" not in level_aspect_ratios:
        level_aspect_ratios["P3"] = _bootstrap_aspect_ratios(num_aspect_ratios)
        level_base_sizes["P3"] = _bootstrap_base_sizes_from_spec("P3", by_level, scales)
    _derive_level_anchors_from_reference(
        "P2", "P3", level_aspect_ratios, level_base_sizes, by_level
    )
    used_set.add("P2")
    return _sort_fpn_levels(used_set)


def _nearest_level_with_anchors(
    target_level: str,
    candidate_levels: Iterable[str],
    level_aspect_ratios: Dict[str, List[float]],
) -> Optional[str]:
    order_index = {name: index for index, name in enumerate(RETINANET_EARLY_FPN_LEVELS)}
    target_index = order_index[target_level]
    available = [
        level
        for level in candidate_levels
        if level in level_aspect_ratios and level in order_index
    ]
    if not available:
        return None
    return min(available, key=lambda level: abs(order_index[level] - target_index))


def _ensure_contiguous_early_levels(
    used_fpn_levels: List[str],
    level_aspect_ratios: Dict[str, List[float]],
    level_base_sizes: Dict[str, List[float]],
    fpn_specs: List[Dict[str, Any]],
    num_aspect_ratios: int,
    scales: List[float],
) -> List[str]:
    used_set = set(used_fpn_levels)
    early_present = [level for level in RETINANET_EARLY_FPN_LEVELS if level in used_set]
    if len(early_present) < 2:
        return _sort_fpn_levels(used_set)

    by_level = _spec_by_level(fpn_specs)
    first_index = RETINANET_EARLY_FPN_LEVELS.index(early_present[0])
    last_index = RETINANET_EARLY_FPN_LEVELS.index(early_present[-1])

    for index in range(first_index, last_index + 1):
        level = RETINANET_EARLY_FPN_LEVELS[index]
        if level in used_set:
            continue

        reference_level = _nearest_level_with_anchors(level, used_set, level_aspect_ratios)
        if reference_level is None:
            level_aspect_ratios[level] = _bootstrap_aspect_ratios(num_aspect_ratios)
            level_base_sizes[level] = _bootstrap_base_sizes_from_spec(level, by_level, scales)
        else:
            logger.warning(
                "Non-contiguous early FPN levels — synthesizing %s anchors from %s.",
                level,
                reference_level,
            )
            _derive_level_anchors_from_reference(
                level, reference_level, level_aspect_ratios, level_base_sizes, by_level
            )
        used_set.add(level)

    return _sort_fpn_levels(used_set)


def _ensure_early_fpn_level(
    used_fpn_levels: List[str],
    level_aspect_ratios: Dict[str, List[float]],
    level_base_sizes: Dict[str, List[float]],
    fpn_specs: List[Dict[str, Any]],
    num_aspect_ratios: int,
    scales: List[float],
) -> List[str]:
    if any(level in used_fpn_levels for level in RETINANET_EARLY_FPN_LEVELS):
        return used_fpn_levels

    fallback_level = "P5"
    logger.warning(
        "No early FPN levels %s in optimized set; prepending %s for RetinaNet init.",
        list(RETINANET_EARLY_FPN_LEVELS),
        fallback_level,
    )

    by_level = _spec_by_level(fpn_specs)
    if fallback_level not in level_aspect_ratios:
        level_aspect_ratios[fallback_level] = _bootstrap_aspect_ratios(num_aspect_ratios)
    if fallback_level not in level_base_sizes:
        level_base_sizes[fallback_level] = _bootstrap_base_sizes_from_spec(
            fallback_level, by_level, scales
        )

    used = [fallback_level] + [
        level for level in used_fpn_levels if level != fallback_level
    ]
    return _sort_fpn_levels(used)


def finalize_retinanet_used_fpn_levels(
    used_fpn_levels: List[str],
    level_aspect_ratios: Dict[str, List[float]],
    level_base_sizes: Dict[str, List[float]],
    fpn_specs: List[Dict[str, Any]],
    *,
    num_aspect_ratios: int,
    scales: List[float],
) -> List[str]:
    """
    Apply RetinaNet FPN constraints after per-level clustering.

    1. **P6/P7 coupling** — both levels and anchor dict entries.
    2. **P5 when P6/P7** — backbone ``LastLevelP6P7`` always reads ``layer4``.
    3. **P2 when P3** — torchvision FPN cannot return ``layer2`` without ``layer1``.
    4. **Contiguous P2–P5** — gaps break FPN lateral wiring and head alignment.
    5. **Early fallback** — at least one of P2–P5 when the set would otherwise be empty.
    """
    used = _couple_p6_p7_levels(
        used_fpn_levels,
        level_aspect_ratios,
        level_base_sizes,
        fpn_specs,
        num_aspect_ratios,
        scales,
    )
    used = _couple_p5_with_p6_p7(
        used,
        level_aspect_ratios,
        level_base_sizes,
        fpn_specs,
        num_aspect_ratios,
        scales,
    )
    used = _couple_p2_with_p3(
        used,
        level_aspect_ratios,
        level_base_sizes,
        fpn_specs,
        num_aspect_ratios,
        scales,
    )
    used = _ensure_contiguous_early_levels(
        used,
        level_aspect_ratios,
        level_base_sizes,
        fpn_specs,
        num_aspect_ratios,
        scales,
    )
    return _ensure_early_fpn_level(
        used,
        level_aspect_ratios,
        level_base_sizes,
        fpn_specs,
        num_aspect_ratios,
        scales,
    )


def _num_aspect_ratios(level_aspect_ratios: Dict[str, List[float]]) -> int:
    return len(next(iter(level_aspect_ratios.values())))


def _finalize_anchor_dicts(
    used_fpn_levels: List[str],
    level_aspect_ratios: Dict[str, List[float]],
    level_base_sizes: Dict[str, List[float]],
    fpn_specs: List[Dict[str, Any]],
    *,
    num_aspect_ratios: int,
    scales: List[float],
) -> List[str]:
    return finalize_retinanet_used_fpn_levels(
        used_fpn_levels,
        level_aspect_ratios,
        level_base_sizes,
        fpn_specs,
        num_aspect_ratios=num_aspect_ratios,
        scales=scales,
    )


def build_retinanet_anchor_config(result: AnchorOptimizationResult) -> Dict[str, Any]:
    """Apply RetinaNet FPN finalization and build the JSON-serializable anchor config."""
    fpn_specs = list(result.fpn_specs)
    level_aspect_ratios = dict(result.level_aspect_ratios)
    level_base_sizes = dict(result.level_base_sizes)
    num_aspect_ratios = _num_aspect_ratios(level_aspect_ratios)

    used_fpn_levels = _finalize_anchor_dicts(
        result.used_fpn_levels,
        level_aspect_ratios,
        level_base_sizes,
        fpn_specs,
        num_aspect_ratios=num_aspect_ratios,
        scales=result.scales,
    )
    validate_retinanet_fpn_levels(used_fpn_levels)

    used_specs = [spec for spec in fpn_specs if spec["level"] in used_fpn_levels]

    return {
        "config_hash": result.config_hash,
        "assignment_method": result.method,
        "fpn_specs": fpn_specs,
        "used_fpn_levels": used_fpn_levels,
        "used_base_sizes": [spec["base_size"] for spec in used_specs],
        "coefficient": result.fpn_coefficient,
        "scales": result.scales,
        "level_aspect_ratios": level_aspect_ratios,
        "level_base_sizes": level_base_sizes,
        "method_parameters": result.method_parameters,
        "seed": result.seed,
        "box_count": result.box_count,
    }


def write_anchor_config(config: Dict[str, Any], config_path: Path) -> Path:
    """Write ``anchor_config.json`` (per-level anchors only)."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4, ensure_ascii=False)
    logger.info("RetinaNet anchor config saved to %s", config_path)
    return config_path


def finalize_retinanet_anchor_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Re-apply RetinaNet FPN finalization on a loaded config (mutates and returns it).
    """
    _ensure_p2_fpn_spec(config["fpn_specs"])
    ratios = config["level_aspect_ratios"]
    config["used_fpn_levels"] = _finalize_anchor_dicts(
        config["used_fpn_levels"],
        ratios,
        config["level_base_sizes"],
        config["fpn_specs"],
        num_aspect_ratios=_num_aspect_ratios(ratios),
        scales=config["scales"],
    )
    return config


def load_anchor_config(config_path: str | Path) -> Dict[str, Any]:
    """Load ``anchor_config.json`` and apply RetinaNet FPN finalization."""
    path = Path(config_path)
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    # Accept legacy wrapper ``{"metadata": {...}}`` from older pipelines.
    if "used_fpn_levels" not in data and "metadata" in data:
        data = data["metadata"]
    return finalize_retinanet_anchor_config(data)


def anchor_spec_from_path(config_path: str | Path) -> AnchorTrainingSpec:
    """Build :class:`AnchorTrainingSpec` from ``anchor_config.json``."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"anchor config not found: {path}")
    config = load_anchor_config(path)
    method = config["assignment_method"]
    method_parameters = config["method_parameters"]
    validate_method_parameters(method, method_parameters)
    validate_retinanet_fpn_levels(config["used_fpn_levels"])
    return AnchorTrainingSpec(
        method=method,
        method_parameters=dict(method_parameters),
        config_hash=config["config_hash"],
        config_path=path.resolve(),
        used_fpn_levels=list(config["used_fpn_levels"]),
    )


def compute_and_save_retinanet_anchors(
    box_dimensions: Sequence[BoxDimensions],
    anchor_config_path: Path,
    *,
    method: str,
    method_parameters: Dict[str, Any],
    force: bool = False,
) -> AnchorTrainingSpec:
    """
    Cluster anchors from GT box sizes and write a single ``anchor_config.json``.

    The training script is responsible for building ``box_dimensions`` (e.g. from YOLO
    labels at the resize used for training).
    """
    anchor_config_path = Path(anchor_config_path)
    if force or not anchor_config_path.is_file():
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
        write_anchor_config(build_retinanet_anchor_config(result), anchor_config_path)
    else:
        logger.info("Reusing anchor config: %s", anchor_config_path)
    return anchor_spec_from_path(anchor_config_path)


def get_anchor_config_hash(config: Dict[str, Any]) -> str:
    """Return the registered anchor ``config_hash`` referenced by *config*."""
    return resolve_anchor_config_hash(config)


def get_anchor_generator_config(config_path: str | Path) -> Dict[str, Any]:
    """Deprecated alias for :func:`load_anchor_config`."""
    return load_anchor_config(config_path)


def build_retinanet_manifest_metadata(result: AnchorOptimizationResult) -> Dict[str, Any]:
    """Deprecated alias for :func:`build_retinanet_anchor_config`."""
    return build_retinanet_anchor_config(result)


def normalize_manifest_fpn_levels(config: Dict[str, Any]) -> Dict[str, Any]:
    """Deprecated alias for :func:`finalize_retinanet_anchor_config`."""
    return finalize_retinanet_anchor_config(config)


def anchor_spec_from_manifest_path(config_path: str | Path) -> AnchorTrainingSpec:
    """Deprecated alias for :func:`anchor_spec_from_path`."""
    return anchor_spec_from_path(config_path)


def build_anchor_generator_from_config(config_path: str | Path) -> AnchorGenerator:
    """Build a torchvision ``AnchorGenerator`` from ``anchor_config.json``."""
    config = load_anchor_config(config_path)
    used_levels = config["used_fpn_levels"]
    validate_retinanet_fpn_levels(used_levels)
    level_ratios_map = config["level_aspect_ratios"]
    level_base_sizes_map = config.get("level_base_sizes", {})
    assignment_method = config["assignment_method"]
    scales = config["scales"]
    coefficient = config["coefficient"]
    fpn_specs = {spec["level"]: spec for spec in config["fpn_specs"]}

    anchor_sizes = []
    aspect_ratios = []

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


def build_anchor_generator_from_manifest(config_path: str | Path) -> AnchorGenerator:
    """Deprecated alias for :func:`build_anchor_generator_from_config`."""
    return build_anchor_generator_from_config(config_path)


def save_anchor_spec_to_artifact_dir(spec: AnchorTrainingSpec, artifact_dir: Path) -> None:
    """Deprecated alias for :func:`save_anchor_spec_to_run_dir`."""
    save_anchor_spec_to_run_dir(spec, artifact_dir)


def save_anchor_spec_to_run_dir(spec: AnchorTrainingSpec, run_dir: Path) -> None:
    """Copy anchor config into a training run directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "anchor_training_spec.json", "w", encoding="utf-8") as file:
        json.dump(spec.to_dict(), file, indent=2, ensure_ascii=False)
    dest = run_dir / ANCHOR_CONFIG_FILENAME
    if not dest.exists():
        shutil.copy2(spec.config_path, dest)


def build_anchor_generator_from_spec(
    anchor_spec: AnchorTrainingSpec,
) -> AnchorGenerator:
    """Build ``AnchorGenerator`` from a resolved :class:`AnchorTrainingSpec`."""
    return build_anchor_generator_from_config(anchor_spec.config_path)
