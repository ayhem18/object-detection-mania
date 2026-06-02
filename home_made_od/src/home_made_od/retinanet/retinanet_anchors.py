"""
Anchor manifests, optimization resolution, and torchvision AnchorGenerator wiring.

RetinaNet FPN / backbone coupling
---------------------------------
Anchor clustering (in ``anchor_computation_strategies``) is detector-agnostic.
This module applies RetinaNet-specific rules so ``used_fpn_levels`` stays aligned
with :func:`retinanet_returned_layers` and the Etalon detector build path.

See also ``retinanet_detector`` and ``tests/retinanet/test_retinanet_building.py``.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from torchvision.models.detection.anchor_utils import AnchorGenerator

from home_made_od.anchors.anchor_computation_strategies import (
    AnchorOptimizationResult,
    FPN_LEVEL_ORDER,
    compute_optimized_anchors,
    validate_method_parameters,
)
from home_made_od.anchors.anchor_config_registry import (
    load_registered_anchor_config,
    resolve_anchor_config_hash,
)
from home_made_od.ds_utils import WeldingDetectionDataset
from home_made_od.path_layout import (
    ANCHOR_MANIFEST_FILENAME,
    anchors_data_dir,
    anchors_data_manifest_path,
    iter_anchor_artifact_dirs,
    iter_split_artifact_dirs,
    split_artifacts_root,
)

logger = logging.getLogger(__name__)

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
    """Resolved anchor manifest for training or inference."""

    method: str
    method_parameters: Dict[str, Any]
    config_hash: str
    manifest_path: Path
    config_dir: Path
    used_fpn_levels: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "method_parameters": self.method_parameters,
            "config_hash": self.config_hash,
            "manifest_path": str(self.manifest_path),
            "config_dir": str(self.config_dir),
            "used_fpn_levels": self.used_fpn_levels,
        }


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


def build_retinanet_manifest_metadata(result: AnchorOptimizationResult) -> Dict[str, Any]:
    """Apply RetinaNet FPN finalization and assemble on-disk manifest metadata."""
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
    target_y_dim, target_x_dim = result.img_size

    return {
        "dataset_hash": result.dataset_hash,
        "config_hash": result.config_hash,
        "img_size": [target_y_dim, target_x_dim],
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
    }


def write_anchor_artifacts(
    output_dir: Path,
    metadata: Dict[str, Any],
    enriched_samples: List[Dict[str, Any]],
) -> Tuple[Path, Path]:
    """Persist ``anchor_config.json`` and ``master_labels_enriched.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_json_path = output_dir / "master_labels_enriched.json"
    anchor_config_path = output_dir / ANCHOR_MANIFEST_FILENAME

    with open(enriched_json_path, "w", encoding="utf-8") as file:
        json.dump(
            {"metadata": metadata, "samples": enriched_samples},
            file,
            indent=4,
            ensure_ascii=False,
        )

    with open(anchor_config_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    logger.info("RetinaNet anchor manifest saved to %s", anchor_config_path)
    return enriched_json_path, anchor_config_path


def normalize_manifest_fpn_levels(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Re-apply RetinaNet FPN finalization on a stored manifest (legacy migration).

    Mutates ``metadata`` in place and returns it.
    """
    _ensure_p2_fpn_spec(metadata["fpn_specs"])
    ratios = metadata["level_aspect_ratios"]
    metadata["used_fpn_levels"] = _finalize_anchor_dicts(
        metadata["used_fpn_levels"],
        ratios,
        metadata["level_base_sizes"],
        metadata["fpn_specs"],
        num_aspect_ratios=_num_aspect_ratios(ratios),
        scales=metadata["scales"],
    )
    return metadata


def _upgrade_legacy_manifest_if_needed(manifest_path: Path) -> None:
    """Rewrite on-disk manifests that predate RetinaNet FPN finalization."""
    with open(manifest_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    original_levels = list(metadata.get("used_fpn_levels", []))
    normalize_manifest_fpn_levels(metadata)
    if metadata["used_fpn_levels"] == original_levels:
        return

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    enriched_path = manifest_path.parent / "master_labels_enriched.json"
    if enriched_path.is_file():
        with open(enriched_path, "r", encoding="utf-8") as file:
            enriched = json.load(file)
        enriched["metadata"] = metadata
        with open(enriched_path, "w", encoding="utf-8") as file:
            json.dump(enriched, file, indent=4, ensure_ascii=False)


def optimize_retinanet_anchors(
    dataset: WeldingDetectionDataset,
    output_dir: Path,
    dataset_hash: str,
    method: str,
    method_parameters: Dict[str, Any],
) -> Tuple[Path, Path]:
    """Cluster anchors, finalize for RetinaNet, and write manifest artifacts once."""
    result = compute_optimized_anchors(
        dataset=dataset,
        dataset_hash=dataset_hash,
        method=method,
        method_parameters=method_parameters,
    )
    metadata = build_retinanet_manifest_metadata(result)
    return write_anchor_artifacts(output_dir, metadata, result.enriched_samples)


def get_anchor_config_hash(config: Dict[str, Any]) -> str:
    """Return the registered anchor ``config_hash`` referenced by *config*."""
    return resolve_anchor_config_hash(config)


def get_anchor_generator_config(manifest_path: str | Path) -> Dict[str, Any]:
    """Load manifest metadata from split-dependent ``anchor_config.json``."""
    with open(manifest_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if "metadata" in data:
        metadata = data["metadata"]
    else:
        metadata = data

    return normalize_manifest_fpn_levels(metadata)


def _spec_from_manifest(manifest_path: Path, config_dir: Path) -> AnchorTrainingSpec:
    metadata = get_anchor_generator_config(manifest_path)
    method = metadata["assignment_method"]
    method_parameters = metadata["method_parameters"]
    validate_method_parameters(method, method_parameters)
    validate_retinanet_fpn_levels(metadata["used_fpn_levels"])
    return AnchorTrainingSpec(
        method=method,
        method_parameters=dict(method_parameters),
        config_hash=metadata["config_hash"],
        manifest_path=manifest_path,
        config_dir=config_dir,
        used_fpn_levels=list(metadata["used_fpn_levels"]),
    )


def ensure_anchor_training_spec(
    config: Dict[str, Any],
    *,
    dataset: WeldingDetectionDataset,
    dataset_hash: str,
    split_hash: str,
    force_reoptimize: bool,
) -> AnchorTrainingSpec:
    """
    Resolve or create split-dependent anchors under
    ``{split_hash}/{anchor_config_hash}/anchors_data/``.

    Requires a registered recipe in ``labeling/anchor_configs/{config_hash}.json``.
    """
    anchor_config_hash = resolve_anchor_config_hash(config)
    recipe = load_registered_anchor_config(anchor_config_hash)
    method = recipe["method"]
    method_parameters = recipe["method_parameters"]

    config_dir = anchors_data_dir(dataset_hash, split_hash, anchor_config_hash)
    manifest_path = config_dir / ANCHOR_MANIFEST_FILENAME

    if force_reoptimize or not manifest_path.is_file():
        logger.info(
            "Optimizing anchors: method=%s, config_hash=%s, dir=%s",
            method,
            anchor_config_hash,
            config_dir,
        )
        optimize_retinanet_anchors(
            dataset=dataset,
            output_dir=config_dir,
            dataset_hash=dataset_hash,
            method=method,
            method_parameters=method_parameters,
        )
    else:
        logger.info("Reusing anchors_data manifest: %s", manifest_path)
        _upgrade_legacy_manifest_if_needed(manifest_path)

    return _spec_from_manifest(manifest_path, config_dir)


def save_anchor_spec_to_artifact_dir(spec: AnchorTrainingSpec, artifact_dir: Path) -> None:
    """Copy anchor manifest metadata into a training run directory."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(artifact_dir / "anchor_training_spec.json", "w", encoding="utf-8") as file:
        json.dump(spec.to_dict(), file, indent=2, ensure_ascii=False)
    dest = artifact_dir / ANCHOR_MANIFEST_FILENAME
    if not dest.exists():
        shutil.copy2(spec.manifest_path, dest)


def _iter_anchor_manifest_paths(dataset_hash: str) -> Iterator[Path]:
    """Yield ``anchor_config.json`` paths under each anchor-config artifact root."""
    for split_dir in iter_split_artifact_dirs(dataset_hash):
        for anchor_dir in iter_anchor_artifact_dirs(dataset_hash, split_dir.name):
            manifest = anchor_dir / "anchors_data" / ANCHOR_MANIFEST_FILENAME
            if manifest.is_file():
                yield manifest


def resolve_split_anchor_config(
    dataset_hash: str,
    config_hash: Optional[str] = None,
    split_hash: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Resolve a split-dependent anchor manifest.

    Returns ``(anchor_config.json path, anchors_data directory)``.
    """
    if config_hash is not None:
        load_registered_anchor_config(config_hash)
        if split_hash is not None:
            cfg_path = anchors_data_manifest_path(dataset_hash, split_hash, config_hash)
            if not cfg_path.is_file():
                raise FileNotFoundError(
                    f"No anchors_data manifest at {cfg_path.parent}. "
                    f"Run anchors_matching_sanity_check.py or training with "
                    f"force_reoptimize_anchors=True for dataset={dataset_hash}, "
                    f"split={split_hash}, config={config_hash}."
                )
            return cfg_path, cfg_path.parent

        matches = [
            path
            for path in _iter_anchor_manifest_paths(dataset_hash)
            if path.parent.parent.name == config_hash
        ]
        if not matches:
            raise FileNotFoundError(
                f"No anchors_data manifest for config_hash={config_hash} "
                f"under dataset {dataset_hash}."
            )
        cfg_path = max(matches, key=lambda path: path.stat().st_mtime)
        return cfg_path, cfg_path.parent

    if split_hash is not None:
        candidates = [
            anchors_data_manifest_path(dataset_hash, split_hash, directory.name)
            for directory in iter_anchor_artifact_dirs(dataset_hash, split_hash)
            if anchors_data_manifest_path(dataset_hash, split_hash, directory.name).is_file()
        ]
        if candidates:
            cfg_path = max(candidates, key=lambda path: path.stat().st_mtime)
            return cfg_path, cfg_path.parent

        raise FileNotFoundError(
            f"No anchors_data manifests under split {split_hash} "
            f"({split_artifacts_root(dataset_hash, split_hash)})."
        )

    candidates = list(_iter_anchor_manifest_paths(dataset_hash))
    if not candidates:
        raise FileNotFoundError(
            f"No anchors_data manifests found for dataset {dataset_hash}. "
            "Run anchors_matching_sanity_check.py first."
        )

    cfg_path = max(candidates, key=lambda path: path.stat().st_mtime)
    return cfg_path, cfg_path.parent


def anchor_spec_from_manifest_path(manifest_path: str | Path) -> AnchorTrainingSpec:
    """Build :class:`AnchorTrainingSpec` from an on-disk anchor manifest."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"anchor manifest not found: {path}")
    return _spec_from_manifest(path, path.parent)


def build_anchor_generator_from_manifest(
    manifest_path: str | Path,
) -> AnchorGenerator:
    """
    Build a torchvision ``AnchorGenerator`` from a split-dependent anchor manifest.
    """
    metadata = get_anchor_generator_config(manifest_path)
    used_levels = metadata["used_fpn_levels"]
    validate_retinanet_fpn_levels(used_levels)
    level_ratios_map = metadata["level_aspect_ratios"]
    level_base_sizes_map = metadata.get("level_base_sizes", {})
    assignment_method = metadata["assignment_method"]
    scales = metadata["scales"]
    coefficient = metadata["coefficient"]
    fpn_specs = {spec["level"]: spec for spec in metadata["fpn_specs"]}

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


def build_anchor_generator_from_spec(
    anchor_spec: AnchorTrainingSpec,
) -> AnchorGenerator:
    """Build ``AnchorGenerator`` from a resolved :class:`AnchorTrainingSpec`."""
    return build_anchor_generator_from_manifest(anchor_spec.manifest_path)
