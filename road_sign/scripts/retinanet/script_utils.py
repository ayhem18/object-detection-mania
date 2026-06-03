"""
Shared helpers for RetinaNet road-sign scripts (training + sanity checks).

Dataset/split resolution, target-size lookup, artifact paths, and monorepo bootstrap.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import yaml

from home_made_od.general.path_utils import (
    DATASET_VERSION_PATCH,
    DATASET_VERSION_RESIZED,
    RoadSignDatasetVersion,
    create_and_cache_split,
    load_dataset_config,
    patch_config_path,
    resolve_latest_dataset_hash,
    resolve_latest_split_hash,
    road_sign_artifacts_root,
)
from home_made_od.retinanet.retinanet_anchors import (
    AnchorTrainingSpec,
    anchor_spec_from_path,
    build_anchor_generator,
    load_anchor_config,
)
from home_made_od.retinanet.retinanet_detector import build_retinanet

logger = logging.getLogger(__name__)

RETINANET_MODEL_NAME = "retinanet"
VERSION_CHOICES: List[RoadSignDatasetVersion] = [
    DATASET_VERSION_RESIZED,
    DATASET_VERSION_PATCH,
]
DEFAULT_DATASET_VERSIONS: List[RoadSignDatasetVersion] = list(VERSION_CHOICES)
DEFAULT_TARGET_SIZE: Tuple[int, int] = (512, 512)


def add_monorepo_to_sys_path(start: Path | None = None) -> Path:
    """Ensure monorepo root (``road_sign`` + ``home_made_od``) is on ``sys.path``."""
    current = (start or Path(__file__).resolve().parent).resolve()
    while current != current.parent:
        if (current / "road_sign").is_dir() and (current / "home_made_od").is_dir():
            root = current
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
        current = current.parent
    raise RuntimeError("Could not find monorepo root (road_sign + home_made_od).")


def add_retinanet_scripts_to_sys_path(caller_file: str | Path) -> Path:
    """Add ``road_sign/scripts/retinanet`` so ``script_utils`` is importable."""
    retinanet_dir = Path(caller_file).resolve().parents[1]
    if str(retinanet_dir) not in sys.path:
        sys.path.insert(0, str(retinanet_dir))
    return retinanet_dir


def add_training_dir_to_sys_path(caller_file: str | Path) -> Path:
    """Add ``road_sign/scripts/retinanet/training`` for ``train_utils`` imports."""
    training_dir = Path(caller_file).resolve().parent
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    return training_dir


def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-hash", default=None, help="Latest for version if omitted.")
    parser.add_argument(
        "--split-hash",
        default=None,
        help="Latest split if omitted; created if none exist (training only).",
    )
    parser.add_argument(
        "--force-recompute-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute anchors even when anchor_config.json exists (default: true).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Resume from a full detector .pt checkpoint.",
    )


def resolve_dataset_versions(
    args: argparse.Namespace,
    *,
    single_attr: str = "dataset_version",
    multi_attr: str | None = "dataset_versions",
) -> List[RoadSignDatasetVersion]:
    """
    Resolve which dataset versions to run.

    Supports either a single ``--dataset-version`` or repeated ``--dataset-versions``.
    """
    if multi_attr is not None:
        multi = getattr(args, multi_attr, None)
        if multi:
            return list(multi)
    single = getattr(args, single_attr, None)
    if single is not None:
        return [single]
    return list(DEFAULT_DATASET_VERSIONS)


def load_optional_patch_config(dataset_hash: str) -> Optional[Dict[str, Any]]:
    path = patch_config_path(dataset_hash)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_target_size(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    *,
    train_config: Dict[str, Any] | None = None,
    cli_target_size: Sequence[int] | None = None,
    default: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> Tuple[int, int]:
    """
    Resolve ``(height, width)`` for a dataset version.

    Precedence: CLI override → training config ``train_params.target_size`` →
    dataset config → patch config (patch_based only) → ``default``.
    """
    if cli_target_size is not None:
        return int(cli_target_size[0]), int(cli_target_size[1])

    if train_config is not None:
        train_params = train_config.get("train_params") or {}
        if "target_size" in train_params:
            ts = train_params["target_size"]
            return int(ts[0]), int(ts[1])

    ds_cfg = load_dataset_config(version, dataset_hash)
    if "target_size" in ds_cfg:
        ts = ds_cfg["target_size"]
        return int(ts[0]), int(ts[1])

    if version == DATASET_VERSION_PATCH:
        patch_cfg = load_optional_patch_config(dataset_hash)
        if patch_cfg and "target_size" in patch_cfg:
            ts = patch_cfg["target_size"]
            return int(ts[0]), int(ts[1])

    return default


def resolve_split_hash(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: Optional[str],
    *,
    train_ratio: float = 0.85,
    seed: int = 42,
    create_if_missing: bool = False,
) -> Optional[str]:
    """
    Resolve a cached split hash, optionally creating one when none exists.
    """
    resolved = split_hash or resolve_latest_split_hash(version, dataset_hash)
    if resolved is not None:
        return resolved
    if not create_if_missing:
        return None

    logger.info(
        "No split found; creating split train_ratio=%s seed=%s",
        train_ratio,
        seed,
    )
    _, _, created = create_and_cache_split(
        version=version,
        dataset_hash=dataset_hash,
        train_ratio=train_ratio,
        seed=seed,
    )
    return created


def resolve_dataset_and_split(
    version: RoadSignDatasetVersion,
    config: Dict[str, Any],
) -> Tuple[str, str]:
    """Resolve dataset/split hashes from a training config dict (creates split if needed)."""
    dataset_hash = config.get("dataset_hash") or resolve_latest_dataset_hash(version)
    if dataset_hash is None:
        raise FileNotFoundError(
            f"No dataset registered for version {version!r}. Run the data prep script first."
        )

    split_params = config["split_params"]
    split_hash = resolve_split_hash(
        version,
        dataset_hash,
        config.get("split_hash"),
        train_ratio=split_params["train_ratio"],
        seed=split_params["seed"],
        create_if_missing=True,
    )
    if split_hash is None:
        raise FileNotFoundError(
            f"No split for {version}/{dataset_hash}. Run create_split.py first."
        )
    return dataset_hash, split_hash


def retinanet_artifact_dir(
    check_name: str,
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    *extra_segments: str,
    output_dir_override: str | Path | None = None,
    mkdir: bool = False,
) -> Path:
    """
    Path under ``road_sign/artifacts/retinanet/{check_name}/...``.

    With ``output_dir_override``, nests under
    ``{override}/{version}/{dataset_hash}/{split_hash}/[*extra]``.
    """
    tail = (version, dataset_hash, split_hash, *extra_segments)
    if output_dir_override:
        path = Path(output_dir_override).joinpath(*tail)
    else:
        path = road_sign_artifacts_root() / RETINANET_MODEL_NAME / check_name
        path = path.joinpath(*tail)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def load_finalized_anchor_spec(config_path: str | Path) -> AnchorTrainingSpec:
    """Load a finalized ``anchor_config.json`` as :class:`AnchorTrainingSpec`."""
    return anchor_spec_from_path(config_path)


def read_anchor_level_metadata(
    source: Union[str, Path, AnchorTrainingSpec],
) -> Dict[str, Any]:
    """Read requested/used FPN levels and provenance from a finalized anchor config."""
    if isinstance(source, AnchorTrainingSpec):
        config_path = source.config_path
    else:
        config_path = Path(source)
    config = load_anchor_config(config_path)
    used_levels = list(config["used_fpn_levels"])
    requested_levels = list(config.get("requested_fpn_levels", used_levels))
    provenance = dict(config.get("level_provenance", {}))
    added_levels = [level for level, source_kind in provenance.items() if source_kind == "added"]
    return {
        "config_path": config_path,
        "requested_fpn_levels": requested_levels,
        "used_fpn_levels": used_levels,
        "used_layers": list(config.get("used_layers", [])),
        "level_provenance": provenance,
        "added_fpn_levels": added_levels,
    }


def log_anchor_level_metadata(
    anchor_spec: AnchorTrainingSpec,
    *,
    label: str = "Building RetinaNet",
) -> Dict[str, Any]:
    meta = read_anchor_level_metadata(anchor_spec)
    logger.info(
        "%s (%s): requested=%s used=%s added=%s from %s",
        label,
        anchor_spec.method,
        meta["requested_fpn_levels"],
        meta["used_fpn_levels"],
        meta["added_fpn_levels"],
        meta["config_path"],
    )
    return meta


def build_retinanet_from_spec(
    anchor_spec: AnchorTrainingSpec,
    *,
    num_classes: int,
    img_size: Tuple[int, int],
    device: torch.device,
    checkpoint_path: str | Path | None = None,
    freeze_backbone_layers: int = 2,
    log_build: bool = True,
) -> torch.nn.Module:
    """
    Build RetinaNet from a finalized anchor spec.

    Wires ``build_anchor_generator`` → ``build_retinanet`` with the same generator
    instance (no duplicate anchor build).
    """
    if log_build:
        log_anchor_level_metadata(anchor_spec)
    anchor_generator = build_anchor_generator(anchor_spec)
    return build_retinanet(
        anchor_spec=anchor_spec,
        num_classes=num_classes,
        img_size=img_size,
        device=device,
        anchor_generator=anchor_generator,
        checkpoint_path=checkpoint_path,
        freeze_backbone_layers=freeze_backbone_layers,
    )


def anchor_config_summary_payload(
    anchor_spec: AnchorTrainingSpec,
    *,
    config_hash: str | None = None,
) -> Dict[str, Any]:
    """Compact anchor metadata for JSON summaries and run records."""
    meta = read_anchor_level_metadata(anchor_spec)
    return {
        "config_hash": config_hash or anchor_spec.config_hash,
        "method": anchor_spec.method,
        "requested_fpn_levels": meta["requested_fpn_levels"],
        "used_fpn_levels": meta["used_fpn_levels"],
        "used_layers": meta["used_layers"],
        "level_provenance": meta["level_provenance"],
    }
