"""
Registered anchor configuration recipes (dataset-independent).

Recipes are written to::

    road_sign/artifacts/{model_name}/anchor_configs/{config_hash}.json

Run ``road_sign/scripts/anchors/register_anchor_configs.py`` (or
``save_all_registered_anchor_configs()``) after editing recipes below.
Training configs reference a ``config_hash`` only — inline method parameters
are rejected by :func:`resolve_anchor_config_hash`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from home_made_od.anchors.anchor_computation_strategies import (
    ASSIGNMENT_METHODS,
    compute_anchor_config_hash,
    validate_method_parameters,
)
from home_made_od.general.path_utils import (
    registered_anchor_config_path,
    registered_anchor_configs_root,
)

DEFAULT_ANCHOR_MODEL_NAME = "retinanet"

# ---------------------------------------------------------------------------
# Canonical recipes (edit here, then register on disk)
# ---------------------------------------------------------------------------

AREA_BASED_ANCHOR_CONFIG: Dict[str, Any] = {
    "method": "area_based",
    "method_parameters": {
        "fpn_coefficient": 4,
        "num_aspect_ratios": 5,
        "num_base_sizes": 5,
        "min_etalons_per_level": 15,
        "scales": [1.0, 2 ** (1.0 / 3), 2 ** (2.0 / 3)],
        "seed": 42,
        "max_iters": 100,
    },
}

DIMENSION_BASED_ANCHOR_CONFIG: Dict[str, Any] = {
    "method": "dimension_based",
    "method_parameters": {
        "fpn_coefficient": 4,
        "num_aspect_ratios": 5,
        "num_base_sizes": 5,
        "min_etalons_per_level": 15,
        "scales": [1.0, 2 ** (1.0 / 3), 2 ** (2.0 / 3)],
        "seed": 42,
        "fallback_level": "P3",
        "max_iters": 100,
    },
}

REGISTERED_ANCHOR_CONFIGS: Dict[str, Dict[str, Any]] = {
    "area_based": AREA_BASED_ANCHOR_CONFIG,
    "dimension_based": DIMENSION_BASED_ANCHOR_CONFIG,
}


def recipe_to_payload(recipe: Dict[str, Any]) -> Dict[str, Any]:
    method = recipe["method"]
    if method not in ASSIGNMENT_METHODS:
        raise ValueError(f"Unknown anchor method {method!r}.")
    method_parameters = validate_method_parameters(method, recipe["method_parameters"])
    config_hash = compute_anchor_config_hash(method, method_parameters)
    return {
        "config_hash": config_hash,
        "method": method,
        "method_parameters": method_parameters,
    }


def save_registered_anchor_config(
    recipe: Dict[str, Any],
    *,
    model_name: str = DEFAULT_ANCHOR_MODEL_NAME,
    overwrite: bool = False,
) -> Path:
    """Write one recipe under ``artifacts/{model_name}/anchor_configs/``."""
    payload = recipe_to_payload(recipe)
    config_hash = payload["config_hash"]
    path = registered_anchor_config_path(config_hash, model_name=model_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file() and not overwrite:
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise FileExistsError(
                f"Anchor config already registered at {path} with different content. "
                "Use overwrite=True to replace."
            )
        return path

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)
    return path


def save_all_registered_anchor_configs(
    *,
    model_name: str = DEFAULT_ANCHOR_MODEL_NAME,
    overwrite: bool = False,
) -> List[Path]:
    paths: List[Path] = []
    for recipe in REGISTERED_ANCHOR_CONFIGS.values():
        paths.append(
            save_registered_anchor_config(
                recipe, model_name=model_name, overwrite=overwrite
            )
        )
    return paths


def load_registered_anchor_config(
    config_hash: str,
    *,
    model_name: str = DEFAULT_ANCHOR_MODEL_NAME,
) -> Dict[str, Any]:
    """
    Load a registered recipe by hash.

    Raises
    ------
    FileNotFoundError
        If the recipe file does not exist under ``registered_anchor_configs_root``.
    """
    path = registered_anchor_config_path(config_hash, model_name=model_name)
    if not path.is_file():
        root = registered_anchor_configs_root(model_name)
        raise FileNotFoundError(
            f"Registered anchor config not found: {path}. "
            f"Run register_anchor_configs.py to populate {root}/*.json."
        )
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("config_hash") != config_hash:
        raise ValueError(
            f"config_hash mismatch in {path}: "
            f"expected {config_hash}, file declares {payload.get('config_hash')}."
        )
    return payload


def resolve_anchor_config_hash(experiment_config: Dict[str, Any]) -> str:
    """
    Extract ``config_hash`` from an experiment / sweep config.

    Accepts ``anchor_config_hash`` or ``anchor_config.config_hash`` only.
    Inline ``method`` / ``method_parameters`` are rejected.
    """
    if "anchor_config_hash" in experiment_config:
        return str(experiment_config["anchor_config_hash"])

    anchor_config = experiment_config.get("anchor_config")
    if isinstance(anchor_config, dict):
        if "config_path" in anchor_config or "manifest_path" in anchor_config:
            raise ValueError(
                "Direct anchor config paths in experiment config are not supported. "
                "Set anchor_config_hash and ensure the registered recipe exists under "
                f"{registered_anchor_configs_root()}."
            )
        if "config_hash" in anchor_config:
            return str(anchor_config["config_hash"])
        if "method" in anchor_config or "method_parameters" in anchor_config:
            raise ValueError(
                "Inline anchor method/method_parameters are not allowed. "
                "Register recipes (register_anchor_configs.py) and set "
                "anchor_config_hash in the experiment config."
            )

    raise ValueError(
        "Experiment config must define anchor_config_hash "
        "(or anchor_config.config_hash)."
    )


def list_registered_anchor_config_paths(
    model_name: str = DEFAULT_ANCHOR_MODEL_NAME,
) -> List[Path]:
    root = registered_anchor_configs_root(model_name)
    if not root.is_dir():
        return []
    return sorted(root.glob("*.json"))
