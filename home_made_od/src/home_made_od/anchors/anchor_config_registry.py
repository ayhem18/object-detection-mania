"""
Registered anchor configuration recipes (dataset-independent).

Recipes are written to ``labeling/anchor_configs/{config_hash}.json`` by
``register_anchor_configs.py``.  All training / sweep / sanity scripts must
reference a registered ``config_hash``; inline method parameters are rejected.
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
from home_made_od.general.path_layout import (
    anchor_configs_root,
    registered_anchor_config_path,
)

# ---------------------------------------------------------------------------
# Canonical recipes (edit here, then run register_anchor_configs.py)
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
    overwrite: bool = False,
) -> Path:
    """Write one recipe to ``labeling/anchor_configs/{config_hash}.json``."""
    payload = recipe_to_payload(recipe)
    config_hash = payload["config_hash"]
    path = registered_anchor_config_path(config_hash)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file() and not overwrite:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing != payload:
            raise FileExistsError(
                f"Anchor config already registered at {path} with different content. "
                "Use overwrite=True to replace."
            )
        return path

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    return path


def save_all_registered_anchor_configs(*, overwrite: bool = False) -> List[Path]:
    paths: List[Path] = []
    for recipe in REGISTERED_ANCHOR_CONFIGS.values():
        paths.append(save_registered_anchor_config(recipe, overwrite=overwrite))
    return paths


def load_registered_anchor_config(config_hash: str) -> Dict[str, Any]:
    """
    Load a registered recipe by hash.

    Raises
    ------
    FileNotFoundError
        If ``labeling/anchor_configs/{config_hash}.json`` does not exist.
    """
    path = registered_anchor_config_path(config_hash)
    if not path.is_file():
        raise FileNotFoundError(
            f"Registered anchor config not found: {path}. "
            "Run register_anchor_configs.py to create labeling/anchor_configs/*.json."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
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
        if "manifest_path" in anchor_config:
            raise ValueError(
                "Direct manifest_path in anchor_config is disabled. "
                "Use anchor_config_hash and ensure anchors_data exists for the split."
            )
        if "config_hash" in anchor_config:
            return str(anchor_config["config_hash"])
        if "method" in anchor_config or "method_parameters" in anchor_config:
            raise ValueError(
                "Inline anchor method/method_parameters are not allowed. "
                "Register configs with register_anchor_configs.py and set "
                "anchor_config_hash in the experiment config."
            )

    raise ValueError(
        "Experiment config must define anchor_config_hash "
        "(or anchor_config.config_hash)."
    )


def list_registered_anchor_config_paths() -> List[Path]:
    root = anchor_configs_root()
    if not root.is_dir():
        return []
    return sorted(root.glob("*.json"))
