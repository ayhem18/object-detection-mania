"""
Register dataset-independent anchor configuration recipes on disk.

Run from repo root::

    uv run python src/dl_lib/etalon_object_detection/scripts/anchors/register_anchor_configs.py

Writes::

    labeling/anchor_configs/{config_hash}.json
"""

from __future__ import annotations

import argparse

from dl_lib.etalon_object_detection.modules.anchors.anchor_config_registry import (
    REGISTERED_ANCHOR_CONFIGS,
    recipe_to_payload,
    save_all_registered_anchor_configs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save anchor configuration recipes to labeling/anchor_configs/."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing JSON files even when content differs.",
    )
    args = parser.parse_args()

    paths = save_all_registered_anchor_configs(overwrite=args.overwrite)

    print(f"Registered {len(paths)} anchor config(s):")
    for name, recipe in REGISTERED_ANCHOR_CONFIGS.items():
        config_hash = recipe_to_payload(recipe)["config_hash"]
        matching = [p for p in paths if p.stem == config_hash]
        path = matching[0] if matching else paths[0]
        print(f"  [{name}] {config_hash} -> {path}")


if __name__ == "__main__":
    main()
