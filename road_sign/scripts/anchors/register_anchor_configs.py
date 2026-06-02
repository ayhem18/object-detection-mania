"""
Write canonical anchor clustering recipes to disk.

    road_sign/artifacts/retinanet/anchor_configs/{config_hash}.json

Run from the monorepo root::

    uv run python road_sign/scripts/anchors/register_anchor_configs.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_current = Path(__file__).resolve().parent
while _current != _current.parent:
    if (_current / "road_sign").is_dir() and (_current / "home_made_od").is_dir():
        if str(_current) not in sys.path:
            sys.path.insert(0, str(_current))
        break
    _current = _current.parent
else:
    raise RuntimeError("Could not find monorepo root.")

from home_made_od.anchors.anchor_config_registry import (  # noqa: E402
    DEFAULT_ANCHOR_MODEL_NAME,
    save_all_registered_anchor_configs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register RetinaNet anchor clustering recipes.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_ANCHOR_MODEL_NAME,
        help="Artifacts namespace (default: retinanet).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = save_all_registered_anchor_configs(
        model_name=args.model_name,
        overwrite=args.overwrite,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
