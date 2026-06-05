"""
Train RetinaNet on a road-sign **patch-based** dataset.

Prerequisites::

    uv run python road_sign/scripts/data_scripts/patch_data/prepare_patch_ds.py
    uv run python road_sign/scripts/data_scripts/create_split.py \\
        --dataset-version patch_based --model-name retinanet

Run from the monorepo root::

    uv run python road_sign/scripts/retinanet/patch_based/train_patch.py
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from script_utils import (  # noqa: E402
    add_common_cli_args,
    add_monorepo_to_sys_path,
    add_training_dir_to_sys_path,
)

add_training_dir_to_sys_path(__file__)
add_monorepo_to_sys_path()

from home_made_od.general.path_utils import DATASET_VERSION_PATCH  # noqa: E402

from train_utils import (  # noqa: E402
    DEFAULT_ANCHOR_CONFIG_HASH,
    DEFAULT_TRAIN_AUGMENTATION,
    DEFAULT_TRAIN_PARAMS,
    run_retinanet_training,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict = {
    "experiment_name": "retinanet_patch",
    "dataset_version": DATASET_VERSION_PATCH,
    "dataset_hash": None,
    "split_hash": None,
    "anchor_config_hash": DEFAULT_ANCHOR_CONFIG_HASH,
    "split_params": {
        "train_ratio": 0.9,
        "seed": 42,
    },
    "train_params": dict(DEFAULT_TRAIN_PARAMS),
    "model_params": {
        "freeze_backbone_layers": 2,
    },
    "augmentation": copy.deepcopy(DEFAULT_TRAIN_AUGMENTATION),
    "class_id_map": None,
    "retinanet_label_start": 1,
    "seed": 42,
    "force_recompute_anchors": True,
    "checkpoint_path": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RetinaNet training on patch-based road-sign data.")
    add_common_cli_args(parser)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    if args.dataset_hash:
        config["dataset_hash"] = args.dataset_hash
    if args.split_hash:
        config["split_hash"] = args.split_hash
    config["force_recompute_anchors"] = args.force_recompute_anchors
    if args.checkpoint:
        config["checkpoint_path"] = args.checkpoint
    if args.batch_size is not None:
        config["train_params"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["train_params"]["epochs"] = args.epochs
    if args.lr is not None:
        config["train_params"]["learning_rate"] = args.lr
    if args.seed is not None:
        config["seed"] = args.seed
        config["split_params"]["seed"] = args.seed


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = DEFAULT_CONFIG.copy()
    config["train_params"] = dict(DEFAULT_CONFIG["train_params"])
    config["split_params"] = dict(DEFAULT_CONFIG["split_params"])
    config["model_params"] = dict(DEFAULT_CONFIG["model_params"])
    config["augmentation"] = copy.deepcopy(DEFAULT_CONFIG["augmentation"])
    apply_cli_overrides(config, args)

    artifact_dir = run_retinanet_training(DATASET_VERSION_PATCH, config)
    logger.info("Training finished. Artifacts: %s", artifact_dir)


if __name__ == "__main__":
    main()
