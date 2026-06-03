"""
Train RetinaNet on a road-sign **patch-based** dataset.

Prerequisites::

    uv run python road_sign/scripts/data_scripts/patch_data/prepare_patch_ds.py
    uv run python road_sign/scripts/data_scripts/create_split.py \\
        --dataset-version patch_based --model-name retinanet

Run from the monorepo root::

    uv run python road_sign/scripts/retinanet/training/train_patch.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from home_made_od.general.path_utils import DATASET_VERSION_PATCH

_current = Path(__file__).resolve().parent
while _current != _current.parent:
    if (_current / "road_sign").is_dir() and (_current / "home_made_od").is_dir():
        if str(_current) not in sys.path:
            sys.path.insert(0, str(_current))
        _training_dir = Path(__file__).resolve().parent
        if str(_training_dir) not in sys.path:
            sys.path.insert(0, str(_training_dir))
        break
    _current = _current.parent
else:
    raise RuntimeError("Could not find monorepo root (road_sign + home_made_od).")

from train_utils import ( 
    DEFAULT_ANCHOR_CONFIG_HASH,
    add_common_cli_args,
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
    "train_params": {
        "target_size": (512, 512),
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 1e-4,
        "early_stop_patience": 15,
        "num_workers": 2,
    },
    "model_params": {
        "freeze_backbone_layers": 2,
    },
    "augmentation": {
        "horizontal_flip_p": 0.5,
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
        },
    },
    "class_id_map": None,
    "retinanet_label_start": 1,
    "seed": 42,
    "force_recompute_anchors": False,
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
    if args.force_recompute_anchors:
        config["force_recompute_anchors"] = True
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
    config["augmentation"] = dict(DEFAULT_CONFIG["augmentation"])
    apply_cli_overrides(config, args)

    artifact_dir = run_retinanet_training(DATASET_VERSION_PATCH, config)
    logger.info("Training finished. Artifacts: %s", artifact_dir)


if __name__ == "__main__":
    main()
