"""
Patch-based RetinaNet inference on full test images (sliding-window + global NMS).

Mirrors ``yolov2/.../patch_inference/basic_inference.py`` but loads a trained RetinaNet
experiment (``experiment_config.json`` + ``checkpoints/best_model.pt``).

Run from monorepo root::

    uv run python road_sign/scripts/retinanet/patch_based/patch_inference/basic_inference.py \\
        --artifact-dir road_sign/artifacts/retinanet/patch_based/{dataset_hash}/{split_hash}/{experiment_hash} \\
        --test-images-dir road_sign/data/test/images \\
        --conf-thresh 0.05 --nms-thresh 0.4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_PATCH_INFERENCE_DIR = Path(__file__).resolve().parent
_RETINANET_SCRIPTS = _PATCH_INFERENCE_DIR.parents[1]
for _path in (_PATCH_INFERENCE_DIR, _RETINANET_SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from script_utils import add_monorepo_to_sys_path  # noqa: E402

add_monorepo_to_sys_path()

from script_utils import (  # noqa: E402
    RETINANET_MODEL_NAME,
    build_retinanet_from_spec,
    load_finalized_anchor_spec,
    resolve_target_size,
)
from train_utils import normalize_class_id_map_config  # noqa: E402

from home_made_od.general.path_utils import (  # noqa: E402
    DATASET_VERSION_PATCH,
    EXPERIMENT_CONFIG_FILENAME,
    model_experiment_dir,
    road_sign_data_root,
    road_sign_root,
)
from home_made_od.retinanet.error_analysis.retinanet_obj_ea import (  # noqa: E402
    configure_retinanet_postprocess,
)
from road_sign.utils.data_utils import (  # noqa: E402
    load_road_sign_class_mapping,
    retinanet_cls_id_to_name,
    retinanet_num_classes,
)

from patch_inference_utils import (  # noqa: E402
    PreSlicedPatchDataset,
    collect_image_stems,
    default_preprocess,
    global_nms,
    map_patch_boxes_to_image,
    prepare_patches_offline,
)

logger = logging.getLogger(__name__)

DEFAULT_PATCH_SCALES = [512, 1024, 2048]
DEFAULT_STRIDE_RATIO = 0.75
TMP_PATCH_DIR_NAME = "temp_retinanet_inference_patches"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch-based RetinaNet test inference.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Experiment directory containing experiment_config.json and checkpoints/.",
    )
    parser.add_argument("--experiment-hash", default=None)
    parser.add_argument("--dataset-hash", default=None)
    parser.add_argument("--split-hash", default=None)
    parser.add_argument(
        "--test-images-dir",
        type=Path,
        default=None,
        help="Directory of full-size test images (default: road_sign/data/test/images).",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--conf-thresh", type=float, default=0.05)
    parser.add_argument("--nms-thresh", type=float, default=0.4)
    parser.add_argument(
        "--patch-nms-thresh",
        type=float,
        default=1.0,
        help="Per-patch RetinaNet NMS (1.0 = disabled; use --nms-thresh for global NMS).",
    )
    parser.add_argument("--patch-scales", type=int, nargs="+", default=None)
    parser.add_argument("--stride-ratio", type=float, default=DEFAULT_STRIDE_RATIO)
    parser.add_argument("--keep-temp-patches", action="store_true")
    parser.add_argument("--device", default=None, help="cuda | cpu (auto if omitted).")
    return parser.parse_args()


def discover_test_images(test_dir: Path) -> List[Path]:
    images: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        images.extend(sorted(test_dir.glob(pattern)))
    return images


def resolve_artifact_dir(args: argparse.Namespace, run_config: Dict[str, Any] | None) -> Path:
    if args.artifact_dir is not None:
        path = args.artifact_dir.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"artifact-dir not found: {path}")
        return path

    experiment_hash = args.experiment_hash
    if experiment_hash is None:
        raise ValueError("Provide --artifact-dir or --experiment-hash (+ dataset/split hashes).")

    dataset_hash = args.dataset_hash or (run_config or {}).get("dataset_hash")
    split_hash = args.split_hash or (run_config or {}).get("split_hash")
    if not dataset_hash or not split_hash:
        raise ValueError(
            "--experiment-hash requires --dataset-hash and --split-hash "
            "(or load them from a partial config)."
        )

    return model_experiment_dir(
        RETINANET_MODEL_NAME,
        DATASET_VERSION_PATCH,
        dataset_hash,
        split_hash,
        experiment_hash,
    )


def load_run_config(artifact_dir: Path) -> Dict[str, Any]:
    config_path = artifact_dir / EXPERIMENT_CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_patch_scales(run_config: Dict[str, Any], cli_scales: Optional[Sequence[int]]) -> List[int]:
    if cli_scales:
        return [int(s) for s in cli_scales]
    patch_cfg = run_config.get("patch_config") or {}
    return [int(s) for s in patch_cfg.get("scales", DEFAULT_PATCH_SCALES)]


def submission_class_names(
    class_mapping: Dict[int, str],
    class_id_map: Dict[int, int],
) -> Dict[int, str]:
    """Model label id → Kaggle-style class name (spaces → underscores)."""
    id_to_name = retinanet_cls_id_to_name(class_mapping, class_id_map)
    return {lid: name.replace(" ", "_") for lid, name in id_to_name.items()}


def format_prediction_string(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: Dict[int, str],
    *,
    fallback_class: str,
) -> str:
    if len(boxes) == 0:
        return f"{fallback_class} 0.0001 0 0 1 1"

    parts = []
    for box, score, label in zip(boxes, scores, labels):
        name = class_names.get(int(label), f"class_{int(label)}")
        x1, y1, x2, y2 = box
        parts.append(
            f"{name} {float(score):.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}"
        )
    return " ".join(parts)


def run_patch_inference(
    *,
    artifact_dir: Path,
    test_images_dir: Path,
    output_csv: Path,
    batch_size: int,
    num_workers: int,
    conf_thresh: float,
    nms_thresh: float,
    patch_nms_thresh: float,
    patch_scales: Sequence[int],
    stride_ratio: float,
    keep_temp_patches: bool,
    device: torch.device,
) -> Path:
    run_config = load_run_config(artifact_dir)
    dataset_hash = run_config["dataset_hash"]
    split_hash = run_config["split_hash"]

    target_size = resolve_target_size(
        DATASET_VERSION_PATCH,
        dataset_hash,
        train_config=run_config,
    )
    class_id_map = normalize_class_id_map_config(run_config["class_id_map"])
    class_mapping = load_road_sign_class_mapping(DATASET_VERSION_PATCH, dataset_hash)
    class_names = submission_class_names(class_mapping, class_id_map)
    fallback_class = next(iter(class_names.values()), "No_parking")

    # anchor_config_path = run_config.get("anchor_config_path")

    anchor_config_path = Path(os.path.join(artifact_dir, "anchor_config.json"))
    if not anchor_config_path:
        raise KeyError("experiment_config.json missing anchor_config_path")
    anchor_spec = load_finalized_anchor_spec(anchor_config_path)

    num_classes = int(run_config.get("num_classes", retinanet_num_classes(class_id_map)))
    checkpoint_path = artifact_dir / "checkpoints" / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_images = discover_test_images(test_images_dir)
    if not test_images:
        raise FileNotFoundError(f"No images found in {test_images_dir}")

    tmp_patch_dir = road_sign_root() / "artifacts" / TMP_PATCH_DIR_NAME
    logger.info("Stage 1: extracting patches to %s", tmp_patch_dir)
    all_metadata = prepare_patches_offline(
        test_images,
        patch_scales,
        target_size,
        tmp_patch_dir,
        stride_ratio=stride_ratio,
    )
    logger.info("Extracted %d patches from %d images", len(all_metadata), len(test_images))

    logger.info("Stage 2: loading model from %s", checkpoint_path)
    model = build_retinanet_from_spec(
        anchor_spec,
        num_classes=num_classes,
        img_size=target_size,
        device=device,
        checkpoint_path=checkpoint_path,
        freeze_backbone_layers=0,
        log_build=True,
    )
    model.eval()
    configure_retinanet_postprocess(
        model, score_thresh=conf_thresh, nms_thresh=patch_nms_thresh
    )

    preprocess = default_preprocess(target_size)
    dataset = PreSlicedPatchDataset(all_metadata, tmp_patch_dir, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    stems = collect_image_stems(test_images)
    aggregated: Dict[str, Dict[str, List[torch.Tensor]]] = {
        stem: {"boxes": [], "scores": [], "labels": []} for stem in stems
    }

    with torch.no_grad():
        for batch_tensors, batch_idxs in tqdm(dataloader, desc="Inferencing patches"):
            batch_tensors = batch_tensors.to(device)
            outputs = model([img for img in batch_tensors])

            for i, output in enumerate(outputs):
                boxes = output["boxes"]
                scores = output["scores"]
                labels = output["labels"]
                if boxes.numel() == 0:
                    continue

                meta = all_metadata[int(batch_idxs[i])]
                img_stem = Path(meta.img_path).stem
                mapped = map_patch_boxes_to_image(boxes, meta, target_size)
                valid = (mapped[:, 2] > mapped[:, 0]) & (mapped[:, 3] > mapped[:, 1])
                if not valid.any():
                    continue

                aggregated[img_stem]["boxes"].append(mapped[valid].cpu())
                aggregated[img_stem]["scores"].append(scores[valid].cpu())
                aggregated[img_stem]["labels"].append(labels[valid].cpu())

    logger.info("Stage 3: global NMS and CSV export")
    results = []
    for stem in stems:
        dets = aggregated[stem]
        if not dets["boxes"]:
            pred_str = format_prediction_string(
                np.array([]),
                np.array([]),
                np.array([]),
                class_names,
                fallback_class=fallback_class,
            )
        else:
            fb = torch.cat(dets["boxes"])
            fs = torch.cat(dets["scores"])
            fl = torch.cat(dets["labels"])
            nms_boxes, nms_scores, nms_labels = global_nms(fb, fs, fl, nms_thresh)
            pred_str = format_prediction_string(
                nms_boxes, nms_scores, nms_labels, class_names, fallback_class=fallback_class
            )
        results.append({"image_id": stem, "PredictionString": pred_str})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    logger.info("Wrote submission: %s", output_csv)

    if not keep_temp_patches and tmp_patch_dir.exists():
        shutil.rmtree(tmp_patch_dir)

    return output_csv


def default_output_csv(artifact_dir: Path, conf: float, nms: float) -> Path:
    name = f"submission_patch_conf{conf}_nms{nms}.csv"
    return artifact_dir / "submissions" / name


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    artifact_dir = os.path.join(road_sign_root(), "artifacts", "retinanet", "patch_based", "43a3ebfcc2136949b96b1935c19655c4", "1f0d2d08db5ed4d8d6d8377c816820e5", "4c7cca486ae49082c4d5d785fdfdfb49")
    test_images_dir = os.path.join(road_sign_root(), "org_data", "test", "images")

    artifact_dir = Path(artifact_dir)
    test_images_dir = Path(test_images_dir)

    # artifact_dir = resolve_artifact_dir(args, None)
    run_config = load_run_config(artifact_dir)
    output_csv = args.output_csv or default_output_csv(
        artifact_dir, args.conf_thresh, args.nms_thresh
    )
    patch_scales = resolve_patch_scales(run_config, args.patch_scales)

    print("=" * 60)
    print("RETINANET PATCH INFERENCE")
    print("=" * 60)
    print(f"  artifact_dir:   {artifact_dir}")
    print(f"  test_images:    {test_images_dir}")
    print(f"  patch_scales:   {patch_scales}")
    print(f"  conf / nms:     {args.conf_thresh} / {args.nms_thresh}")
    print(f"  device:         {device}")

    run_patch_inference(
        artifact_dir=artifact_dir,
        test_images_dir=test_images_dir,
        output_csv=output_csv.resolve(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        patch_nms_thresh=args.patch_nms_thresh,
        patch_scales=patch_scales,
        stride_ratio=args.stride_ratio,
        keep_temp_patches=args.keep_temp_patches,
        device=device,
    )


if __name__ == "__main__":
    main()
