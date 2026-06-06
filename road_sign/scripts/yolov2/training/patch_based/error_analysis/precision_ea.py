"""
GT confidence analysis for patch-based YOLOv2 experiments.

Outputs under ``diagnosis/gt_analysis/``:

- ``pure_gt_matching_results.csv`` — per-GT best-prediction stats
- ``object_confidence_distribution.png`` — score / objectness histograms
- ``diagnostic_plots/`` — feature-map area vs confidence scatter plots
- ``visualizations/`` — first N GT matches (fixed preview set for cross-version comparison)
- ``visualizations/uncertain/`` — GT (green) + best pred (red) for low-confidence matches
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
while True:
    if "road_sign" in os.listdir(current_dir) and os.path.isdir(os.path.join(current_dir, "road_sign")):
        break
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:
        raise RuntimeError("Could not find 'road_sign' directory in the parent path.")
    current_dir = parent_dir

workspace_root = current_dir
road_sign_root = os.path.join(workspace_root, "road_sign")
sys.path.insert(0, workspace_root)

from home_made_od.yolo_family.diagnosis.yolov2_diagnosis import (
    DiagnosisDataset,
    analyze_gt_matches,
    plot_diagnostic_results,
    render_deferred_visualizations,
)
from mypt.code_utils.pytorch_utils import seed_everything
from road_sign.scripts.yolov2.training.patch_based.train_scripts.train_utils import (
    build_model,
    gather_pairs_from_subdirs,
    split_by_original_image,
)

DEFAULT_EXPERIMENT_HASH = "06a8faf539a77dad0751e1e855445d05"


def resolve_artifact_dir(experiment_hash: str) -> str:
    base = os.path.join(road_sign_root, "artifacts", "patch_based")
    for name in (experiment_hash, f"__{experiment_hash}"):
        path = os.path.join(base, name)
        if os.path.isfile(os.path.join(path, "config.yaml")):
            return path
    raise FileNotFoundError(f"No artifact directory found for experiment hash {experiment_hash!r}")


def resolve_split_hash_dir(data_dir: str, split_hash_dir: str) -> str:
    if os.path.isdir(split_hash_dir):
        return split_hash_dir
    candidate = os.path.join(data_dir, os.path.basename(split_hash_dir))
    return candidate if os.path.isdir(candidate) else data_dir


def load_anchors(config: Dict[str, Any], data_dir: str, split_hash_dir: str) -> List[List[float]]:
    if "anchors" in config:
        return config["anchors"]
    for root in (split_hash_dir, data_dir):
        path = os.path.join(root, "anchors.json")
        if os.path.isfile(path):
            with open(path, "r") as f:
                return json.load(f)["anchors"]
    raise FileNotFoundError("anchors.json not found")


def load_class_mapping(config: Dict[str, Any], data_dir: str) -> Dict[str, str]:
    for path in (
        os.path.join(data_dir, "class_mapping.json"),
        os.path.join(road_sign_root, "org_data", "class_mapping.json"),
        os.path.join(road_sign_root, "data", "class_mapping.json"),
    ):
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return {str(k): v for k, v in json.load(f).items()}
    return {str(i): f"Class_{i}" for i in range(config["num_classes"])}


def get_val_pairs(data_dir: str, config: Dict[str, Any]) -> List[Tuple[str, str]]:
    _, val_subdirs = split_by_original_image(
        data_dir,
        train_ratio=config.get("train_ratio", 0.9),
        seed=config.get("seed", 42),
    )
    return gather_pairs_from_subdirs(data_dir, val_subdirs)


def load_gt_boxes(label_path: str, orig_w: int, orig_h: int) -> List[List[float]]:
    if not os.path.isfile(label_path):
        return []
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = [float(x) for x in line.split()]
            if len(parts) < 5:
                continue
            cls, ncx, ncy, nw, nh = parts[:5]
            boxes.append([
                cls,
                (ncx - nw / 2) * orig_w,
                (ncy - nh / 2) * orig_h,
                (ncx + nw / 2) * orig_w,
                (ncy + nh / 2) * orig_h,
            ])
    return boxes


def plot_confidence_distribution(df: pd.DataFrame, output_dir: str, low_conf_threshold: float) -> None:
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["pred_obj_prob"], bins=50, color="royalblue", edgecolor="white")
    axes[0].axvline(low_conf_threshold, color="red", linestyle="--", label=f"threshold={low_conf_threshold}")
    axes[0].set_title("Objectness of best-matching prediction per GT")
    axes[0].set_xlabel("P(object)")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(df["pred_score"], bins=50, color="darkorange", edgecolor="white")
    axes[1].axvline(low_conf_threshold, color="red", linestyle="--", label=f"threshold={low_conf_threshold}")
    axes[1].set_title("Detection score of best-matching prediction per GT")
    axes[1].set_xlabel("score")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "object_confidence_distribution.png"), dpi=150)
    plt.close()


def _retarget_vis_tasks(vis_tasks, output_dir: str):
    return [(*task[:8], output_dir, task[9]) for task in vis_tasks]


def run_analysis(
    model,
    val_pairs: List[Tuple[str, str]],
    anchors: List[List[float]],
    img_size: Tuple[int, int],
    preprocess: v2.Compose,
    class_mapping: Dict[str, str],
    device: torch.device,
    output_root: str,
    *,
    low_conf_threshold: float = 0.1,
    preview_count: int = 15,
) -> pd.DataFrame:
    os.makedirs(output_root, exist_ok=True)
    preview_dir = os.path.join(output_root, "visualizations")
    uncertain_dir = os.path.join(output_root, "visualizations", "uncertain")
    os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(uncertain_dir, exist_ok=True)

    val_img_paths = [pair[0] for pair in val_pairs]
    diag_ds = DiagnosisDataset(val_img_paths, img_size, preprocess)
    diag_loader = DataLoader(diag_ds, batch_size=16, shuffle=False, num_workers=2)

    all_diagnostics = []
    all_vis_tasks = []

    for batch_tensors, batch_idxs in tqdm(diag_loader, desc="GT matching"):
        batch_tensors = batch_tensors.to(device)
        batch_gt_boxes = []
        batch_orig_sizes = []
        batch_image_paths = []

        for idx_tensor in batch_idxs:
            idx = idx_tensor.item()
            img_path, label_path = val_pairs[idx]
            orig_w, orig_h = diag_ds.original_sizes[idx]
            gt = load_gt_boxes(label_path, orig_w, orig_h)
            batch_gt_boxes.append(
                torch.tensor(gt, device=device) if gt else torch.zeros((0, 5), device=device)
            )
            batch_orig_sizes.append((orig_w, orig_h))
            batch_image_paths.append(img_path)

        batch_diags, batch_vis_tasks = analyze_gt_matches(
            model=model,
            images=batch_tensors,
            gt_boxes_list=batch_gt_boxes,
            anchors=anchors,
            orig_sizes=batch_orig_sizes,
            visualize=True,
            output_dir=preview_dir,
            image_paths=batch_image_paths,
            class_mapping=class_mapping,
            defer_visualization=True,
        )
        all_diagnostics.extend(batch_diags)
        all_vis_tasks.extend(batch_vis_tasks)

    df = pd.DataFrame(all_diagnostics)
    csv_path = os.path.join(output_root, "pure_gt_matching_results.csv")
    df.to_csv(csv_path, index=False)

    plot_confidence_distribution(df, output_root, low_conf_threshold)
    plot_diagnostic_results(csv_path, os.path.join(output_root, "diagnostic_plots"))

    if df.empty:
        print("No GT objects found in validation split.")
        return df

    preview_tasks = all_vis_tasks[:preview_count]
    uncertain_images = set(
        df.loc[df["pred_score"] < low_conf_threshold, "image_path"].dropna()
    )
    uncertain_tasks = [task for task in all_vis_tasks if task[0] in uncertain_images]

    print(
        f"GT objects: {len(df)} | uncertain (score < {low_conf_threshold}): "
        f"{int((df['pred_score'] < low_conf_threshold).sum())}"
    )
    print(f"Rendering preview set: {len(preview_tasks)} images -> {preview_dir}")
    render_deferred_visualizations(_retarget_vis_tasks(preview_tasks, preview_dir))
    print(f"Rendering uncertain set: {len(uncertain_tasks)} images -> {uncertain_dir}")
    render_deferred_visualizations(_retarget_vis_tasks(uncertain_tasks, uncertain_dir))
    return df


def main():
    parser = argparse.ArgumentParser(description="YOLOv2 patch GT confidence analysis")
    parser.add_argument("--experiment-hash", default=DEFAULT_EXPERIMENT_HASH)
    parser.add_argument("--low-conf-threshold", type=float, default=0.1)
    parser.add_argument(
        "--preview-count",
        type=int,
        default=25,
        help="Number of first validation patches with GT to always visualize",
    )
    args = parser.parse_args()

    seed_everything(42)
    artifact_dir = resolve_artifact_dir(args.experiment_hash)

    with open(os.path.join(artifact_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    data_dir = os.path.join(road_sign_root, config["data_dir"])
    split_hash_dir = resolve_split_hash_dir(data_dir, config.get("split_hash_dir", data_dir))
    img_size = tuple(config.get("img_size", [512, 512]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchors = load_anchors(config, data_dir, split_hash_dir)
    model, transform_stats = build_model(config["num_classes"], len(anchors))
    model.to(device)

    checkpoint_path = os.path.join(artifact_dir, "checkpoints", "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    preprocess = v2.Compose([
        v2.Resize(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=transform_stats.mean, std=transform_stats.std),
    ])

    val_pairs = get_val_pairs(data_dir, config)
    class_mapping = load_class_mapping(config, data_dir)
    output_root = os.path.join(artifact_dir, "diagnosis", "gt_analysis")

    print(f"Experiment: {args.experiment_hash}")
    print(f"Validation patches: {len(val_pairs)}")
    print(f"Output: {output_root}")

    run_analysis(
        model,
        val_pairs,
        anchors,
        img_size,
        preprocess,
        class_mapping,
        device,
        output_root,
        low_conf_threshold=args.low_conf_threshold,
        preview_count=args.preview_count,
    )
    print("Done.")


if __name__ == "__main__":
    main()
