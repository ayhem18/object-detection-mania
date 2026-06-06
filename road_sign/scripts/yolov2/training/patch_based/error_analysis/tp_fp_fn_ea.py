"""
Classify validation predictions into TP / FP / FN and visualize samples.

Uses conf_threshold=0.3 by default. A prediction is TP when it matches an unmatched GT
at iou_threshold (default 0.5); FP otherwise; FN is an unmatched GT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.ops as ops
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

from home_made_od.yolo_family.diagnosis.yolov2_diagnosis import DiagnosisDataset  # noqa: E402
from mypt.code_utils.pytorch_utils import seed_everything  # noqa: E402
from road_sign.scripts.yolov2.training.patch_based.train_scripts.train_utils import (  # noqa: E402
    build_model,
    gather_pairs_from_subdirs,
    split_by_original_image,
)

DEFAULT_EXPERIMENT_HASH = "06a8faf539a77dad0751e1e855445d05"
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_NMS_THRESHOLD = 0.2
DEFAULT_NUM_SAMPLES = 120


def match_predictions_to_gt(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = pred_boxes.device
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)

    sort_idx = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sort_idx]
    tp = torch.zeros(num_preds, device=device)
    fp = torch.zeros(num_preds, device=device)

    if num_preds == 0:
        return tp, fp, sort_idx
    if num_gts == 0:
        return tp, torch.ones(num_preds, device=device), sort_idx

    iou_matrix = ops.box_iou(pred_boxes, gt_boxes)
    gt_matched = torch.zeros(num_gts, dtype=torch.bool, device=device)

    for i in range(num_preds):
        best_iou, best_gt_idx = torch.max(iou_matrix[i], dim=0)
        if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1

    return tp, fp, sort_idx


@dataclass
class ClassifiedDetection:
    category: str
    image_path: str
    score: Optional[float] = None
    pred_class: Optional[int] = None
    gt_class: Optional[int] = None
    pred_box: Optional[Tuple[float, float, float, float]] = None
    gt_box: Optional[Tuple[float, float, float, float]] = None
    iou: Optional[float] = None


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


def load_gt_boxes(label_path: str, orig_w: int, orig_h: int) -> torch.Tensor:
    rows = []
    if not os.path.isfile(label_path):
        return torch.zeros((0, 5))
    with open(label_path, "r") as f:
        for line in f:
            parts = [float(x) for x in line.split()]
            if len(parts) < 5:
                continue
            cls, ncx, ncy, nw, nh = parts[:5]
            rows.append([
                cls,
                (ncx - nw / 2) * orig_w,
                (ncy - nh / 2) * orig_h,
                (ncx + nw / 2) * orig_w,
                (ncy + nh / 2) * orig_h,
            ])
    return torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, 5))


def scale_boxes_to_original(
    boxes: torch.Tensor,
    target_size: Tuple[int, int],
    orig_size: Tuple[int, int],
) -> torch.Tensor:
    orig_w, orig_h = orig_size
    target_h, target_w = target_size
    scaled = boxes.clone()
    scale_x = orig_h / target_h if target_h else 1.0
    scale_y = orig_w / target_w if target_w else 1.0
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    scaled[:, 2] *= scale_x
    scaled[:, 3] *= scale_y
    return scaled


def classify_image(
    preds: torch.Tensor,
    gt_boxes: torch.Tensor,
    image_path: str,
    iou_threshold: float,
) -> List[ClassifiedDetection]:
    records: List[ClassifiedDetection] = []
    gt_coords = gt_boxes[:, 1:5]
    gt_classes = gt_boxes[:, 0]

    if len(preds) == 0:
        for i in range(len(gt_boxes)):
            records.append(ClassifiedDetection(
                category="FN",
                image_path=image_path,
                gt_class=int(gt_classes[i].item()),
                gt_box=tuple(gt_coords[i].tolist()),
            ))
        return records

    pred_boxes = preds[:, :4]
    pred_scores = preds[:, 4]
    pred_classes = preds[:, 5]

    if len(gt_boxes) == 0:
        for i in range(len(preds)):
            records.append(ClassifiedDetection(
                category="FP",
                image_path=image_path,
                score=float(pred_scores[i].item()),
                pred_class=int(pred_classes[i].item()),
                pred_box=tuple(pred_boxes[i].tolist()),
            ))
        return records

    tp, fp, sort_idx = match_predictions_to_gt(
        pred_boxes, pred_scores, gt_coords, iou_threshold=iou_threshold
    )
    sorted_boxes = pred_boxes[sort_idx]
    sorted_scores = pred_scores[sort_idx]
    sorted_classes = pred_classes[sort_idx]
    iou_matrix = ops.box_iou(sorted_boxes, gt_coords)

    gt_matched = torch.zeros(len(gt_coords), dtype=torch.bool)
    for i in range(len(sorted_boxes)):
        best_iou, gt_idx = torch.max(iou_matrix[i], dim=0)
        if tp[i] == 1:
            gt_matched[gt_idx] = True
            records.append(ClassifiedDetection(
                category="TP",
                image_path=image_path,
                score=float(sorted_scores[i].item()),
                pred_class=int(sorted_classes[i].item()),
                gt_class=int(gt_classes[gt_idx].item()),
                pred_box=tuple(sorted_boxes[i].tolist()),
                gt_box=tuple(gt_coords[gt_idx].tolist()),
                iou=float(best_iou.item()),
            ))
        elif fp[i] == 1:
            records.append(ClassifiedDetection(
                category="FP",
                image_path=image_path,
                score=float(sorted_scores[i].item()),
                pred_class=int(sorted_classes[i].item()),
                pred_box=tuple(sorted_boxes[i].tolist()),
                iou=float(best_iou.item()) if len(gt_coords) else None,
            ))

    for gt_idx in range(len(gt_coords)):
        if not gt_matched[gt_idx]:
            records.append(ClassifiedDetection(
                category="FN",
                image_path=image_path,
                gt_class=int(gt_classes[gt_idx].item()),
                gt_box=tuple(gt_coords[gt_idx].tolist()),
            ))

    return records


def draw_sample(record: ClassifiedDetection, class_mapping: Dict[str, str], output_path: str) -> None:
    img = cv2.imread(record.image_path)
    if img is None:
        return

    if record.gt_box is not None:
        gx1, gy1, gx2, gy2 = map(int, record.gt_box)
        cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        gt_name = class_mapping.get(str(record.gt_class), str(record.gt_class))
        cv2.putText(
            img, f"GT:{gt_name}", (gx1, max(gy1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )

    if record.pred_box is not None:
        px1, py1, px2, py2 = map(int, record.pred_box)
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2)
        pred_name = class_mapping.get(str(record.pred_class), str(record.pred_class))
        score_txt = f"{record.score:.2f}" if record.score is not None else "?"
        iou_txt = f" IoU:{record.iou:.2f}" if record.iou is not None else ""
        cv2.putText(
            img, f"Pred:{pred_name} {score_txt}{iou_txt}", (px1, min(py2 + 16, img.shape[0] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )

    cv2.putText(
        img, record.category, (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )
    cv2.imwrite(output_path, img)


def plot_counts(counts: Dict[str, int], output_path: str) -> None:
    categories = ["TP", "FP", "FN"]
    values = [counts.get(cat, 0) for cat in categories]
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, values, color=colors, edgecolor="white")
    plt.title("Detection outcomes on validation patches")
    plt.ylabel("count")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value),
                 ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def visualize_samples(
    records: List[ClassifiedDetection],
    class_mapping: Dict[str, str],
    output_root: str,
    num_samples: Optional[int],
) -> None:
    by_category: Dict[str, List[ClassifiedDetection]] = {"TP": [], "FP": [], "FN": []}
    for record in records:
        by_category[record.category].append(record)



    for category, items in by_category.items():
        out_dir = os.path.join(output_root, "visualizations", category.lower())
        os.makedirs(out_dir, exist_ok=True)

        if num_samples is not None:
            vis_items = items[:num_samples]
        else:
            vis_items = items

        for i, record in enumerate(vis_items):
            stem = Path(record.image_path).stem
            suffix = f"_{record.score:.3f}" if record.score is not None else ""
            out_path = os.path.join(out_dir, f"{category.lower()}_{i:02d}_{stem}{suffix}.png")
            draw_sample(record, class_mapping, out_path)


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
    conf_threshold: float,
    iou_threshold: float,
    nms_threshold: float,
    num_samples: int,
) -> pd.DataFrame:
    os.makedirs(output_root, exist_ok=True)

    val_img_paths = [pair[0] for pair in val_pairs]
    diag_ds = DiagnosisDataset(val_img_paths, img_size, preprocess)
    diag_loader = DataLoader(diag_ds, batch_size=16, shuffle=False, num_workers=2)

    all_records: List[ClassifiedDetection] = []

    for batch_tensors, batch_idxs in tqdm(diag_loader, desc="Classifying predictions"):
        batch_tensors = batch_tensors.to(device)
        batch_preds = model.inference(
            batch_tensors,
            anchors=anchors,
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_threshold,
        )

        for i, idx_tensor in enumerate(batch_idxs):
            idx = idx_tensor.item()
            img_path, label_path = val_pairs[idx]
            orig_w, orig_h = diag_ds.original_sizes[idx]
            gt_boxes = load_gt_boxes(label_path, orig_w, orig_h)

            preds = batch_preds[i]
            if len(preds):
                preds = preds.clone()
                preds[:, :4] = scale_boxes_to_original(
                    preds[:, :4], img_size, (orig_w, orig_h)
                )

            all_records.extend(
                classify_image(preds.cpu(), gt_boxes, img_path, iou_threshold)
            )

    df = pd.DataFrame(asdict(r) for r in all_records)
    df.to_csv(os.path.join(output_root, "classified_detections.csv"), index=False)

    counts = df["category"].value_counts().to_dict() if not df.empty else {}
    summary = {
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "nms_threshold": nms_threshold,
        "counts": {cat: int(counts.get(cat, 0)) for cat in ("TP", "FP", "FN")},
        "num_samples_visualized_per_category": num_samples,
    }
    with open(os.path.join(output_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    plot_counts(summary["counts"], os.path.join(output_root, "counts_bar.png"))
    visualize_samples(all_records, class_mapping, output_root, num_samples)

    print(json.dumps(summary, indent=2))
    return df


def main():
    parser = argparse.ArgumentParser(description="YOLOv2 patch TP/FP/FN analysis")
    parser.add_argument("--experiment-hash", default=DEFAULT_EXPERIMENT_HASH)
    parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--nms-threshold", type=float, default=DEFAULT_NMS_THRESHOLD)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
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
    output_root = os.path.join(artifact_dir, "diagnosis", "tp_fp_fn")

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
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        nms_threshold=args.nms_threshold,
        num_samples=args.num_samples,
    )
    print("Done.")


if __name__ == "__main__":
    main()
