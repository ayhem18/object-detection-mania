"""
Run a trained RetinaNet detector on unlabeled images and export noisy YOLO labels for CVAT.

Edit ``CONFIG`` below, then::

    uv run python src/dl_lib/etalon_object_detection/scripts/labeling/large_diameter_model_predictions/large_diameter_model_preds.py
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from dl_lib.common_tools.path_utils import get_data_dir
from dl_lib.etalon_object_detection.modules.path_layout import (
    legacy_run_dir,
    resolve_run_anchor_manifest,
    resolve_run_img_size,
    run_checkpoint_path,
)
from dl_lib.etalon_object_detection.modules.ds_utils import DL_LIB_ETALON_FIXED_SIZE, collate_fn
from dl_lib.etalon_object_detection.modules.retinanet.error_analysis.retinanet_obj_ea import (
    configure_retinanet_postprocess,
)
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_anchors import anchor_spec_from_manifest_path
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_detector import (
    build_dl_lab_etalon_retinanet_from_trained_weights,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

CVAT_PREFIX = "digital_lab_wired_etalons/data_as_images"
RETINANET_CLASS_ID_ETALON = 1
YOLO_CLASS_ID_WIRED = 0
BATCH_SIZE_INFERENCE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG: Dict[str, Any] = {
    # Required — experiment run directory under artifacts/runs/.../.../...
    "run_dir": legacy_run_dir(
        "5cd1d267a6d8e41976d73036687689b9",
        "47f9a3b20ce7b5cc45d6dbcfb18e3eec",
        "5433699fed01bdcc5001d7507e7f383b",
    ),
    "images_dir": get_data_dir() / "labeling" / "data_as_images" / "batch_small_diameteres_2",

    # Optional — auto-resolved from run_dir when omitted
    "anchor_manifest": "",
    "img_size": None,  # defaults to model_params.img_size or DL_LIB_ETALON_FIXED_SIZE
    # Inference
    "score_thresh": 0.5,
    "nms_thresh": 0.3,
    "batch_size": BATCH_SIZE_INFERENCE,
}


def collect_image_paths(images_dir: Path) -> List[Path]:
    image_paths: List[Path] = []
    for root, _, files in os.walk(images_dir):
        for file_name in sorted(files):
            if file_name.lower().endswith(".png"):
                image_paths.append(Path(root) / file_name)
    return sorted(image_paths)


def resolve_inference_settings(
    run_dir: Path,
    anchor_manifest: Optional[Path],
    img_size: Optional[Tuple[int, int]],
) -> Tuple[Path, Path, Tuple[int, int]]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    checkpoint = run_checkpoint_path(run_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    resolved_manifest = resolve_run_anchor_manifest(
        run_dir,
        explicit=str(anchor_manifest) if anchor_manifest else None,
    )
    resolved_img_size = resolve_run_img_size(run_dir, img_size=img_size)
    if resolved_img_size is None:
        resolved_img_size = DL_LIB_ETALON_FIXED_SIZE

    return resolved_manifest, checkpoint, resolved_img_size


class RetinaNetImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], img_size: Tuple[int, int]):
        self.image_paths = image_paths
        self.img_size = img_size
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize(img_size),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        tensor = self.transforms(img)
        return tensor, str(path), (orig_w, orig_h)


def scale_boxes_to_original(
    boxes: torch.Tensor,
    model_size: Tuple[int, int],
    orig_size: Tuple[int, int],
) -> torch.Tensor:
    model_y, model_x = model_size
    orig_w, orig_h = orig_size
    scale_x = orig_w / model_x
    scale_y = orig_h / model_y
    scaled = boxes.clone()
    scaled[:, [0, 2]] *= scale_x
    scaled[:, [1, 3]] *= scale_y
    return scaled


def boxes_to_yolo_lines(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    orig_size: Tuple[int, int],
    score_thresh: float,
) -> List[str]:
    orig_w, orig_h = orig_size
    yolo_lines: List[str] = []

    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        if int(label) != RETINANET_CLASS_ID_ETALON or score < score_thresh:
            continue
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2
        cy = y1 + bh / 2
        yolo_lines.append(
            f"{YOLO_CLASS_ID_WIRED} {cx / orig_w:.6f} {cy / orig_h:.6f} {bw / orig_w:.6f} {bh / orig_h:.6f}"
        )
    return yolo_lines


def cvat_rel_dcm(image_path: Path, images_dir: Path, data_as_images_root: Path) -> str:
    try:
        rel = image_path.relative_to(data_as_images_root)
        return str(rel.parent).replace("\\", "/")
    except ValueError:
        rel = image_path.relative_to(images_dir)
        parent = rel.parent.as_posix()
        if parent == ".":
            return images_dir.name
        return f"{images_dir.name}/{parent}"


def create_cvat_zip(batch_data_dir: str, output_zip: str) -> None:
    print(f"Creating ZIP archive: {output_zip}")
    with zipfile.ZipFile(output_zip, "w") as zf:
        for root, _, files in os.walk(batch_data_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path_in_zip = os.path.relpath(abs_path, batch_data_dir)
                zf.write(abs_path, rel_path_in_zip)


def package_labels_for_cvat(
    labels_dir: Path,
    input_folder_name: str,
    processed_frames: List[Tuple[str, str]],
    output_zip: Path,
) -> None:
    temp_root = labels_dir.parent / "cvat_upload_temp"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    dest_labels_dir = temp_root / "labels" / "train" / CVAT_PREFIX / input_folder_name
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    for item in labels_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, dest_labels_dir / item.name)

    train_txt_path = temp_root / "train.txt"
    with open(train_txt_path, "w", encoding="utf-8") as f:
        for rel_dcm, frame_name in processed_frames:
            line = f"data/images/train/{CVAT_PREFIX}/{rel_dcm}/{frame_name}".replace("\\", "/")
            f.write(line + "\n")

    data_yaml_content = f"names:\n  {YOLO_CLASS_ID_WIRED}: wired\npath: .\ntrain: train.txt\n"
    with open(temp_root / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml_content)

    create_cvat_zip(str(temp_root), str(output_zip))
    shutil.rmtree(temp_root)


def run_large_diameter_predictions(
    run_dir: Path,
    images_dir: Path,
    anchor_manifest: Optional[Path] = None,
    img_size: Optional[Tuple[int, int]] = None,
    score_thresh: float = 0.5,
    nms_thresh: float = 0.3,
    batch_size: int = BATCH_SIZE_INFERENCE,
    output_root: Optional[Path] = None,
) -> Path:
    images_dir = images_dir.resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    anchor_manifest, checkpoint, img_size = resolve_inference_settings(run_dir, anchor_manifest, img_size)
    input_folder_name = images_dir.name

    data_dir = get_data_dir()
    output_root = output_root or (
        data_dir / "labeling" / "noisy_labels" / "large_diameter_model_preds" / input_folder_name
    )
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(images_dir)
    if not image_paths:
        raise RuntimeError(f"No PNG images found under {images_dir}")
    print(f"Found {len(image_paths)} images in {images_dir}")

    anchor_spec = anchor_spec_from_manifest_path(anchor_manifest)
    model = build_dl_lab_etalon_retinanet_from_trained_weights(
        anchor_spec=anchor_spec,
        trained_weights_path=str(checkpoint),
        img_size=img_size,
        device=DEVICE,
        freeze_layers=0,
    )
    configure_retinanet_postprocess(model, score_thresh=score_thresh, nms_thresh=nms_thresh)
    model.eval()
    print(f"Loaded RetinaNet from {checkpoint} (run dir: {run_dir.resolve()})")
    print(f"Anchor manifest: {anchor_manifest}")
    print(f"Image size (y, x): {img_size}")

    dataset = RetinaNetImageDataset(image_paths, img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    data_as_images_root = data_dir / "labeling" / "data_as_images"
    processed_frames: List[Tuple[str, str]] = []
    labels_written = 0

    with torch.no_grad():
        for images, paths, orig_sizes in tqdm(dataloader, desc="RetinaNet inference"):
            images_device = [img.to(DEVICE) for img in images]
            outputs = model(images_device)

            for output, path_str, orig_size in zip(outputs, paths, orig_sizes):
                image_path = Path(path_str)
                rel_inside_output = image_path.relative_to(images_dir)
                label_path = output_root / rel_inside_output.with_suffix(".txt")
                label_path.parent.mkdir(parents=True, exist_ok=True)

                boxes = scale_boxes_to_original(output["boxes"], img_size, orig_size)
                yolo_lines = boxes_to_yolo_lines(
                    boxes,
                    output["labels"],
                    output["scores"],
                    orig_size,
                    score_thresh=score_thresh,
                )

                if yolo_lines:
                    with open(label_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(yolo_lines))
                    labels_written += 1

                rel_dcm = cvat_rel_dcm(image_path, images_dir, data_as_images_root)
                processed_frames.append((rel_dcm, image_path.name))

    cvat_zip_path = output_root.parent / f"{input_folder_name}_cvat_upload.zip"
    package_labels_for_cvat(output_root, input_folder_name, processed_frames, cvat_zip_path)

    print(f"Noisy labels saved to {output_root}")
    print(f"Frames with detections: {labels_written} / {len(processed_frames)}")
    print(f"CVAT upload zip: {cvat_zip_path}")
    return cvat_zip_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RetinaNet noisy-label generation for CVAT.")
    parser.add_argument("--run-dir", type=str, default=CONFIG["run_dir"])
    parser.add_argument("--images-dir", type=str, default=CONFIG["images_dir"])
    parser.add_argument("--anchor-manifest", type=str, default=CONFIG["anchor_manifest"])
    parser.add_argument("--score-thresh", type=float, default=CONFIG["score_thresh"])
    parser.add_argument("--nms-thresh", type=float, default=CONFIG["nms_thresh"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.run_dir:
        raise ValueError("Set CONFIG['run_dir'] or pass --run-dir.")
    if not args.images_dir:
        raise ValueError("Set CONFIG['images_dir'] or pass --images-dir.")

    anchor_manifest = Path(args.anchor_manifest) if args.anchor_manifest else None
    run_large_diameter_predictions(
        run_dir=Path(args.run_dir),
        images_dir=Path(args.images_dir),
        anchor_manifest=anchor_manifest,
        img_size=CONFIG["img_size"],
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
