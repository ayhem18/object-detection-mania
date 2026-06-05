"""
Shared sliding-window patch extraction and full-image aggregation for inference.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import ops
from torchvision.transforms import v2
from tqdm import tqdm


@dataclass(frozen=True)
class PatchMeta:
    """One patch crop on a source image."""

    patch_id: str
    img_path: str
    x1: int
    y1: int
    x2: int
    y2: int
    scale: int
    orig_w: int
    orig_h: int

    @property
    def crop_width(self) -> int:
        return self.x2 - self.x1

    @property
    def crop_height(self) -> int:
        return self.y2 - self.y1


def generate_sliding_window_metadata(
    image_paths: Sequence[Path],
    patch_scales: Sequence[int],
    *,
    stride_ratio: float = 0.75,
) -> List[PatchMeta]:
    """Pre-compute patch windows for each image (same grid as YOLO patch inference)."""
    metadata: List[PatchMeta] = []
    scales = [int(s) for s in patch_scales]

    for img_path in tqdm(image_paths, desc="Generating sliding window metadata"):
        img_path_str = str(img_path)
        probe = cv2.imread(img_path_str)
        if probe is None:
            continue
        orig_h, orig_w = probe.shape[:2]

        for scale in scales:
            stride = int(scale * stride_ratio)
            y_starts = sorted(
                set(list(range(0, orig_h, stride)) + [max(0, orig_h - scale)])
            )
            x_starts = sorted(
                set(list(range(0, orig_w, stride)) + [max(0, orig_w - scale)])
            )

            for y in y_starts:
                for x in x_starts:
                    x2 = min(x + scale, orig_w)
                    y2 = min(y + scale, orig_h)
                    patch_id = f"{img_path.stem}_s{scale}_y{y}_x{x}"
                    metadata.append(
                        PatchMeta(
                            patch_id=patch_id,
                            img_path=img_path_str,
                            x1=x,
                            y1=y,
                            x2=x2,
                            y2=y2,
                            scale=scale,
                            orig_w=orig_w,
                            orig_h=orig_h,
                        )
                    )
    return metadata


def _extract_and_save_patches(task: Tuple[str, List[PatchMeta], str, Tuple[int, int]]) -> bool:
    img_path, patch_list, output_dir, target_size = task
    img = cv2.imread(img_path)
    if img is None:
        return False

    th, tw = target_size
    for meta in patch_list:
        patch = img[meta.y1 : meta.y2, meta.x1 : meta.x2]
        ph, pw = patch.shape[:2]
        if ph < meta.scale or pw < meta.scale:
            patch = cv2.copyMakeBorder(
                patch,
                0,
                meta.scale - ph,
                0,
                meta.scale - pw,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
        patch = cv2.resize(patch, (tw, th))
        out_path = Path(output_dir) / f"{meta.patch_id}.jpg"
        cv2.imwrite(str(out_path), patch)
    return True


def prepare_patches_offline(
    image_paths: Sequence[Path],
    patch_scales: Sequence[int],
    target_size: Tuple[int, int],
    tmp_patch_dir: Path,
    *,
    stride_ratio: float = 0.75,
    max_workers: int | None = None,
) -> List[PatchMeta]:
    """Extract patches to disk in parallel; return metadata for every patch."""
    import concurrent.futures

    if tmp_patch_dir.exists():
        shutil.rmtree(tmp_patch_dir)
    tmp_patch_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = generate_sliding_window_metadata(
        image_paths, patch_scales, stride_ratio=stride_ratio
    )
    if not all_metadata:
        return []

    by_image: Dict[str, List[PatchMeta]] = {}
    for meta in all_metadata:
        by_image.setdefault(meta.img_path, []).append(meta)

    print(f"Extracting {len(all_metadata)} patches using multiprocessing...")

    tasks = [
        (img_path, patches, str(tmp_patch_dir), target_size)
        for img_path, patches in by_image.items()
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(_extract_and_save_patches, tasks), total=len(tasks), desc="Extracting patches"))

    return all_metadata


class PreSlicedPatchDataset(Dataset):
    """Loads pre-saved patch JPEGs from disk."""

    def __init__(
        self,
        metadata: Sequence[PatchMeta],
        patch_dir: Path,
        transform: v2.Compose,
    ):
        self.metadata = list(metadata)
        self.patch_dir = patch_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        meta = self.metadata[idx]
        patch_path = self.patch_dir / f"{meta.patch_id}.jpg"
        img_bgr = cv2.imread(str(patch_path))
        if img_bgr is None:
            th, tw = 512, 512
            return torch.zeros(3, th, tw), idx

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), idx


def map_patch_boxes_to_image(
    boxes: torch.Tensor,
    meta: PatchMeta,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """Map XYXY boxes from patch tensor space to full-image coordinates."""
    if boxes.numel() == 0:
        return boxes

    th, tw = target_size
    sx = meta.crop_width / tw
    sy = meta.crop_height / th
    mapped = boxes.clone().float()
    mapped[:, 0] = mapped[:, 0] * sx + meta.x1
    mapped[:, 1] = mapped[:, 1] * sy + meta.y1
    mapped[:, 2] = mapped[:, 2] * sx + meta.x1
    mapped[:, 3] = mapped[:, 3] * sy + meta.y1
    mapped[:, 0].clamp_(0, meta.orig_w)
    mapped[:, 2].clamp_(0, meta.orig_w)
    mapped[:, 1].clamp_(0, meta.orig_h)
    mapped[:, 3].clamp_(0, meta.orig_h)
    return mapped


def global_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class-aware NMS on full-image boxes (offset trick per class)."""
    if boxes.numel() == 0:
        return np.array([]), np.array([]), np.array([])

    offsets = labels.float() * 100_000.0
    keep = ops.nms(boxes + offsets.unsqueeze(1), scores, iou_threshold)
    return (
        boxes[keep].cpu().numpy(),
        scores[keep].cpu().numpy(),
        labels[keep].cpu().numpy(),
    )


def collect_image_stems(image_paths: Sequence[Path]) -> List[str]:
    return [p.stem for p in image_paths]


def default_preprocess(target_size: Tuple[int, int]) -> v2.Compose:
    """Eval-style preprocessing (no resize — patches are already target_size)."""
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
