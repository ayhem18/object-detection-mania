from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from home_made_od.general.path_utils import (
    DATASET_VERSION_PATCH,
    RoadSignDatasetVersion,
    class_mapping_path,
    get_train_image_label_pairs,
    load_dataset_config,
    load_split_lists,
)

class YoloFormatDataset(Dataset):
    def __init__(self, 
                 data_pairs: List[Tuple[str, str]],
                 img_shape: Tuple[int, int],
                 transforms: Union[Callable, List[Callable]],
                 filter_fn: Optional[Callable] = None,
                 engine: Literal["cv2", "pil"] = "cv2"):
        """
        A PyTorch Dataset for YOLO-format data that forces a fixed output size.
        
        Args:
            data_pairs: List of tuples containing (image_path, label_path).
            img_shape: Target image dimensions (H, W) to enforce fixed sizes.
            transforms: torchvision v2 transforms.
            filter_fn: A callable taking (img_path, label_path) returning True if it should be included.
            engine: The image loading engine to use ('cv2' or 'pil').
        """
        if filter_fn is not None:
            self.data_pairs = [pair for pair in data_pairs if filter_fn(pair[0], pair[1])]
        else:
            self.data_pairs = data_pairs
            
        self.img_shape = img_shape
        self.engine = engine.lower()
        
        # Normalize transforms into a list
        if hasattr(transforms, 'transforms'):
            self.transforms_list = list(transforms.transforms)
        elif isinstance(transforms, (list, tuple)):
            self.transforms_list = list(transforms)
        else:
            self.transforms_list = [transforms]
            
        self._verify_and_append_transforms()
        self.transforms = v2.Compose(self.transforms_list)

    def _load_image(self, img_path: str) -> Union[Image.Image, np.ndarray]:
        """Loads image using the specified engine."""
        if self.engine == "pil":
            img = Image.open(img_path)
            return ImageOps.exif_transpose(img).convert("RGB")
        else:
            # OpenCV approach
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Failed to load image at {img_path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _verify_and_append_transforms(self):
        """Ensures Resize, ToImage, and ToDtype are present for fixed input size."""
        has_resize = any(isinstance(t, v2.Resize) for t in self.transforms_list)
        has_to_tensor = any(isinstance(t, (v2.ToImage, v2.PILToTensor)) for t in self.transforms_list)
        has_to_type = any(isinstance(t, v2.ToDtype) and getattr(t, 'dtype', None) == torch.float32 and getattr(t, 'scale', False) for t in self.transforms_list)
        
        if not has_resize:
            self.transforms_list.append(v2.Resize(self.img_shape))
        if not has_to_tensor:
            self.transforms_list.append(v2.ToImage())
        if not has_to_type:
            self.transforms_list.append(v2.ToDtype(torch.float32, scale=True))

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.data_pairs[idx]
        
        # Load image using the selected engine
        img = self._load_image(img_path)
        
        # Get dimensions (PIL uses .size, OpenCV/Numpy uses .shape)
        if hasattr(img, 'size'):
            # img.size can be either a tuple of an integer or a tuple of two integers
            if isinstance(img.size, int):
                w_orig = h_orig = img.size
            else:
                w_orig, h_orig = img.size
        else:
            h_orig, w_orig = img.shape[:2]
        
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = [float(x) for x in line.split()]
                    if len(parts) >= 5:
                        targets.append(parts[:5])
        
        if not targets:
            # Handle empty images
            img = self.transforms(img)
            return img, torch.zeros((0, 5))
        
        targets = torch.tensor(targets, dtype=torch.float32)
        cls_ids = targets[:, 0]
        bboxes = targets[:, 1:] # cx, cy, w, h (normalized)
        
        # YOLO bounding boxes are normalized to [0, 1].
        # torchvision v2 transforms expect absolute coordinates based on the original image size.
        # Avoid clone: create a new tensor with absolute coordinates
        bboxes_abs = torch.cat([
            (bboxes[:, 0] * w_orig).unsqueeze(1),
            (bboxes[:, 1] * h_orig).unsqueeze(1),
            (bboxes[:, 2] * w_orig).unsqueeze(1),
            (bboxes[:, 3] * h_orig).unsqueeze(1)
        ], dim=1)
        
        boxes_datapoint = tv_tensors.BoundingBoxes(
            bboxes_abs, 
            format="CXCYWH", 
            canvas_size=(h_orig, w_orig)
        )
        
        # Apply the transformations (including Resizing)
        img, boxes_transformed = self.transforms(img, boxes_datapoint)
        
        # After transforms, the image is a tensor. Get the new dimensions.
        _, h_new, w_new = img.shape
        
        # Convert the transformed absolute boxes back to normalized boxes
        # Avoid clone: create the output tensor directly
        bboxes_out = torch.cat([
            (boxes_transformed[:, 0] / w_new).unsqueeze(1),
            (boxes_transformed[:, 1] / h_new).unsqueeze(1),
            (boxes_transformed[:, 2] / w_new).unsqueeze(1),
            (boxes_transformed[:, 3] / h_new).unsqueeze(1)
        ], dim=1)
        
        # Recombine into [cls_id, cx, cy, w, h]
        final_targets = torch.cat([cls_ids.unsqueeze(1), bboxes_out], dim=1)
            
        return img, final_targets

def yolov2_collate_fn(batch):
    """
    Custom collate function to handle variable number of objects per image.
    Returns:
        images: (B, 3, H, W)
        targets: (Total_Objects, 6) -> [batch_idx, cls_id, cx, cy, w, h]
    """
    images, targets = zip(*batch)
    
    images = torch.stack(images, dim=0)
    
    all_targets = []
    for i, t in enumerate(targets):
        if t.shape[0] > 0:
            batch_idx = torch.full((t.shape[0], 1), i)
            t_with_idx = torch.cat([batch_idx, t], dim=1)
            all_targets.append(t_with_idx)
            
    if not all_targets:
        targets = torch.zeros((0, 6))
    else:
        targets = torch.cat(all_targets, dim=0)
        
    return images, targets


def load_road_sign_class_mapping(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
) -> Dict[int, str]:
    path = class_mapping_path(version, dataset_hash)
    if not path.is_file():
        config = load_dataset_config(version, dataset_hash)
        source_version = config.get("source_version", "original")
        source_hash = config.get("source_dataset_hash")
        if source_hash:
            path = class_mapping_path(source_version, source_hash)
    if not path.is_file():
        raise FileNotFoundError(
            f"class_mapping.json not found for {version}/{dataset_hash} "
            f"(also checked source dataset)."
        )
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def road_sign_num_classes(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
) -> int:
    """Foreground class count for detection heads (one label per sign type in the mapping)."""
    return len(load_road_sign_class_mapping(version, dataset_hash))


def _yolo_lines_to_xyxy(
    label_path: Path,
    image_width: int,
    image_height: int,
    label_id_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse YOLO ``cls cx cy w h`` (normalized) into XYXY tensors for RetinaNet."""
    boxes: list[list[float]] = []
    labels: list[int] = []

    if label_path.is_file():
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0])) + label_id_offset
                cx = float(parts[1]) * image_width
                cy = float(parts[2]) * image_height
                bw = float(parts[3]) * image_width
                bh = float(parts[4]) * image_height
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id)

    if boxes:
        box_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        label_tensor = torch.as_tensor(labels, dtype=torch.int64)
    else:
        box_tensor = torch.zeros((0, 4), dtype=torch.float32)
        label_tensor = torch.zeros((0,), dtype=torch.int64)

    return box_tensor, label_tensor


def _ensure_retinanet_transforms(
    target_size: Tuple[int, int],
    transformations: Optional[Callable],
) -> Callable:
    """Ensure Resize + ToImage + ToDtype for fixed-size RetinaNet inputs."""
    target_h, target_w = target_size
    resize_op = v2.Resize((target_h, target_w))

    if transformations is None:
        return v2.Compose([v2.ToImage(), resize_op, v2.ToDtype(torch.float32, scale=True)])

    if hasattr(transformations, "transforms"):
        transform_list = list(transformations.transforms)
    elif isinstance(transformations, (list, tuple)):
        transform_list = list(transformations)
    else:
        transform_list = [transformations]

    has_resize = any(isinstance(t, v2.Resize) for t in transform_list)
    if not has_resize:
        insert_idx = 0
        for i, t in enumerate(transform_list):
            if "ToImage" in type(t).__name__:
                insert_idx = i + 1
                break
        transform_list.insert(insert_idx, resize_op)

    has_to_image = any(isinstance(t, (v2.ToImage, v2.PILToTensor)) for t in transform_list)
    if not has_to_image:
        transform_list.insert(0, v2.ToImage())

    has_float = any(
        isinstance(t, v2.ToDtype)
        and getattr(t, "dtype", None) == torch.float32
        and getattr(t, "scale", False)
        for t in transform_list
    )
    if not has_float:
        transform_list.append(v2.ToDtype(torch.float32, scale=True))

    return v2.Compose(transform_list)


class RoadSignRetinaNetDataset(Dataset):
    """
    Road-sign dataset in torchvision RetinaNet format.

    Each ``__getitem__`` returns ``(image, target)`` where:

    - ``image`` is a ``float32`` CHW tensor in ``[0, 1]``
    - ``target`` is a dict with ``boxes`` (XYXY), ``labels``, ``image_id``, ``area``, ``iscrowd``

    YOLO label files use 0-based class ids; by default ``label_id_offset=1`` maps them to
    RetinaNet's 1-based foreground labels (background is implicit at 0).
    """

    def __init__(
        self,
        data_pairs: Sequence[Tuple[str, str]],
        target_size: Tuple[int, int],
        transformations: Optional[Callable] = None,
        label_id_offset: int = 1,
        sample_ids: Optional[Sequence[str]] = None,
    ):
        self.data_pairs = list(data_pairs)
        self.target_size = target_size
        self.label_id_offset = label_id_offset
        self.sample_ids = (
            list(sample_ids)
            if sample_ids is not None
            else [Path(img).stem for img, _ in self.data_pairs]
        )
        if len(self.sample_ids) != len(self.data_pairs):
            raise ValueError("sample_ids length must match data_pairs length.")

        self.transformations = _ensure_retinanet_transforms(target_size, transformations)

    @classmethod
    def from_split(
        cls,
        version: RoadSignDatasetVersion,
        dataset_hash: str,
        split_hash: str,
        split: Literal["train", "val"],
        target_size: Tuple[int, int],
        transformations: Optional[Callable] = None,
        label_id_offset: int = 1,
    ) -> RoadSignRetinaNetDataset:
        """
        Build a dataset from ``data/{version}/{dataset_hash}/splits/{split_hash}/``.
        """
        train_ids, val_ids, _ = load_split_lists(version, dataset_hash, split_hash)
        unit_ids = train_ids if split == "train" else val_ids
        pairs = get_train_image_label_pairs(version, dataset_hash, unit_ids)
        if version == DATASET_VERSION_PATCH:
            sample_ids = [
                f"{Path(img).parent.name}/{Path(img).stem}" for img, _ in pairs
            ]
        else:
            sample_ids = [Path(img).stem for img, _ in pairs]
        return cls(
            data_pairs=pairs,
            target_size=target_size,
            transformations=transformations,
            label_id_offset=label_id_offset,
            sample_ids=sample_ids,
        )

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_path, label_path = self.data_pairs[idx]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        box_tensor, label_tensor = _yolo_lines_to_xyxy(
            Path(label_path),
            orig_w,
            orig_h,
            self.label_id_offset,
        )

        target: Dict[str, Any] = {
            "boxes": tv_tensors.BoundingBoxes(
                box_tensor,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(orig_h, orig_w),
            ),
            "labels": label_tensor,
            "image_id": torch.tensor([idx]),
            "sample_id": self.sample_ids[idx],
        }

        if len(box_tensor) > 0:
            target["area"] = (box_tensor[:, 2] - box_tensor[:, 0]) * (
                box_tensor[:, 3] - box_tensor[:, 1]
            )
            target["iscrowd"] = torch.zeros((len(label_tensor),), dtype=torch.int64)
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        img, target = self.transformations(img, target)
        return img, target


def retinanet_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]],
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, Any], ...]]:
    """
    RetinaNet expects a list of images and a list of target dicts (not stacked tensors).
    """
    return tuple(zip(*batch))


def build_retinanet_dataloaders(
    version: RoadSignDatasetVersion,
    dataset_hash: str,
    split_hash: str,
    target_size: Tuple[int, int],
    batch_size: int,
    train_transforms: Optional[Callable] = None,
    val_transforms: Optional[Callable] = None,
    label_id_offset: int = 1,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, Any]]:
    """
    Convenience helper for training scripts: train/val loaders + dataset metadata.
    """
    from torch.utils.data import DataLoader

    train_ds = RoadSignRetinaNetDataset.from_split(
        version,
        dataset_hash,
        split_hash,
        split="train",
        target_size=target_size,
        transformations=train_transforms,
        label_id_offset=label_id_offset,
    )
    val_ds = RoadSignRetinaNetDataset.from_split(
        version,
        dataset_hash,
        split_hash,
        split="val",
        target_size=target_size,
        transformations=val_transforms,
        label_id_offset=label_id_offset,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=retinanet_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=retinanet_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    meta = {
        "dataset_version": version,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "target_size": list(target_size),
        "num_classes": road_sign_num_classes(version, dataset_hash),
        "class_mapping": load_road_sign_class_mapping(version, dataset_hash),
        "label_id_offset": label_id_offset,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "dataset_config": load_dataset_config(version, dataset_hash),
    }
    return train_loader, val_loader, meta
