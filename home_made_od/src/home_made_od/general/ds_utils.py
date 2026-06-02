import os
import json
import torch
import random
import hashlib
import numpy as np

from PIL import Image
from pathlib import Path
from torchvision import tv_tensors
from torch.utils.data import Dataset
from typing import List, Optional, Callable, Dict, Tuple, Any


from torchvision.transforms import v2
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, Tuple, Any


DL_LIB_ETALON_FIXED_SIZE = (960, 2400) # (y_dim, x_dim)

class WeldingDetectionDataset(Dataset):
    """
    Dataset for Object Detection in welding images.
    Loads PIL images and handles Pascal VOC [x1, y1, x2, y2] labels.
    """
    def __init__(
        self,
        images_dir: str,
        cache_dir: str,
        target_size: Tuple[int, int],
        initial_2_ds_cls_ids: Optional[Dict[int, int]] = None,
        transformations: Optional[Callable] = None,
        path_filter: Optional[Callable[[str], bool]] = None
    ):
        """
        Args:
            images_dir: Path to 'data_as_images' containing the PNG frames.
            cache_dir: Path to the 'obj_det/cache' directory containing master_labels.json.
            target_size: Fixed size to resize images to (y_dim, x_dim).
            initial_2_ds_cls_ids: Dict mapping original annotation IDs to training class IDs.
            transformations: A function/transform that takes (image, target) and returns (image, target).
            path_filter: A callable that accepts an img_path (str) and returns True to keep the sample.
        """
        self.images_dir = Path(images_dir)
        self.cache_dir = Path(cache_dir)
        self.target_size = target_size # (H, W)
        self.initial_2_ds_cls_ids = initial_2_ds_cls_ids or {}
        
        # Ensure Resize is present in transforms
        self.transformations = self._ensure_resize(transformations)
        
        master_json_path = self.cache_dir / "master_labels.json"
        if not master_json_path.exists():
            raise FileNotFoundError(f"Master index not found at {master_json_path}. Run obj_det_prepare.py first.")
            
        with open(master_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.samples = data.get("samples", [])
        
        # Apply Path Filtering
        if path_filter:
            self.samples = [s for s in self.samples if path_filter(s['img_path'])]
            
        print(f"WeldingDetectionDataset initialized with {len(self.samples)} samples. Target size: {self.target_size}")

    def _ensure_resize(self, transformations: Optional[Callable]) -> Callable:
        """
        Heuristic to ensure v2.Resize is in the transforms list.
        If not found, adds it after ToImage or at the beginning.
        """
        target_y_dim, target_x_dim = self.target_size
        resize_op = v2.Resize((target_y_dim, target_x_dim))

        if transformations is None or (isinstance(transformations, (Tuple, List)) and len(transformations) == 0):
            return v2.Compose([v2.ToImage(), resize_op, v2.ToDtype(torch.float32, scale=True)])

        # If it's a Compose, check its members
        if hasattr(transformations, 'transforms'):
            has_resize = any(isinstance(t, v2.Resize) for t in transformations.transforms)
            if has_resize:
                return transformations.transforms
            
            # Not found, insert after ToImage if possible
            new_list = list(transformations.transforms)
            insert_idx = 0
            for i, t in enumerate(new_list):
                if "ToImage" in str(type(t)):
                    insert_idx = i + 1
                    break
            new_list.insert(insert_idx, resize_op)
            return v2.Compose(new_list)
        
        # Fallback: wrap in Compose
        return v2.Compose([transformations, resize_op])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        
        # 1. Load Image as PIL
        img_path = self.images_dir / sample['img_path']
        img = Image.open(img_path).convert("RGB")
        # PIL.Image.size returns (width, height) i.e. (x_dim, y_dim)
        orig_x_dim, orig_y_dim = img.size
        
        # 2. Load Labels from Cache (Assumed format: Pascal [x1, y1, x2, y2])
        lbl_path = self.cache_dir / sample['lbl_path']
        with open(lbl_path, 'r', encoding='utf-8') as f:
            lbl_data = json.load(f)
            
        boxes_raw = np.array(lbl_data['boxes'], dtype=np.float32).reshape(-1, 4)
        
        original_classes = lbl_data['classes']
        mapped_classes = [self.initial_2_ds_cls_ids.get(c, c) for c in original_classes]
                    
        # Convert to Tensors (Original coordinates)
        target_boxes = torch.as_tensor(boxes_raw, dtype=torch.float32)
        target_labels = torch.as_tensor(mapped_classes, dtype=torch.int64)
        
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                target_boxes, 
                format=tv_tensors.BoundingBoxFormat.XYXY, 
                canvas_size=(orig_y_dim, orig_x_dim)
            ),
            "labels": target_labels,
            "image_id": torch.tensor([idx]),
            "area": (target_boxes[:, 3] - target_boxes[:, 1]) * (target_boxes[:, 2] - target_boxes[:, 0]) if len(target_boxes) > 0 else torch.zeros((0,)),
            "iscrowd": torch.zeros((len(target_labels),), dtype=torch.int64)
        }
        
        # 3. Transform (Resize, ToDtype, etc.)
        # tv_tensors will be automatically resized here
        if self.transformations is not None:
            img, target = self.transformations(img, target)
            
        return img, target



def collate_fn(batch):
    return tuple(zip(*batch))


def get_path_split_callables(
    master_json_path: str,
    val_ratio: float = 0.15,
    seed: int = 42,
    save_dir: Optional[str] = None) -> Tuple[Callable[[str], bool], Callable[[str], bool], str]:
    """
    Groups samples by DCM folder, performs a random split of DCMs, 
    hashes the split, caches the lists, and returns callables for Dataset filtering 
    along with the split hash.
    """
    with open(master_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = data["samples"]
    
    # 1. Group by DCM Folder to prevent leakage
    dcm_to_frames = {}
    
    for s in samples:
        path = s['img_path']
        # images are saved as labeling/data_as_images/dcm_name/frame_xxx.png
        # or batch_xxxx/dcm_name/frame_xxx.png
        # We use the parent folder of the frame as the grouping key
        dcm_folder = str(Path(path).parent)
        if dcm_folder not in dcm_to_frames:
            dcm_to_frames[dcm_folder] = []
        dcm_to_frames[dcm_folder].append(path)

    # 2. Shuffle and Split DCMs
    dcm_list = sorted(list(dcm_to_frames.keys()))
    random.seed(seed)
    random.shuffle(dcm_list)
    
    val_size = int(len(dcm_list) * val_ratio)
    val_dcm_list = sorted(dcm_list[:val_size])
    train_dcm_list = sorted(dcm_list[val_size:])
    
    # 3. Generate Hash
    hash_str = "train__" + "|".join(train_dcm_list) + "__val__" + "|".join(val_dcm_list)
    split_hash = hashlib.md5(hash_str.encode('utf-8')).hexdigest()
    
    # 4. Flatten back to file lists
    train_files = []
    for dcm in train_dcm_list:
        train_files.extend(dcm_to_frames[dcm])
        
    val_files = []
    for dcm in val_dcm_list:
        val_files.extend(dcm_to_frames[dcm])
        
    # 5. Cache the split for transparency
    if save_dir is not None:
        from dl_lib.etalon_object_detection.modules.path_layout import (
            dataset_hash_from_master_labels,
            split_metadata_dir,
        )

        dataset_hash = dataset_hash_from_master_labels(master_json_path)
        split_save_dir = split_metadata_dir(dataset_hash, split_hash)
        split_save_dir.mkdir(parents=True, exist_ok=True)
        with open(split_save_dir / "train_files.json", "w", encoding="utf-8") as f:
            json.dump(sorted(train_files), f, indent=4, ensure_ascii=False)
        with open(split_save_dir / "val_files.json", "w", encoding="utf-8") as f:
            json.dump(sorted(val_files), f, indent=4, ensure_ascii=False)
        print(
            f"Split cached in {split_save_dir}. Hash: {split_hash}. "
            f"Train: {len(train_files)} files, Val: {len(val_files)} files."
        )

    # 6. Create Callables
    train_set = set(train_files)
    val_set = set(val_files)
    
    return lambda p: p in train_set, lambda p: p in val_set, split_hash
