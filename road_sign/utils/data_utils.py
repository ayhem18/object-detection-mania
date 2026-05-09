import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from torchvision import tv_tensors
from torch.utils.data import Dataset
from torchvision.transforms import v2
from typing import Tuple, List, Union, Callable, Optional, Literal

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
