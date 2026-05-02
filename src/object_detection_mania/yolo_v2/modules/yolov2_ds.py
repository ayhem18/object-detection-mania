import os
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Any, Union, Callable
from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors

from object_detection_mania.data.synthetic_toy_ds.visualize_scenes import render_scene

class YoloV2Dataset(Dataset):
    def __init__(self, 
                 config_path: str, 
                 output_dir: str, 
                 img_shape: Tuple[int, int],
                 transforms: Union[Callable, List[Callable]],
                 force_render: bool = False):
        """
        YOLOv2 Dataset that renders and saves data on initialization.
        
        Args:
            config_path (str): Path to the JSON configuration file.
            output_dir (str): Directory where rendered images and labels will be saved.
            img_shape (Tuple[int, int]): Target image dimensions (H, W).
            transforms (Callable): Transformations to apply to the image.
            force_render (bool): If True, re-renders even if files exist.
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.img_shape = img_shape
        
        # Ensure transforms is a list for easy manipulation during verification
        if hasattr(transforms, 'transforms'):
            self.transforms_list = list(transforms.transforms)
        elif isinstance(transforms, (list, tuple)):
            self.transforms_list = list(transforms)
        else:
            self.transforms_list = [transforms]
            
        self._verify_and_append_transforms()
        self.transforms = v2.Compose(self.transforms_list)
        
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'r') as f:
            self.configs = json.load(f)
            
        self.img_map = {}
        self.label_map = {}
        
        self._prepare_data(force_render)
        
    def _verify_and_append_transforms(self):
        """Ensures Resize, ToImage (or ToTensor), and ToDtype are present."""
        has_resize = any(isinstance(t, v2.Resize) for t in self.transforms_list)
        # Check for any tensor conversion (v2.ToImage, v2.Compose with ToImage, or legacy ToTensor)
        has_to_tensor = any(isinstance(t, (v2.ToImage, v2.PILToTensor)) for t in self.transforms_list)
        # Check for ToDtype with float32 and scale=True
        has_to_type = any(isinstance(t, v2.ToDtype) and t.dtype == torch.float32 and t.scale for t in self.transforms_list)
        
        if not has_resize:
            print(f"Adding missing Resize({self.img_shape}) to transforms.")
            self.transforms_list.append(v2.Resize(self.img_shape))
            
        if not has_to_tensor:
            print("Adding missing ToImage() to transforms.")
            self.transforms_list.append(v2.ToImage())
            
        if not has_to_type:
            print("Adding missing ToDtype(torch.float32, scale=True) to transforms.")
            self.transforms_list.append(v2.ToDtype(torch.float32, scale=True))

    def _prepare_data(self, force_render: bool):
        """Renders and saves all images and labels defined in the config."""
        print(f"Preparing dataset in {self.output_dir}...")
        
        for i, config in enumerate(self.configs):
            sample_id = config.get('id', i)
            img_filename = f"sample_{sample_id:06d}.png"
            label_filename = f"sample_{sample_id:06d}.txt"
            
            img_path = self.images_dir / img_filename
            label_path = self.labels_dir / label_filename
            
            # Store in maps for constant time access
            self.img_map[i] = str(img_path)
            self.label_map[i] = str(label_path)
            
            # Check if rendering is needed
            if force_render or not img_path.exists() or not label_path.exists():
                config_objects = {int(k): v for k, v in config['objects'].items()}
                render_params = config.copy()
                render_params['objects'] = config_objects
                
                img = render_scene(render_params, config['seed'], self.img_shape)
                
                # Save Image (BGR for OpenCV consistency on disk)
                cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
                # Save Labels (YOLO format: cls, cx, cy, w, h)
                with open(label_path, 'w') as f:
                    for obj_idx, obj in config_objects.items():
                        cls_id, _, cx, cy, w, h = obj
                        f.write(f"{cls_id} {cx} {cy} {w} {h}\n")
                        
        print(f"Dataset preparation complete. {len(self.configs)} samples mapped.")

    def __len__(self) -> int:
        return len(self.configs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image (torch.Tensor): Transformed image tensor
            targets (torch.Tensor): (N, 5) -> [cls_id, cx, cy, w, h]
        """
        img_path = self.img_map[idx]
        label_path = self.label_map[idx]
        
        # Load as PIL Image
        img = Image.open(img_path).convert("RGB")
        
        # Load labels
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # YOLO: cls, cx, cy, w, h
                    parts = [float(x) for x in line.split()]
                    targets.append(parts)
        
        if not targets:
            # Handle empty images
            img = self.transforms(img)
            return img, torch.zeros((0, 5))
        
        targets = torch.tensor(targets)
        cls_ids = targets[:, 0]
        bboxes = targets[:, 1:] # cx, cy, w, h (normalized)
        
        # Wrap in v2 Datapoints
        # We use format CXCYWH and canvas_size for normalized coordinates to be handled correctly
        # Note: v2.BoundingBoxes expects unnormalized coordinates if canvas_size is not used for some transforms,
        # but for many v2 transforms, passing canvas_size and normalized bboxes works.
        # However, to be safe with all v2 transforms, we convert to absolute, transform, then back to normalized.
        h_orig, w_orig = self.img_shape
        bboxes_abs = bboxes.clone()
        bboxes_abs[:, [0, 2]] *= w_orig
        bboxes_abs[:, [1, 3]] *= h_orig
        
        boxes_datapoint = tv_tensors.BoundingBoxes(
            bboxes_abs, 
            format="CXCYWH", 
            canvas_size=self.img_shape
        )
        
        # Apply transforms to both
        img, boxes_transformed = self.transforms(img, boxes_datapoint)
        
        # Convert boxes back to normalized
        _, h_new, w_new = img.shape # img is now a tensor
        bboxes_out = boxes_transformed.clone()
        bboxes_out[:, [0, 2]] /= w_new
        bboxes_out[:, [1, 3]] /= h_new
        
        # Reconstruct targets [cls, cx, cy, w, h]
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
