import os
import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any

from home_made_od.yolo_family.yolov2.yolov2_model import YoloV2
from home_made_od.yolo_family.anchors.anchor_utils import generate_anchors
from mypt.backbones.resnetFE import ResnetFE

def get_config_hash(config_dict: Dict[str, Any]) -> str:
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def split_by_original_image(data_dir: str, train_ratio: float = 0.9, seed: int = 42) -> Tuple[List[str], List[str]]:
    img_dir = os.path.join(data_dir, 'train', 'images')
    
    # 1. Get and sort subdirectories (which represent original images)
    all_subdirs = sorted([d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))])
    
    # 2. Shuffle deterministically
    import random
    rng = random.Random(seed)
    rng.shuffle(all_subdirs)
    
    split_idx = int(len(all_subdirs) * train_ratio)
    
    train_subdirs = sorted(all_subdirs[:split_idx])
    val_subdirs = sorted(all_subdirs[split_idx:])
    
    return train_subdirs, val_subdirs

def gather_pairs_from_subdirs(data_dir: str, subdirs: List[str]) -> List[Tuple[str, str]]:
    img_dir = os.path.join(data_dir, 'train', 'images')
    label_dir = os.path.join(data_dir, 'labels', 'annotations')
    
    pairs = []
    for subdir_name in subdirs:
        img_subdir = os.path.join(img_dir, subdir_name)
        lbl_subdir = os.path.join(label_dir, subdir_name)
        if not os.path.exists(lbl_subdir): continue
        
        # Sort files to ensure deterministic order
        img_files = sorted(os.listdir(img_subdir))
        
        for img_file in img_files:
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                stem = Path(img_file).stem
                label_file = os.path.join(lbl_subdir, f"{stem}.txt")
                if os.path.exists(label_file):
                    pairs.append((os.path.join(img_subdir, img_file), label_file))
    return pairs

def prepare_data_and_anchors(data_dir: str, train_ratio: float, seed: int, num_anchors: int) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[List[float]], str]:
    """
    Splits the data, computes a stable hash for the split, generates anchors on the train set,
    and saves the configuration to a hash-specific folder.
    
    Returns:
        train_pairs, val_pairs, anchors, hash_dir
    """
    # 1. Get split subdirectories
    train_subdirs, val_subdirs = split_by_original_image(data_dir, train_ratio, seed)
    
    # 2. Create Split Config
    split_config = {
        "train_ratio": train_ratio,
        "seed": seed,
        "train_subdirs": train_subdirs,
        "val_subdirs": val_subdirs
    }
    
    # 3. Generate Hash and Directory
    split_hash = get_config_hash(split_config)
    hash_dir = os.path.join(data_dir, split_hash)
    os.makedirs(hash_dir, exist_ok=True)
    
    # 4. Save Split Config
    split_config_path = os.path.join(hash_dir, "split_config.json")
    with open(split_config_path, "w") as f:
        json.dump(split_config, f, indent=4)
        
    # 5. Gather Pairs
    train_pairs = gather_pairs_from_subdirs(data_dir, train_subdirs)
    val_pairs = gather_pairs_from_subdirs(data_dir, val_subdirs)
    
    # 6. Generate Anchors (only if not already generated for this exact split hash)
    anchors_path = os.path.join(hash_dir, "anchors.json")
    if os.path.exists(anchors_path):
        print(f"Loading cached anchors from {anchors_path}")
        with open(anchors_path, "r") as f:
            anchors_list = json.load(f)["anchors"]
    else:
        print("Extracting bounding boxes from training set for anchor generation...")
        wh_list = []
        for _, label_path in train_pairs:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 5:
                        w, h = float(parts[3]), float(parts[4])
                        wh_list.append((w, h))
        
        if wh_list:
            print("Running KMeans-IoU to generate anchors...")
            anchors_arr = generate_anchors(wh_list, num_anchors=num_anchors, seed=seed)
            anchors_list = anchors_arr.tolist()
            
            with open(anchors_path, "w") as f:
                json.dump({"anchors": anchors_list}, f, indent=4)
            print(f"Saved generated anchors to {anchors_path}")
        else:
            raise ValueError("No bounding boxes found in the training split to generate anchors.")
            
    return train_pairs, val_pairs, anchors_list, hash_dir

def build_model(num_classes: int, num_anchors: int):
    resnet_fe = ResnetFE(
        build_by_layer=True,
        num_extracted_layers=-1, 
        num_extracted_bottlenecks=-1,
        freeze=2, 
        freeze_by_layer=True,
        add_global_average=False,
        architecture=50
    )
    
    backbone_out_channels = 2048 
    
    model = YoloV2(
        backbone=resnet_fe,
        backbone_out_channels=backbone_out_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        num_conv_blocks=2
    )
    
    return model, resnet_fe.transform
