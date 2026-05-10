import os
import sys
import json
import yaml
import torch
import hashlib

from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import DataLoader

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
while True:
    if 'road_sign' in os.listdir(current_dir) and os.path.isdir(os.path.join(current_dir, 'road_sign')):
        break
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:
        raise RuntimeError("Could not find 'road_sign' directory in the parent path.")
    current_dir = parent_dir

workspace_root = current_dir
road_sign_root = os.path.join(workspace_root, 'road_sign')

sys.path.insert(0, workspace_root)

from home_made_od.yolo_v2.modules.yolov2_model import YoloV2
from home_made_od.yolo_v2.modules.yolov2_loss import YoloV2Loss
from home_made_od.yolo_v2.modules.target_calculation import YoloV2TargetCalculator
from home_made_od.yolo_v2.modules.yolov2_train import run_training_loop
from road_sign.utils.data_utils import YoloFormatDataset, yolov2_collate_fn

from mypt.backbones.resnetFE import ResnetFE
from mypt.code_utils.pytorch_utils import seed_everything

def split_by_original_image(data_dir, train_ratio=0.9):
    img_dir = os.path.join(data_dir, 'train', 'images')
    label_dir = os.path.join(data_dir, 'labels', 'annotations')
    
    all_subdirs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    
    import random
    random.shuffle(all_subdirs)
    split_idx = int(len(all_subdirs) * train_ratio)
    
    train_subdirs = all_subdirs[:split_idx]
    val_subdirs = all_subdirs[split_idx:]
    
    def gather_pairs(subdirs):
        pairs = []
        for subdir_name in subdirs:
            img_subdir = os.path.join(img_dir, subdir_name)
            lbl_subdir = os.path.join(label_dir, subdir_name)
            if not os.path.exists(lbl_subdir): continue
            
            for img_file in os.listdir(img_subdir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    stem = Path(img_file).stem
                    label_file = os.path.join(lbl_subdir, f"{stem}.txt")
                    if os.path.exists(label_file):
                        pairs.append((os.path.join(img_subdir, img_file), label_file))
        return pairs
        
    return gather_pairs(train_subdirs), gather_pairs(val_subdirs)

def build_model(num_classes, num_anchors):
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

def get_config_hash(config_dict):
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

from dotenv import load_dotenv

# --- Default Configuration ---
DEFAULT_CONFIG = {
    "experiment_name": "patch_adaptive_resnet50",
    "data_dir": "data_patch_adaptive",
    "img_size": [512, 512],
    "batch_size": 128, 
    "epochs": 150,    
    "lr": 1e-3,       
    "num_classes": 8,
    "early_stop_patience": 20,
    "seed": 42
}

def main():
    load_dotenv()
    
    # 1. Config Management
    config = DEFAULT_CONFIG.copy()
    DATA_DIR = os.path.join(road_sign_root, config["data_dir"])

    # --- Load Patch Config generated during dataset preparation ---
    patch_config_path = os.path.join(DATA_DIR, "patch_config.yaml")
    if os.path.exists(patch_config_path):
        with open(patch_config_path, "r") as f:
            config["patch_config"] = yaml.safe_load(f)
        print(f"Merged patch configuration from {patch_config_path}")
    else:
        print(f"Warning: No patch_config.yaml found in {DATA_DIR}. Hashing might not be fully reproducible.")

    config_hash = get_config_hash(config)
    
    artifact_root = os.path.join(road_sign_root, 'artifacts', 'patch_based', config_hash)
    os.makedirs(artifact_root, exist_ok=True)
    
    # Save active config for reproducibility
    with open(os.path.join(artifact_root, "config.yaml"), "w") as f:
        yaml.dump(config, f)
        
    print(f"--- Experiment Hash: {config_hash} ---")
    print(f"Artifacts will be saved to: {artifact_root}")

    seed_everything(config["seed"])
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Anchors generated during dataset preparation ---
    anchors_path = os.path.join(DATA_DIR, "anchors.json")
    if os.path.exists(anchors_path):
        with open(anchors_path, "r") as f:
            anchors = json.load(f)["anchors"]
        print(f"Loaded anchors from {anchors_path}: {anchors}")
    else:
        raise FileNotFoundError(f"Anchors not found at {anchors_path}. Please run prepare_patch_ds.py first.")
    
    num_anchors = len(anchors)

    train_pairs, val_pairs = split_by_original_image(DATA_DIR, train_ratio=0.9)
    print(f"Found {len(train_pairs)} training patches and {len(val_pairs)} validation patches.")

    model, imagenet_transform = build_model(config["num_classes"], num_anchors)
    model = model.to(DEVICE)
    
    mean, std = imagenet_transform.mean, imagenet_transform.std

    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.Resize(config["img_size"]),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])
    
    val_transforms = v2.Compose([
        v2.Resize(config["img_size"]),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])

    train_ds = YoloFormatDataset(train_pairs, config["img_size"], train_transforms)
    val_ds = YoloFormatDataset(val_pairs, config["img_size"], val_transforms)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=yolov2_collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=yolov2_collate_fn, num_workers=2)
    
    feature_map_shape = (config["img_size"][0] // 32, config["img_size"][1] // 32)
    target_calculator = YoloV2TargetCalculator(config["num_classes"], anchors, feature_map_shape)
    criterion = YoloV2Loss(config["num_classes"], num_anchors)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config["lr"])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config["lr"], 
        steps_per_epoch=len(train_loader), 
        epochs=config["epochs"]
    )

    from mypt.loggers import get_logger
    logger = get_logger('tensorboard', log_dir=artifact_root)
    
    run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        target_calculator=target_calculator,
        criterion=criterion,
        device=DEVICE,
        epochs=config["epochs"],
        artifact_dir=artifact_root,
        early_stop_patience=config["early_stop_patience"],
        logger=logger,
        scheduler=scheduler,
        lr_update_level="batch"
    )

if __name__ == "__main__":
    main()
