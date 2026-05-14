import os
import sys
import json
import yaml
import torch
import hashlib

from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split

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

# Add the workspace root to sys.path so 'road_sign' can be imported
sys.path.insert(0, workspace_root)

from home_made_od.yolo_family.yolov2.yolov2_model import YoloV2
from home_made_od.yolo_family.losses.yolov2_loss import YoloV2Loss
from home_made_od.yolo_family.target_calculation.single_scale_no_ignore import YoloV2TargetCalculator
from home_made_od.yolo_v2.modules.yolov2_train import run_training_loop
from home_made_od.yolo_family.anchors.anchor_utils import generate_anchors
from road_sign.utils.data_utils import YoloFormatDataset, yolov2_collate_fn

from mypt.backbones.resnetFE import ResnetFE
from mypt.code_utils.pytorch_utils import seed_everything

def get_data_pairs(data_dir):
    img_dir = os.path.join(data_dir, 'train', 'images')
    label_dir = os.path.join(data_dir, 'labels', 'annotations')
    
    pairs = []
    if not os.path.exists(img_dir):
        print(f"Warning: Image directory not found at {img_dir}")
        return pairs

    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            stem = Path(img_file).stem
            label_file = os.path.join(label_dir, f"{stem}.txt")
            if not os.path.exists(label_file):
                continue
            pairs.append((os.path.join(img_dir, img_file), label_file))
    return pairs

def get_anchors(data_pairs, num_anchors=5):
    print("Generating anchors from dataset...")
    wh_list = []
    for _, label_path in data_pairs:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    w, h = float(parts[3]), float(parts[4])
                    wh_list.append((w, h))
    anchors = generate_anchors(wh_list, num_anchors=num_anchors)
    print(f"Generated Anchors: {anchors.tolist()}")
    return anchors.tolist()

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
    """Generates a stable MD5 hash of the configuration dictionary."""
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

from dotenv import load_dotenv

# --- Default Configuration ---
DEFAULT_CONFIG = {
    "experiment_name": "baseline_resnet50",
    "data_dir": "data_512",
    "img_size": [512, 512],
    "batch_size": 64, 
    "epochs": 250,    
    "lr": 1e-3,       
    "num_classes": 8,
    "num_anchors": 5,
    "early_stop_patience": 20,
    "seed": 42
}

def main():
    load_dotenv()
    
    # 1. Config Management
    config = DEFAULT_CONFIG.copy()
    config_hash = get_config_hash(config)
    
    artifact_root = os.path.join(road_sign_root, 'artifacts', 'baseline', config_hash)
    os.makedirs(artifact_root, exist_ok=True)
    
    # Save active config for reproducibility
    with open(os.path.join(artifact_root, "config.yaml"), "w") as f:
        yaml.dump(config, f)
        
    print(f"--- Experiment Hash: {config_hash} ---")
    print(f"Artifacts will be saved to: {artifact_root}")

    seed_everything(config["seed"])
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Preparation
    DATA_PATH = os.path.join(road_sign_root, config["data_dir"])
    all_pairs = get_data_pairs(DATA_PATH)
    print(f"Found {len(all_pairs)} image-label pairs.")

    train_size = int(0.8 * len(all_pairs))
    val_size = len(all_pairs) - train_size
    train_pairs, val_pairs = random_split(all_pairs, [train_size, val_size])

    # 3. Dynamic Anchors
    anchors = get_anchors([all_pairs[i] for i in train_pairs.indices], num_anchors=config["num_anchors"])

    # 4. Build Model & Transforms
    model, imagenet_transform = build_model(config["num_classes"], config["num_anchors"])
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

    train_ds = YoloFormatDataset([all_pairs[i] for i in train_pairs.indices], config["img_size"], train_transforms)
    val_ds = YoloFormatDataset([all_pairs[i] for i in val_pairs.indices], config["img_size"], val_transforms)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=yolov2_collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=yolov2_collate_fn, num_workers=2)
    
    # 5. Training Components
    f_map_shape = (config["img_size"][0] // 32, config["img_size"][1] // 32)
    target_calculator = YoloV2TargetCalculator(config["num_classes"], anchors, f_map_shape)
    criterion = YoloV2Loss(config["num_classes"], config["num_anchors"])
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config["lr"])

    # OneCycleLR Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config["lr"], 
        steps_per_epoch=len(train_loader), 
        epochs=config["epochs"]
    )

    from mypt.loggers import get_logger
    logger = get_logger('tensorboard', log_dir=artifact_root)
    
    # 6. Run Training
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

