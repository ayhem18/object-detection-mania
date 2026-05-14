import os
import sys
import json
import yaml
import torch

from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from dotenv import load_dotenv

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

# --- Custom Imports ---
from mypt.code_utils.pytorch_utils import seed_everything
from home_made_od.yolo_family.yolov2.yolov2_train import run_training_loop
from home_made_od.yolo_family.losses.yolov2_loss import SSIgnoreMaskYoloV2Loss
from home_made_od.yolo_family.target_calculation.ss_ignore_mask import SingleScaleIgnoreTargetCalculator

from road_sign.utils.data_utils import YoloFormatDataset, yolov2_collate_fn
from road_sign.scripts.training.patch_based.train_scripts.train_utils import build_model, get_config_hash, prepare_data_and_anchors

# --- Default Configuration ---
DEFAULT_CONFIG = {
    "experiment_name": "patch_adaptive_resnet50_v2",
    "data_dir": "data_patch_adaptive",
    "img_size": [512, 512],
    "batch_size": 128, 
    "epochs": 150,    
    "lr": 1e-3,       
    "num_classes": 8,
    "num_anchors": 5,
    "train_ratio": 0.9,
    "early_stop_patience": 20,
    "seed": 42,
    
    # --- New v2 Hyperparameters ---
    "iou_threshold_ignore_anchor": 0.5,
    
    "loss_weights": {
        "lambda_coord": 5.0,
        "lambda_obj": 1.0,
        "lambda_cls": 1.0,
        "background_obj_coeff": 0.25 # Lowered heavily because we use mean reduction now
    },
    
    "augmentations": {
        "hflip_p": 0.5,
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2
        }
    }
}

def build_transforms(config, mean, std):
    """Dynamically builds torchvision transforms based on the configuration."""
    aug_cfg = config.get("augmentations", {})
    
    train_tfms = []
    if "hflip_p" in aug_cfg:
        train_tfms.append(v2.RandomHorizontalFlip(p=aug_cfg["hflip_p"]))
        
    if "color_jitter" in aug_cfg:
        cj = aug_cfg["color_jitter"]
        train_tfms.append(v2.ColorJitter(brightness=cj.get("brightness", 0.0), contrast=cj.get("contrast", 0.0)))
        
    # Standard mandatory ops for YOLOv2 (for both Train and Val)
    common_tfms = [
        v2.Resize(config["img_size"]),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ]
    
    train_tfms.extend(common_tfms)
    val_tfms = common_tfms.copy()
    
    return v2.Compose(train_tfms), v2.Compose(val_tfms)

def build_loss_and_target_calculator(config, anchors, feature_map_shape):
    """Instantiates the specific target calculator and loss criterion defined by the config."""
    
    target_calculator = SingleScaleIgnoreTargetCalculator(
        num_classes=config["num_classes"],
        anchors=anchors,
        feature_map_shape=feature_map_shape,
        iou_threshold_ignore_anchor=config["iou_threshold_ignore_anchor"]
    )
    
    loss_weights = config["loss_weights"]
    criterion = SSIgnoreMaskYoloV2Loss(
        num_classes=config["num_classes"],
        num_anchors=config["num_anchors"],
        reg_loss_type="mse",
        background_obj_coeff=loss_weights["background_obj_coeff"],
        lambda_coord=loss_weights["lambda_coord"],
        lambda_obj=loss_weights["lambda_obj"],
        lambda_cls=loss_weights["lambda_cls"]
    )
    
    return target_calculator, criterion

def main():
    load_dotenv()
    
    # 1. Config Management
    config = DEFAULT_CONFIG.copy()
    DATA_DIR = os.path.join(road_sign_root, config["data_dir"])

    # Load Patch Config
    patch_config_path = os.path.join(DATA_DIR, "patch_config.yaml")
    if os.path.exists(patch_config_path):
        with open(patch_config_path, "r") as f:
            config["patch_config"] = yaml.safe_load(f)
        print(f"Merged patch configuration from {patch_config_path}")

    seed_everything(config["seed"])
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Splitting, Hashing, and Anchor Generation
    train_pairs, val_pairs, anchors, split_hash_dir = prepare_data_and_anchors(
        data_dir=DATA_DIR, 
        train_ratio=config["train_ratio"], 
        seed=config["seed"], 
        num_anchors=config["num_anchors"]
    )
    
    print(f"Found {len(train_pairs)} training patches and {len(val_pairs)} validation patches.")
    config["anchors"] = anchors
    config["split_hash_dir"] = os.path.basename(split_hash_dir)
    
    # 3. Experiment Hashing & Artifact Directory
    config_hash = get_config_hash(config)
    artifact_root = os.path.join(road_sign_root, 'artifacts', 'patch_based', config_hash)
    os.makedirs(artifact_root, exist_ok=True)
    
    # Save active config for reproducibility
    with open(os.path.join(artifact_root, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    print(f"--- Experiment Hash (v2): {config_hash} ---")
    print(f"Artifacts will be saved to: {artifact_root}")

    # 4. Model Instantiation
    model, imagenet_transform = build_model(config["num_classes"], config["num_anchors"])
    model = model.to(DEVICE)
    
    # 5. Transforms and Dataloaders
    train_transforms, val_transforms = build_transforms(config, imagenet_transform.mean, imagenet_transform.std)

    train_ds = YoloFormatDataset(train_pairs, config["img_size"], train_transforms)
    val_ds = YoloFormatDataset(val_pairs, config["img_size"], val_transforms)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=yolov2_collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=yolov2_collate_fn, num_workers=2)
    
    # 6. Loss and Target Calculator
    # Using stride of 32 for ResNet50 default backbone
    feature_map_shape = (config["img_size"][0] // 32, config["img_size"][1] // 32)
    target_calculator, criterion = build_loss_and_target_calculator(config, anchors, feature_map_shape)
    
    # 7. Optimizer and Scheduler
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
    
    # 8. Start Training
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
