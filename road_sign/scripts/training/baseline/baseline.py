import os
import sys
import torch
import logging

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
# mypt is installed as a package, so we don't need to append its src directory to sys.path

from home_made_od.yolo_v2.modules.yolov2_model import YoloV2
from home_made_od.yolo_v2.modules.yolov2_loss import YoloV2Loss
from home_made_od.yolo_v2.modules.target_calculation import YoloV2TargetCalculator
from home_made_od.yolo_v2.modules.yolov2_train import run_training_loop
from home_made_od.yolo_v2.modules.anchors.anchor_utils import generate_anchors
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
    # Use the custom ResnetFE wrapper to build ResNet50
    # Freeze 3 out of 4 layers
    resnet_fe = ResnetFE(
        build_by_layer=True,
        num_extracted_layers=-1, # Extract all 4 layers
        num_extracted_bottlenecks=-1,
        freeze=3, # Freeze the first 3 layers
        freeze_by_layer=True,
        add_global_average=False,
        architecture=50
    )
    
    # The output of ResNet50 layer4 is 2048 channels
    backbone_out_channels = 2048 
    
    model = YoloV2(
        backbone=resnet_fe,
        backbone_out_channels=backbone_out_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        num_conv_blocks=2
    )
    
    # Return both model and the ImageNet transform expected by the weights
    return model, resnet_fe.transform

from dotenv import load_dotenv

def main():
    load_dotenv()
    
    # Set seed for reproducibility
    seed_everything(42)
    
    # --- Configuration ---
    DATA_DIR = os.path.join(road_sign_root, 'data_512')
    ARTIFACT_DIR = os.path.join(road_sign_root, 'artifacts', 'baseline')
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 256
    EPOCHS = 50
    NUM_CLASSES = 8
    NUM_ANCHORS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Prepare Data Pairs
    all_pairs = get_data_pairs(DATA_DIR)
    print(f"Found {len(all_pairs)} image-label pairs.")

    # 2. Split Data
    train_size = int(0.8 * len(all_pairs))
    val_size = len(all_pairs) - train_size
    train_pairs, val_pairs = random_split(all_pairs, [train_size, val_size])

    # 3. Anchors
    anchors = get_anchors([all_pairs[i] for i in train_pairs.indices], num_anchors=NUM_ANCHORS)

    # 4. Build Model, Loss, and Target Calculator
    model, imagenet_transform = build_model(NUM_CLASSES, NUM_ANCHORS)
    model = model.to(DEVICE)
    
    # Extract normalization mean/std from the backbone's expected transform
    mean = imagenet_transform.mean
    std = imagenet_transform.std

    # 5. Datasets & Dataloaders
    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.Resize(IMG_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std) # Added ImageNet Normalization
    ])
    
    val_transforms = v2.Compose([
        v2.Resize(IMG_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std) # Added ImageNet Normalization
    ])

    train_ds = YoloFormatDataset([all_pairs[i] for i in train_pairs.indices], IMG_SIZE, train_transforms)
    val_ds = YoloFormatDataset([all_pairs[i] for i in val_pairs.indices], IMG_SIZE, val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=yolov2_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=yolov2_collate_fn)
    
    # 512 / 32 = 16
    feature_map_shape = (IMG_SIZE[0] // 32, IMG_SIZE[1] // 32)
    
    target_calculator = YoloV2TargetCalculator(NUM_CLASSES, anchors, feature_map_shape)
    criterion = YoloV2Loss(NUM_CLASSES, NUM_ANCHORS)
    
    # Only pass trainable parameters to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

    from mypt.loggers import get_logger
    logger = get_logger('tensorboard', log_dir=ARTIFACT_DIR)
    
    # 6. Run Training
    run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        target_calculator=target_calculator,
        criterion=criterion,
        device=DEVICE,
        epochs=EPOCHS,
        artifact_dir=ARTIFACT_DIR,
        early_stop_patience=10,
        logger=logger,
    )

if __name__ == "__main__":
    main()
