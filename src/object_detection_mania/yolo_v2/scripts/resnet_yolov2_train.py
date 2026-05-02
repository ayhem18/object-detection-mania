import os
import sys
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from mypt.backbones.resnetFE import ResnetFE
from object_detection_mania.yolo_v2.modules.yolov2_ds import YoloV2Dataset, yolov2_collate_fn
from object_detection_mania.yolo_v2.modules.yolov2_model import YoloV2
from object_detection_mania.yolo_v2.modules.target_calculation import YoloV2TargetCalculator
from object_detection_mania.yolo_v2.modules.yolov2_loss import YoloV2Loss
from object_detection_mania.yolo_v2.modules.yolov2_train import run_training_loop

def main():
    # 1. Paths and Config Loading
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    artifacts_dir = project_root / "artifacts" / "yolo_v2_artifacts" / "synthetic"
    
    config_path = artifacts_dir / "yolo_configurations" / "resnet_yolov2.yaml"
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Check for dataset configurations
    ds_config_dir = artifacts_dir / "configs"
    if not ds_config_dir.exists():
        print(f"Dataset configurations directory not found: {ds_config_dir}")
        print("Please run src/object_detection_mania/data/synthetic_toy_ds/dataset_generator.py first.")
        return

    train_cfg_path = ds_config_dir / config['dataset']['train_config']
    val_cfg_path = ds_config_dir / config['dataset']['val_config']
    
    if not train_cfg_path.exists() or not val_cfg_path.exists():
        print(f"Required dataset configs not found in {ds_config_dir}")
        print("Please ensure you have generated the correct configs.")
        return

    # 3. Setup Data
    img_size = config['training']['img_size']
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((img_size, img_size))
    ])

    rendered_dir = artifacts_dir / "rendered"
    
    print("Initializing Datasets...")
    train_ds = YoloV2Dataset(
        config_path=str(train_cfg_path),
        output_dir=str(rendered_dir / "train"),
        img_shape=(img_size, img_size),
        transforms=transforms
    )
    
    val_ds = YoloV2Dataset(
        config_path=str(val_cfg_path),
        output_dir=str(rendered_dir / "val"),
        img_shape=(img_size, img_size),
        transforms=transforms
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        collate_fn=yolov2_collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        collate_fn=yolov2_collate_fn,
        num_workers=4
    )

    # 4. Setup Model
    print("Building Model...")
    # Initialize ResNetFE backbone
    backbone_fe = ResnetFE(
        architecture=config['model']['backbone_arch'],
        build_by_layer=True,
        num_extracted_layers=config['model']['num_extracted_layers'],
        num_extracted_bottlenecks=-1,
        freeze=False,
        freeze_by_layer=True,
        add_global_average=False
    )
    
    # The ResnetFE wraps the feature extractor in self._feature_extractor
    # but based on its implementation it should be accessible.
    # Actually ResnetFE inherits from WrapperLikeModuleMixin which might expose it.
    model = YoloV2(
        backbone=backbone_fe,
        backbone_out_channels=config['model']['backbone_out_channels'],
        num_anchors=config['model']['num_anchors'],
        num_classes=config['model']['num_classes'],
        num_conv_blocks=config['model']['num_conv_blocks']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 5. Setup Training Components
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    
    # Feature map shape depends on backbone downsampling (ResNet is 32x if 4 layers extracted)
    # 416 / 32 = 13
    f_map_size = img_size // 32
    target_calc = YoloV2TargetCalculator(
        num_classes=config['model']['num_classes'],
        anchors=config['model']['anchors'],
        feature_map_shape=(f_map_size, f_map_size)
    )
    
    criterion = YoloV2Loss(
        num_classes=config['model']['num_classes'],
        num_anchors=config['model']['num_anchors'],
        reg_loss_type=config['training']['reg_loss_type'],
        background_obj_coeff=config['training']['background_obj_coeff']
    )

    # 6. Run Training
    run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        target_calculator=target_calc,
        criterion=criterion,
        device=device,
        epochs=config['training']['epochs'],
        artifact_dir=str(artifacts_dir / "training_runs" / "resnet_yolov2"),
        early_stop_patience=config['training']['early_stop_patience']
    )

if __name__ == "__main__":
    main()
