import json
import os
import random
from pathlib import Path

from mypt.code_utils.pytorch_utils import seed_everything
from object_detection_mania.data.synthetic_toy_ds.scene_generation import generate_scene_parameters

def generate_dataset_configs(
    output_dir: str,
    num_train: int = 1000,
    num_val: int = 200,
    num_test: int = 100,
    seed: int = 42
):
    """
    Generates JSON configuration files for train, val, and test splits.
    These configs contain the parameters needed to render scenes deterministically.
    """
    seed_everything(seed=seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test
    }
    
    global_idx = 0
    for split_name, count in splits.items():
        print(f"Generating config for {split_name} split ({count} samples)...")
        configs = []
        for i in range(count):
            sample_seed = seed + global_idx
            params = generate_scene_parameters(seed=sample_seed)
            # Add seed to params for triangle orientation in render_scene
            params['seed'] = sample_seed
            params['id'] = global_idx
            configs.append(params)
            global_idx += 1
            
        config_file = output_path / f"{split_name}_{count}_{seed}_config.json"
        with open(config_file, 'w') as f:
            json.dump(configs, f, indent=4)
        print(f"  Saved to {config_file}")

if __name__ == "__main__":
    NUM_TRAIN = 10000
    NUM_VAL = 2000
    NUM_TEST = 1000

    # Default location for configs
    script_dir = Path(__file__).resolve().parent
    current_dir = script_dir
    while "artifacts" not in os.listdir(str(current_dir)):
        current_dir = current_dir.parent

    yolov2_artifacts_dir = current_dir / "artifacts" / "yolo_v2_artifacts" / "synthetic" / "configs"
    yolov2_artifacts_dir.mkdir(parents=True, exist_ok=True)
    generate_dataset_configs(output_dir=str(yolov2_artifacts_dir), num_train=NUM_TRAIN, num_val=NUM_VAL, num_test=NUM_TEST)
