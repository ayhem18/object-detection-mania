import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set

from dl_lib.etalon_object_detection.modules.path_layout import (
    legacy_split_dir,
    resolve_latest_dataset_hash,
    split_metadata_dir,
    split_metadata_root,
)


def extract_dcm_folders(file_paths: List[str]) -> Set[str]:
    """Extracts the unique parent DCM folders from a list of frame paths."""
    return {os.path.dirname(path) for path in file_paths}


def _resolve_split_dir(dataset_hash: str, split_hash: str) -> Path | None:
    new_dir = split_metadata_dir(dataset_hash, split_hash)
    if (new_dir / "train_files.json").is_file():
        return new_dir
    old_dir = legacy_split_dir(dataset_hash, split_hash)
    if (old_dir / "train_files.json").is_file():
        return old_dir
    return None


def run_data_split_sanity_check(config: Dict[str, Any]) -> None:
    """Reads cached splits and verifies mutual exclusivity at the DCM level."""
    dataset_hash = config["dataset_hash"]
    splits_root = split_metadata_root(dataset_hash)

    if not splits_root.exists():
        legacy_root = legacy_split_dir(dataset_hash, "")
        legacy_parent = legacy_root.parent
        if legacy_parent.is_dir() and any(legacy_parent.iterdir()):
            splits_root = legacy_parent
        else:
            print(f"Error: No splits found in {splits_root}.")
            return

    split_hash = config.get("split_hash", "latest")
    if split_hash == "latest":
        subdirs = [d for d in splits_root.iterdir() if d.is_dir()]
        if not subdirs:
            print(f"Error: No split subdirectories found in {splits_root}.")
            return
        latest_split_dir = max(subdirs, key=os.path.getmtime)
        split_hash = latest_split_dir.name

    split_dir = _resolve_split_dir(dataset_hash, split_hash)
    if split_dir is None:
        print(f"Error: Split files not found for split hash {split_hash}.")
        return

    train_file = split_dir / "train_files.json"
    val_file = split_dir / "val_files.json"

    with open(train_file, "r", encoding="utf-8") as f:
        train_files = json.load(f)

    with open(val_file, "r", encoding="utf-8") as f:
        val_files = json.load(f)

    train_dcms = extract_dcm_folders(train_files)
    val_dcms = extract_dcm_folders(val_files)
    intersection = train_dcms.intersection(val_dcms)

    print("\n=======================================================")
    print("             DATA SPLIT SANITY CHECK                   ")
    print("=======================================================")
    print(f"Dataset Hash: {dataset_hash}")
    print(f"Split Hash:   {split_hash}")
    print(f"Split Dir:    {split_dir}")
    print(f"Total Train Files: {len(train_files):5d} | Unique Train DCMs: {len(train_dcms):4d}")
    print(f"Total Val Files:   {len(val_files):5d} | Unique Val DCMs:   {len(val_dcms):4d}")
    print("-------------------------------------------------------")

    if len(intersection) == 0:
        print("SUCCESS: No data leakage detected.")
        print("Train and Val sets contain mutually exclusive DCMs.")
    else:
        print(f"FAILURE: Found {len(intersection)} overlapping DCMs!")
        print("The following DCMs appear in BOTH splits:")
        for dcm in intersection:
            print(f"     - {dcm}")
        raise AssertionError("Data leakage detected between training and validation splits!")


def main() -> None:
    config: Dict[str, Any] = {
        "dataset_hash": "latest",
        "split_hash": "latest",
    }

    if config["dataset_hash"] == "latest":
        latest = resolve_latest_dataset_hash()
        if latest:
            config["dataset_hash"] = latest
            print(f"Using latest dataset hash: {config['dataset_hash']}")

    run_data_split_sanity_check(config)


if __name__ == "__main__":
    main()
