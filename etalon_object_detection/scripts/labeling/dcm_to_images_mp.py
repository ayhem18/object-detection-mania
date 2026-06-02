import argparse
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from dl_lib.common_tools.dcm_utils import clean_name, get_dcm_files
from dl_lib.common_tools.path_utils import get_data_dir
from dl_lib.etalon_object_detection.modules.path_layout import get_data_maps
from dl_lib.etalon_object_detection.scripts.labeling.initial_data_extraction import (
    get_frames_digital_lab,
    imwrite_unicode,
)
from dl_lib.preprocessing.modules.utils.leveling_wrapper_filter import leveling_defect_filter_u8

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("dl_lib.common_tools.data_extraction.data_extractor").setLevel(logging.WARNING)


@dataclass(frozen=True)
class DcmExtractionTask:
    dcm_file: str
    file_dest_dir: str
    dcm_key: str
    rel_path: str
    cleaned_name: str
    target_batch: str
    skip_cached: bool


@dataclass(frozen=True)
class DcmExtractionResult:
    dcm_key: str
    cleaned_name: str
    rel_path: str
    target_batch: str


def _process_single_dcm(task: DcmExtractionTask) -> Optional[DcmExtractionResult]:
    file_dest_dir = Path(task.file_dest_dir)

    if task.skip_cached and file_dest_dir.exists() and any(file_dest_dir.iterdir()):
        return DcmExtractionResult(
            dcm_key=task.dcm_key,
            cleaned_name=task.cleaned_name,
            rel_path=task.rel_path,
            target_batch=task.target_batch,
        )

    try:
        frames = get_frames_digital_lab(task.dcm_file)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", task.dcm_file, exc)
        return None

    if not frames:
        logger.warning("No frames extracted from %s", task.dcm_file)
        return None

    file_dest_dir.mkdir(parents=True, exist_ok=True)
    file_name = Path(task.dcm_file).stem

    for i, frame in enumerate(frames):
        frame_idx = i + 1
        if frame.dtype != np.uint16:
            frame = frame.astype(np.uint16)

        try:
            proc_frame_u8 = leveling_defect_filter_u8(frame)
        except Exception as exc:
            logger.warning("Error preprocessing %s frame %d: %s", file_name, frame_idx, exc)
            continue

        save_path = file_dest_dir / f"frame_{frame_idx:03d}.png"
        if not imwrite_unicode(str(save_path), proc_frame_u8):
            logger.warning("Failed to write %s", save_path)

    return DcmExtractionResult(
        dcm_key=task.dcm_key,
        cleaned_name=task.cleaned_name,
        rel_path=task.rel_path,
        target_batch=task.target_batch,
    )


def _build_tasks(
    data_maps: Dict[str, str],
    dest_root: Path,
    skip_cached: bool,
) -> List[DcmExtractionTask]:
    tasks: List[DcmExtractionTask] = []

    for source_dir, target_batch in data_maps.items():
        batch_dir = dest_root / target_batch
        batch_dir.mkdir(parents=True, exist_ok=True)

        dcm_files = sorted(get_dcm_files(source_dir))
        print(f"Found {len(dcm_files)} DCM files for extraction at {source_dir} -> {target_batch}.")

        for dcm_file in dcm_files:
            file_name = Path(dcm_file).stem
            cleaned_name = clean_name(file_name)
            tasks.append(
                DcmExtractionTask(
                    dcm_file=str(dcm_file),
                    file_dest_dir=str(batch_dir / cleaned_name),
                    dcm_key=str(dcm_file),
                    rel_path=f"{target_batch}/{cleaned_name}",
                    cleaned_name=cleaned_name,
                    target_batch=target_batch,
                    skip_cached=skip_cached,
                )
            )

    return tasks


def extract_and_index_data_mp(
    location: str = "local",
    num_workers: Optional[int] = None,
    skip_cached: bool = False,
) -> Dict[str, str]:
    """
    Multiprocess variant of ``extract_and_index_data``.

    Each DCM file is processed in a separate worker process.
    """
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1) - 1)

    data_maps = get_data_maps(location)
    dest_root = get_data_dir() / "labeling" / "data_as_images"
    dest_root.mkdir(parents=True, exist_ok=True)

    tasks = _build_tasks(data_maps, dest_root, skip_cached=skip_cached)
    if not tasks:
        print("No DCM files found.")
        return {}

    print(f"Processing {len(tasks)} DCM files with {num_workers} workers.")

    batch_indices: Dict[str, Dict[str, str]] = defaultdict(dict)
    combined_index: Dict[str, str] = {}
    failed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_single_dcm, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting DCMs"):
            result = future.result()
            if result is None:
                failed += 1
                continue

            batch_indices[result.target_batch][result.dcm_key] = result.cleaned_name
            combined_index[result.dcm_key] = result.rel_path

    for target_batch, batch_index in batch_indices.items():
        batch_index_path = dest_root / target_batch / "index.json"
        with open(batch_index_path, "w", encoding="utf-8") as f:
            json.dump(batch_index, f, indent=4, ensure_ascii=False)
        print(f"Batch '{target_batch}' complete. Index saved to {batch_index_path}")

    combined_index_path = dest_root / "index.json"
    with open(combined_index_path, "w", encoding="utf-8") as f:
        json.dump(combined_index, f, indent=4, ensure_ascii=False)

    print(
        f"Extraction complete. Combined index saved to {combined_index_path}. "
        f"Succeeded: {len(combined_index)}, failed: {failed}."
    )
    return combined_index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multiprocess DCM frame extraction.")
    parser.add_argument(
        "--location",
        choices=("local", "remote"),
        default="remote",
        help="Data source location (default: local).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1).",
    )
    parser.add_argument(
        "--skip-cached",
        action="store_true",
        help="Skip DCM folders that already contain extracted frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    extract_and_index_data_mp(
        location=args.location,
        num_workers=args.workers,
        skip_cached=args.skip_cached,
    )


if __name__ == "__main__":
    main()
