"""
Export YOLO labels from CVAT and store them under the batch-aware layout::

    labeling/labels/{batch}/{dcm_stem}/frame_xxx.txt

Relative keys (``batch/dcm_stem/frame_xxx``) link labels to PNGs via
:mod:`dl_lib.etalon_object_detection.modules.path_layout`.
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    import cvat_sdk
except ImportError:
    print(
        "This script requires the cvat-sdk package. "
        "Install it with: pip install cvat-sdk"
    )
    raise SystemExit(1)

from dl_lib.etalon_object_detection.modules.path_layout import (
    RelativeKeyError,
    build_dcm_stem_to_rel_map,
    dcm_relative_key_from_cvat_path,
    is_frame_name,
    label_path_from_relative_key,
    labels_dir,
    load_index_maps,
    make_frame_relative_key,
    require_dcm_relative_key,
    resolve_dcm_relative_key,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

SKIP_DIR_NAMES = frozenset({"obj_train_data", "labels", "train", "val", "test"})


def _is_frame_label_file(name: str) -> bool:
    return name.endswith(".txt") and is_frame_name(Path(name).stem)


def _collect_label_files(extract_root: Path) -> List[Tuple[Path, str]]:
    """
    Walk a CVAT YOLO export and return ``(label_file, path_hint)`` pairs.

    ``path_hint`` is the full path string used to recover ``batch/dcm_stem``
    from CVAT prefixes when present.
    """
    collected: List[Tuple[Path, str]] = []
    for root, _, files in os.walk(extract_root):
        frame_files = [f for f in files if _is_frame_label_file(f)]
        if not frame_files:
            continue

        folder_name = Path(root).name
        if folder_name.lower() in SKIP_DIR_NAMES:
            continue

        path_hint = str(root)
        for file_name in frame_files:
            collected.append((Path(root) / file_name, path_hint))
    return collected


def _install_label_file(
    label_file: Path,
    dcm_relative_key: str,
    output_dir: Path,
    installed_keys: Set[str],
) -> bool:
    dcm_key = require_dcm_relative_key(dcm_relative_key)
    batch, dcm_stem = dcm_key.split("/", 1)
    frame_relative_key = make_frame_relative_key(batch, dcm_stem, label_file.stem)
    dest_path = label_path_from_relative_key(frame_relative_key, output_dir)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(label_file, dest_path)
    installed_keys.add(frame_relative_key)
    return True


def export_labels_from_cvat(
    task_map: Dict[int, str],
    output_dir: str | Path | None = None,
    cvat_url: str = "http://10.1.11.11:8081",
    credentials: tuple = ("ayhem", "0000"),
) -> Dict[str, int]:
    """
    Export CVAT tasks and write labels as ``output_dir/{batch}/{dcm_stem}/frame_xxx.txt``.
    """
    output_path = Path(output_dir) if output_dir is not None else labels_dir()
    output_path.mkdir(parents=True, exist_ok=True)

    rel_to_dcm, _ = load_index_maps()
    stem_to_rel = build_dcm_stem_to_rel_map(rel_to_dcm)

    annotation_format = "Ultralytics YOLO Detection 1.0"
    installed_keys: Set[str] = set()
    skipped_dcm_stems: Set[str] = set()
    tasks_processed = 0

    print(f"Connecting to CVAT at {cvat_url}...")
    with cvat_sdk.make_client(host=cvat_url, credentials=credentials) as client:
        for task_id, task_label in task_map.items():
            print(f"Processing Task ID: {task_id} ({task_label})")

            zip_path = output_path / f"task_{task_id}_{task_label}.zip"
            if zip_path.exists():
                zip_path.unlink()

            try:
                task = client.tasks.retrieve(task_id)
                task.export_dataset(
                    format_name=annotation_format,
                    filename=str(zip_path),
                    include_images=False,
                )
            except Exception as e:
                logger.error("Failed to export task %s: %s", task_id, e)
                continue

            extract_root = output_path / f"_tmp_{task_id}"
            if extract_root.exists():
                shutil.rmtree(extract_root)
            extract_root.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_root)

            moved_count = 0
            for label_file, path_hint in _collect_label_files(extract_root):
                dcm_stem = label_file.parent.name

                dcm_relative_key = resolve_dcm_relative_key(
                    dcm_stem,
                    path_hint=path_hint,
                    rel_to_dcm=rel_to_dcm,
                    stem_to_rel=stem_to_rel,
                )
                if dcm_relative_key is None:
                    from_cvat = dcm_relative_key_from_cvat_path(path_hint)
                    if from_cvat is not None:
                        dcm_relative_key = from_cvat

                if dcm_relative_key is None:
                    skipped_dcm_stems.add(dcm_stem)
                    logger.warning(
                        "No batch mapping for DCM '%s' (from %s). Skipping.",
                        dcm_stem,
                        label_file,
                    )
                    continue

                try:
                    if _install_label_file(
                        label_file, dcm_relative_key, output_path, installed_keys
                    ):
                        moved_count += 1
                except RelativeKeyError as exc:
                    skipped_dcm_stems.add(dcm_stem)
                    logger.warning("Invalid relative key for '%s': %s", label_file, exc)

            for manifest_name in ("train", "val", "test"):
                manifest = extract_root / f"{manifest_name}.txt"
                if manifest.is_file():
                    shutil.copy2(
                        manifest,
                        output_path / f"task_{task_id}_{manifest_name}.txt",
                    )

            shutil.rmtree(extract_root, ignore_errors=True)
            if zip_path.exists():
                zip_path.unlink()

            print(f"Successfully exported task {task_id}. Installed {moved_count} label files.")
            tasks_processed += 1

    stats = {
        "tasks_processed": tasks_processed,
        "frames_installed": len(installed_keys),
        "skipped_dcm_stems": len(skipped_dcm_stems),
    }
    if skipped_dcm_stems:
        logger.warning("Skipped DCM stems: %s", ", ".join(sorted(skipped_dcm_stems)))
    return stats


def main() -> None:
    tasks_to_export = {
        # task_id: label/description
        7: "batch_large_diameters", # those should be the batch names used for 
        11: "batch_small_diameteres_2", 
        12: "batch_small_diameteres_1",
    }

    if not tasks_to_export:
        print(
            "No tasks configured for export. "
            "Update the 'tasks_to_export' dictionary in main()."
        )
        return

    stats = export_labels_from_cvat(tasks_to_export)
    print(
        "Done: {frames_installed} frame labels written to {labels_root} "
        "({skipped_dcm_stems} DCM stems skipped).".format(
            labels_root=labels_dir(),
            **stats,
        )
    )


if __name__ == "__main__":
    main()
