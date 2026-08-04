"""Convert the FLIR ADAS 1.3 thermal annotations (COCO format) to a YOLO dataset.

FLIR ships one ``thermal_annotations.json`` (COCO format) per split under
``train/`` and ``val/``. This script builds a self-contained Ultralytics dataset::

    <out>/
    ├── images/{train,val}/FLIR_xxxxx.jpeg   (symlinks to the originals)
    └── labels/{train,val}/FLIR_xxxxx.txt    (YOLO: `cls cx cy w h`, normalized)

and writes/refreshes ``configs/flir_thermal.yaml`` so training always has a
correct, absolute ``path``.

Key decisions
-------------
- Only the 4 classes actually annotated in FLIR are kept, remapped to contiguous
  YOLO indices: person(1)->0, bicycle(2)->1, car(3)->2, dog(17)->3.
- Only images that actually exist on disk are included. The public FLIR download
  references more images than it ships, so ~40% of the JSON entries are skipped.
- `iscrowd` and degenerate (<1px) boxes are dropped; boxes are clipped to the
  image bounds (a few FLIR boxes spill over by a pixel or two).
- Images with no surviving objects get an empty `.txt` — Ultralytics treats these
  as background, which is what you want for detection.

Usage
-----
    python src/coco_to_yolo.py \
        --flir-root Dataset/FLIR_ADAS_1_3 \
        --out       Dataset/FLIR_ADAS_1_3/yolo
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

# Sparse COCO ids used by FLIR ADAS -> contiguous YOLO class indices.
# The order here defines the class list written to the data yaml.
COCO_TO_YOLO = {1: 0, 2: 1, 3: 2, 17: 3}
YOLO_NAMES = ["person", "bicycle", "car", "dog"]

SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--flir-root",
        type=Path,
        default=Path("Dataset/FLIR_ADAS_1_3"),
        help="FLIR root containing train/ and val/ (each with thermal_annotations.json)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("Dataset/FLIR_ADAS_1_3/yolo"),
        help="Output root for the YOLO dataset",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of symlinking (slower, ~2x disk)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/flir_thermal.yaml"),
        help="Ultralytics data yaml to write (absolute path + names)",
    )
    p.add_argument(
        "--no-yaml",
        action="store_true",
        help="Do not write/refresh the data yaml",
    )
    return p.parse_args()


def link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        import shutil

        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def convert_split(split: str, flir_root: Path, out: Path, *, copy: bool) -> dict[str, int]:
    """Build images/<split> and labels/<split>; return per-split stats."""
    ann_path = flir_root / split / "thermal_annotations.json"
    split_root = flir_root / split  # file_name is relative to this (e.g. thermal_8_bit/FLIR_x.jpeg)

    with ann_path.open() as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {"written": 0, "background": 0, "missing": 0, "dropped_boxes": 0}

    for img_id, img in images.items():
        src = split_root / img["file_name"]
        if not src.exists():
            stats["missing"] += 1
            continue

        stem = Path(img["file_name"]).stem  # FLIR_00001
        ext = Path(img["file_name"]).suffix  # .jpeg
        link_or_copy(src, img_dir / f"{stem}{ext}", copy=copy)

        W, H = img["width"], img["height"]
        lines: list[str] = []
        for ann in anns_by_img.get(img_id, []):
            if ann.get("iscrowd", 0) == 1:
                stats["dropped_boxes"] += 1
                continue
            cid = ann["category_id"]
            if cid not in COCO_TO_YOLO:
                stats["dropped_boxes"] += 1
                continue

            x, y, w, h = ann["bbox"]  # COCO: top-left x, y, width, height (pixels)
            x0 = max(0.0, float(x))
            y0 = max(0.0, float(y))
            x1 = min(float(W), x + w)
            y1 = min(float(H), y + h)
            bw, bh = x1 - x0, y1 - y0
            if bw <= 1 or bh <= 1:
                stats["dropped_boxes"] += 1
                continue

            cx = (x0 + bw / 2) / W
            cy = (y0 + bh / 2) / H
            lines.append(f"{COCO_TO_YOLO[cid]} {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")

        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["written"] += 1
        if not lines:
            stats["background"] += 1

    return stats


def write_data_yaml(config: Path, out: Path) -> None:
    config.parent.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(YOLO_NAMES))
    config.write_text(
        "# Ultralytics data config for FLIR ADAS 1.3 (thermal, 8-bit).\n"
        "# Auto-generated by src/coco_to_yolo.py — re-run it if you move the dataset.\n"
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(YOLO_NAMES)}\n"
        "names:\n"
        f"{names}\n"
    )


def main() -> None:
    args = parse_args()

    total = {"written": 0, "background": 0, "missing": 0, "dropped_boxes": 0}
    for split in SPLITS:
        stats = convert_split(split, args.flir_root, args.out, copy=args.copy)
        print(
            f"[{split:>5}] {stats['written']:>5} images "
            f"({stats['background']} background), "
            f"{stats['missing']} referenced-but-missing, "
            f"{stats['dropped_boxes']} boxes dropped"
        )
        for k in total:
            total[k] += stats[k]

    print(f"\n[output] {args.out.resolve()}")
    print(f"  total: {total['written']} images, {total['missing']} missing on disk")

    if not args.no_yaml:
        write_data_yaml(args.config, args.out)
        print(f"  config: {args.config} (path -> {args.out.resolve()})")


if __name__ == "__main__":
    main()
