"""Convert the FLIR ADAS v2 thermal annotations (COCO format) to a YOLO dataset.

FLIR ADAS v2 ships one COCO ``coco.json`` per split, each alongside a ``data/``
image folder::

    Dataset/FLIR_ADAS_v2/
    ├── images_thermal_train/{coco.json, data/*.jpg}
    ├── images_thermal_val/{coco.json, data/*.jpg}
    └── video_thermal_test/{coco.json, data/*.jpg}

This script builds a self-contained Ultralytics dataset::

    <out>/
    ├── images/{train,val,test}/*.jpg   (symlinks to the originals)
    └── labels/{train,val,test}/*.txt   (YOLO: `cls cx cy w h`, normalized)

and writes/refreshes ``configs/flir_thermal.yaml`` so training always has a
correct, absolute ``path``.

Key decisions
-------------
- Classes are selected by name via ``--classes`` (default: person bike car) and
  their COCO ids are looked up from ``coco.json`` — so the mapping stays correct
  even though v2 renumbered/renamed categories vs v1.3 (e.g. `bicycle`->`bike`).
- `iscrowd` and degenerate (<1px) boxes are dropped; boxes are clipped to the
  image bounds.
- Images with no surviving objects get an empty `.txt` — Ultralytics treats
  these as background, which is what you want for detection.
- Images referenced by the JSON but missing on disk are skipped.

Usage
-----
    python src/coco_to_yolo.py \
        --flir-root Dataset/FLIR_ADAS_v2 \
        --out       Dataset/FLIR_ADAS_v2/yolo
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

# YOLO split name -> FLIR v2 split directory.
SPLIT_DIRS = {
    "train": "images_thermal_train",
    "val": "images_thermal_val",
    "test": "video_thermal_test",
}

DEFAULT_CLASSES = ["person", "bike", "car"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--flir-root",
        type=Path,
        default=Path("Dataset/FLIR_ADAS_v2"),
        help="FLIR ADAS v2 root (contains images_thermal_{train,val}/ and video_thermal_test/)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("Dataset/FLIR_ADAS_v2/yolo"),
        help="Output root for the YOLO dataset",
    )
    p.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Category names to keep, in the desired YOLO index order",
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


def build_class_map(coco: dict, classes: list[str]) -> dict[int, int]:
    """Map each kept category's COCO id -> contiguous YOLO index (order = `classes`)."""
    name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    missing = [c for c in classes if c not in name_to_id]
    if missing:
        raise SystemExit(f"Class(es) {missing} not found in coco.json. Available: {sorted(name_to_id)}")
    return {name_to_id[name]: yolo_idx for yolo_idx, name in enumerate(classes)}


def convert_split(
    split: str, split_dir: Path, out: Path, coco_to_yolo: dict[int, int], *, copy: bool
) -> dict[str, int]:
    """Build images/<split> and labels/<split>; return per-split stats."""
    with (split_dir / "coco.json").open() as f:
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
        src = split_dir / img["file_name"]  # file_name is e.g. "data/video-...jpg"
        if not src.exists():
            stats["missing"] += 1
            continue

        stem = Path(img["file_name"]).stem
        ext = Path(img["file_name"]).suffix
        link_or_copy(src, img_dir / f"{stem}{ext}", copy=copy)

        W, H = img["width"], img["height"]
        lines: list[str] = []
        for ann in anns_by_img.get(img_id, []):
            if ann.get("iscrowd"):
                stats["dropped_boxes"] += 1
                continue
            cid = ann["category_id"]
            if cid not in coco_to_yolo:
                continue  # class we're not keeping — silently skip

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
            lines.append(f"{coco_to_yolo[cid]} {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")

        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["written"] += 1
        if not lines:
            stats["background"] += 1

    return stats


def write_data_yaml(config: Path, out: Path, classes: list[str], splits: list[str]) -> None:
    config.parent.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    lines = [
        "# Ultralytics data config for FLIR ADAS v2 (thermal).",
        "# Auto-generated by src/coco_to_yolo.py — re-run it if you move the dataset.",
        f"path: {out.resolve()}",
        "train: images/train",
        "val: images/val",
    ]
    if "test" in splits:
        lines.append("test: images/test")
    lines += [f"nc: {len(classes)}", "names:", names, ""]
    config.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()

    # Category ids are consistent across splits; read them once from train.
    with (args.flir_root / SPLIT_DIRS["train"] / "coco.json").open() as f:
        coco_to_yolo = build_class_map(json.load(f), args.classes)
    print(f"[classes] {list(enumerate(args.classes))}")

    done_splits = []
    total = {"written": 0, "background": 0, "missing": 0, "dropped_boxes": 0}
    for split, subdir in SPLIT_DIRS.items():
        split_dir = args.flir_root / subdir
        if not (split_dir / "coco.json").exists():
            print(f"[{split:>5}] skipped — no {split_dir / 'coco.json'}")
            continue
        stats = convert_split(split, split_dir, args.out, coco_to_yolo, copy=args.copy)
        done_splits.append(split)
        print(
            f"[{split:>5}] {stats['written']:>6} images "
            f"({stats['background']} background), "
            f"{stats['missing']} missing, {stats['dropped_boxes']} boxes dropped"
        )
        for k in total:
            total[k] += stats[k]

    print(f"\n[output] {args.out.resolve()}")
    print(f"  total: {total['written']} images, {total['missing']} missing on disk")

    if not args.no_yaml:
        write_data_yaml(args.config, args.out, args.classes, done_splits)
        print(f"  config: {args.config} (path -> {args.out.resolve()})")


if __name__ == "__main__":
    main()
