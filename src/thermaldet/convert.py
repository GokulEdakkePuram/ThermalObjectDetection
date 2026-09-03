"""Build an Ultralytics-shaped dataset out of a FLIR ADAS download.

FLIR has shipped two incompatible layouts. Both are COCO underneath, so the
only real difference is where the JSON lives and what the splits are called::

    ADAS 1.3                          ADAS v2
    train/thermal_annotations.json    images_thermal_train/coco.json
    train/thermal_8_bit/*.jpeg        images_thermal_train/data/*.jpg
    train/thermal_16_bit/*.tiff       (no 16-bit imagery)
    val/, video/                      images_thermal_val/, video_thermal_test/

1.3 is the layout this project uses, for one reason: it ships the raw 16-bit
radiometric TIFFs alongside the 8-bit JPEGs, and the difference between those
two is the second ablation in this repo (see :mod:`thermaldet.radiometry`).
v2 has more images and no 16-bit, so it supports the transfer ablation only.

The output is one tree per preprocessing arm::

    data/flir_<arm>/
      images/{train,val,test}/...
      labels/{train,val,test}/*.txt
    configs/data/flir_<arm>.yaml

Every arm is built from the *same* frame list -- see :func:`frame_index` --
so a difference between two arms cannot be a difference in which frames they
saw.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import CONFIG_DIR, DATA_DIR

# Split name -> (directory, annotation filename) for each known layout.
LAYOUT_V1 = {
    "train": ("train", "thermal_annotations.json"),
    "val": ("val", "thermal_annotations.json"),
    "test": ("video", "thermal_annotations.json"),
}
LAYOUT_V2 = {
    "train": ("images_thermal_train", "coco.json"),
    "val": ("images_thermal_val", "coco.json"),
    "test": ("video_thermal_test", "coco.json"),
}

# Where each layout keeps its imagery, relative to the split directory.
IMAGE_SUBDIR_V1 = {"agc": "thermal_8_bit", "raw": "thermal_16_bit"}
IMAGE_SUBDIR_V2 = {"agc": "data"}

# 1.3 annotates four classes. `dog` is excluded by default and that is a
# deliberate call, not an oversight: it has 244 boxes in train and 16 in val,
# against 44,185 for `car`. mAP weights every class equally, so a class whose
# AP is decided by sixteen boxes would add more variance between two ablation
# arms than the thing being ablated. Pass --classes to put it back.
LABEL_CLASSES_V1 = ["person", "bicycle", "car", "dog"]
DEFAULT_CLASSES_V1 = ["person", "bicycle", "car"]
DEFAULT_CLASSES_V2 = ["person", "bike", "car"]

IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png", ".tiff")

# Finder and Explorer resolve a name collision by appending " 2", " 3" and so
# on, and a dataset copied between machines a few times collects them. They are
# duplicates of a frame that is already present, so they must not enter the
# dataset as extra training examples. FLIR stems are always `FLIR_00001` or
# `FLIR_video_00001`, so nothing legitimate ends in a space and a number.
COPY_SUFFIX = re.compile(r" \d+$")


@dataclass(frozen=True)
class Layout:
    """Which FLIR release we are looking at, and where its pieces live."""

    version: str
    splits: dict[str, tuple[str, str]]
    image_subdirs: dict[str, str]

    @property
    def has_raw(self) -> bool:
        """Whether this release ships the 16-bit radiometric imagery."""
        return "raw" in self.image_subdirs


def detect_layout(root: Path) -> Layout:
    """Work out which FLIR release ``root`` contains."""
    if (root / "images_thermal_train").is_dir():
        return Layout("v2", LAYOUT_V2, IMAGE_SUBDIR_V2)
    if (root / "train" / "thermal_8_bit").is_dir() or (root / "train" / "thermal_16_bit").is_dir():
        return Layout("1.3", LAYOUT_V1, IMAGE_SUBDIR_V1)
    raise SystemExit(
        f"{root} looks like neither FLIR ADAS 1.3 nor v2.\n"
        f"Expected either train/thermal_8_bit/ or images_thermal_train/."
    )


def _stems(directory: Path) -> set[str]:
    """Image stems in a directory, minus copy-collision duplicates."""
    if not directory.is_dir():
        return set()
    return {
        p.stem
        for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES and not COPY_SUFFIX.search(p.stem)
    }


def coco_labels(annotations: Path, keep: list[str]) -> dict[str, list[str]]:
    """Read a COCO JSON into ``{stem: [yolo lines]}``.

    Categories are matched by *name*, not id: 1.3 uses COCO's original ids
    (person=1, car=3, dog=18) while v2 renumbered and renamed them, so any
    hard-coded mapping is wrong on one of the two.
    """
    coco = json.loads(annotations.read_text())

    by_name = {c["name"]: c["id"] for c in coco["categories"]}
    missing = [c for c in keep if c not in by_name]
    if missing:
        raise SystemExit(f"Class(es) {missing} not in {annotations}. Available: {sorted(by_name)}")
    coco_to_yolo = {by_name[name]: keep.index(name) for name in keep}

    sizes = {
        img["id"]: (img["width"], img["height"], Path(img["file_name"]).stem)
        for img in coco["images"]
    }
    out: dict[str, list[str]] = {stem: [] for _, _, stem in sizes.values()}

    for ann in coco["annotations"]:
        if ann.get("iscrowd") or ann["category_id"] not in coco_to_yolo:
            continue
        width, height, stem = sizes[ann["image_id"]]
        line = _yolo_line(coco_to_yolo[ann["category_id"]], ann["bbox"], width, height)
        if line:
            out[stem].append(line)

    return out


def _yolo_line(class_id: int, bbox: list[float], width: int, height: int) -> str | None:
    """Convert one COCO ``[x, y, w, h]`` box to a normalised YOLO line.

    Boxes are clipped to the frame and degenerate ones dropped -- FLIR's
    annotations include a handful of zero-width boxes that Ultralytics would
    otherwise silently turn into NaN loss.
    """
    x, y, w, h = (float(v) for v in bbox)
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 1 or bh <= 1:
        return None
    cx, cy = (x0 + bw / 2) / width, (y0 + bh / 2) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}"


def adopted_labels(
    directory: Path, label_classes: list[str], keep: list[str]
) -> dict[str, list[str]]:
    """Read an existing YOLO label directory, remapping to the kept classes.

    The escape hatch for a download whose annotation JSONs have gone missing.
    ``label_classes`` names the class order those files were written in --
    there is nothing in a YOLO ``.txt`` that records it, so it has to be
    supplied rather than guessed.
    """
    remap = {label_classes.index(name): keep.index(name) for name in keep if name in label_classes}

    out: dict[str, list[str]] = {}
    for path in sorted(directory.glob("*.txt")):
        if COPY_SUFFIX.search(path.stem):
            continue
        lines = []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            source_id = int(float(parts[0]))
            if source_id in remap:
                lines.append(" ".join([str(remap[source_id]), *parts[1:5]]))
        out[path.stem] = lines
    return out


def frame_index(
    root: Path, layout: Layout, labels: dict[str, list[str]]
) -> tuple[list[str], dict[str, int]]:
    """The frames every arm is built from, and what got excluded.

    Intersected across *every* image source the release ships, not just the
    one the current arm needs. Doing it per-arm looks equivalent and is not:
    this download is missing ~15% of the 8-bit JPEGs, so the arms came out at
    7,562 and 7,543 training frames. A 19-frame difference is small, but it is
    exactly the kind that makes a 1% gap in mAP unattributable.

    Those 19 turned out to be Finder ``" 2"`` copies sitting beside the JPEGs
    and not beside the TIFFs, which is why ``_stems`` now filters them at
    source. The intersection stays regardless: it is the half that does not
    depend on having noticed the cause.
    """
    available = {name: _stems(root / sub) for name, sub in layout.image_subdirs.items()}

    stems = set(labels)
    counts = {"labelled": len(stems)}
    for name, present in available.items():
        counts[f"missing_{name}"] = len(stems - present)
        stems &= present

    counts["kept"] = len(stems)
    return sorted(stems), counts


def link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        import shutil

        shutil.copy2(src, dst)
    else:
        # Relative symlink, so the built tree survives the repo being moved.
        os.symlink(os.path.relpath(src.resolve(), start=dst.parent), dst)


def write_data_yaml(arm: str, classes: list[str], splits: list[str], note: str = "") -> Path:
    """Write the Ultralytics data YAML for one preprocessing arm."""
    out = CONFIG_DIR / "data" / f"flir_{arm}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Ultralytics data config for the '{arm}' preprocessing arm.",
        "# Generated by `thermaldet convert` -- re-run it rather than editing.",
    ]
    if note:
        lines += [f"# {line}" for line in note.splitlines()]
    lines += [
        f"path: {os.path.relpath((DATA_DIR / f'flir_{arm}').resolve(), CONFIG_DIR.parent)}",
        "train: images/train",
        "val: images/val",
    ]
    if "test" in splits:
        lines.append("test: images/test")
    lines += [f"nc: {len(classes)}", "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(classes)]

    out.write_text("\n".join(lines) + "\n")
    return out


def class_histogram(labels: dict[str, list[str]], stems: list[str]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for stem in stems:
        for line in labels.get(stem, []):
            counts[int(line.split()[0])] += 1
    return counts


def load_labels(
    root: Path,
    layout: Layout,
    split: str,
    keep: list[str],
    label_classes: list[str],
    adopt_from: Path | None,
) -> dict[str, list[str]] | None:
    """Get labels for one split from whichever source is actually present."""
    subdir, filename = layout.splits[split]

    annotations = root / subdir / filename
    if annotations.exists():
        return coco_labels(annotations, keep)

    if adopt_from is not None:
        directory = adopt_from / split
        if directory.is_dir():
            return adopted_labels(directory, label_classes, keep)

    return None


def build_split(
    root: Path,
    layout: Layout,
    split: str,
    arm: str,
    stems: list[str],
    labels: dict[str, list[str]],
    render,
    copy: bool,
) -> None:
    """Materialise one split of one arm: images plus label files."""
    out = DATA_DIR / f"flir_{arm}"
    image_dir, label_dir = out / "images" / split, out / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    source_dir = root / layout.image_subdirs["agc" if render is None else "raw"]

    keep = set(stems)
    for stale in (p for p in (*image_dir.iterdir(), *label_dir.iterdir()) if p.stem not in keep):
        stale.unlink()

    for stem in stems:
        lines = labels.get(stem, [])
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    if render is None:
        for stem in stems:
            src = next(
                p for suffix in IMAGE_SUFFIXES if (p := source_dir / f"{stem}{suffix}").exists()
            )
            link_or_copy(src, image_dir / src.name, copy=copy)
        return

    # Rendering 10k TIFFs is the one genuinely slow step in the pipeline, and
    # it is embarrassingly parallel. The mapping objects are frozen dataclasses
    # precisely so they survive being pickled out to a worker.
    pending = [
        (source_dir / f"{stem}.tiff", image_dir / f"{stem}.png")
        for stem in stems
        if not (image_dir / f"{stem}.png").exists()
    ]
    if not pending:
        return
    sources, targets = zip(*pending, strict=True)
    with ProcessPoolExecutor() as pool:
        for _ in pool.map(render, sources, targets, chunksize=32):
            pass


def convert(
    root: Path,
    arm: str = "agc",
    classes: list[str] | None = None,
    label_classes: list[str] | None = None,
    adopt_from: Path | None = None,
    render=None,
    copy: bool = False,
) -> dict:
    """Build one preprocessing arm and return what was written.

    ``render`` is ``None`` for the arm that uses FLIR's shipped 8-bit JPEGs,
    and a ``(tiff, png) -> None`` callable for an arm rendered from the raw
    16-bit imagery. The frame list is computed identically either way.
    """
    layout = detect_layout(root)
    classes = classes or (DEFAULT_CLASSES_V1 if layout.version == "1.3" else DEFAULT_CLASSES_V2)
    label_classes = label_classes or (
        LABEL_CLASSES_V1 if layout.version == "1.3" else DEFAULT_CLASSES_V2
    )

    if render is not None and not layout.has_raw:
        raise SystemExit(
            f"FLIR {layout.version} ships no 16-bit imagery, so the '{arm}' arm cannot be built. "
            f"The radiometry ablation needs the 1.3 release."
        )

    print(f"[layout ] FLIR ADAS {layout.version} at {root}")
    print(f"[classes] {list(enumerate(classes))}")

    summary: dict[str, dict] = {}
    for split in layout.splits:
        subdir, _ = layout.splits[split]
        split_root = root / subdir
        if not split_root.is_dir():
            continue

        labels = load_labels(root, layout, split, classes, label_classes, adopt_from)
        if labels is None:
            print(f"[{split:>5}  ] skipped -- no annotations and nothing to adopt")
            continue

        stems, counts = frame_index(split_root, layout, labels)
        if not stems:
            print(f"[{split:>5}  ] skipped -- no frame has both a label and an image")
            continue

        build_split(split_root, layout, split, arm, stems, labels, render, copy)

        histogram = class_histogram(labels, stems)
        empty = sum(1 for stem in stems if not labels.get(stem))
        summary[split] = {
            "frames": len(stems),
            "background_frames": empty,
            "boxes": sum(histogram.values()),
            "excluded": {k: v for k, v in counts.items() if k.startswith("missing_") and v},
            "class_counts": {classes[i]: histogram[i] for i in range(len(classes))},
        }
        dropped = counts["labelled"] - counts["kept"]
        print(
            f"[{split:>5}  ] {len(stems):>6,} frames, {sum(histogram.values()):>7,} boxes"
            + (f"  ({dropped:,} labelled frames had no image)" if dropped else "")
        )

    if not summary:
        raise SystemExit("Nothing was converted. Check --flir-root and --adopt-labels.")

    note = (
        "Frames are the intersection of those with a label and an image in every\n"
        "source, so each arm sees exactly the same frames."
    )
    data_yaml = write_data_yaml(arm, classes, list(summary), note)

    manifest = DATA_DIR / f"flir_{arm}" / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "arm": arm,
                "flir_version": layout.version,
                "source": str(root),
                "classes": classes,
                # How the pixels were produced. Inference has to reproduce it:
                # a model trained on a global window and fed AGC JPEGs sees a
                # different image of the same scene, and fails quietly.
                "mapping": (
                    {"kind": "agc"}
                    if render is None
                    else {"kind": type(render).__name__, **asdict(render)}
                ),
                "splits": summary,
            },
            indent=2,
        )
    )
    print(f"[output ] {DATA_DIR / f'flir_{arm}'}")
    print(f"[config ] {data_yaml}")
    return {"arm": arm, "data": str(data_yaml), "splits": summary}
