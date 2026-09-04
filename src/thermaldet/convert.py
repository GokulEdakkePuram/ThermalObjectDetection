"""Build an Ultralytics-shaped dataset out of the FLIR ADAS v2 download.

v2 ships six directories -- three splits in each of two spectra -- and each one
is self-describing::

    images_thermal_train/   coco.json  data/*.jpg  analyticsData/*.tiff
    images_thermal_val/     coco.json  data/*.jpg  analyticsData/*.tiff
    video_thermal_test/     coco.json  data/*.jpg  analyticsData/*.tiff
    images_rgb_train/       coco.json  data/*.jpg
    images_rgb_val/         coco.json  data/*.jpg
    video_rgb_test/         coco.json  data/*.jpg

``data/`` holds the 8-bit frames the camera writes; ``analyticsData/`` holds
the raw 16-bit radiometric frames behind them, which is what the radiometry
ablation needs. The release notes mention that directory once.

The output is one tree per arm::

    data/flir_<arm>/
      images/{train,val,test}/...
      labels/{train,val,test}/*.txt
    configs/data/flir_<arm>.yaml

The three thermal arms are built from the *same* frame list -- see
:func:`frame_index` -- so a difference between two of them cannot be a
difference in which frames they saw.
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

# The five classes with enough annotations to measure. v2 declares all 80 COCO
# categories and uses 16, but the tail is unusable -- `dog` has four boxes in
# the entire training split, and mAP would weight it as heavily as `car`'s
# 73,623.
#
# `light` and `sign` are in the default set deliberately. They are the two
# large classes a thermal sensor should find *harder* than a visible one: a
# traffic light signals with colour and a street sign with printed contrast,
# and neither survives an infrared sensor. Without them the modality
# comparison is a foregone conclusion.
DEFAULT_CLASSES = ["person", "bike", "car", "light", "sign"]

IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png", ".tiff")

# Finder and Explorer resolve a name collision by appending " 2", " 3" and so
# on, and a dataset copied between machines a few times collects them. They
# duplicate a frame that is already present, so they must not enter as extra
# training examples. Nothing FLIR ships ends in a space and a number.
COPY_SUFFIX = re.compile(r" \d+$")


@dataclass(frozen=True)
class Spectrum:
    """One sensor's three splits, and where each keeps its imagery."""

    name: str
    splits: dict[str, str]
    # Source name -> subdirectory. "base" is the 8-bit frame every arm starts
    # from; "raw" is the 16-bit radiometric frame, thermal only.
    image_subdirs: dict[str, str]

    @property
    def has_raw(self) -> bool:
        return "raw" in self.image_subdirs


THERMAL = Spectrum(
    "thermal",
    {
        "train": "images_thermal_train",
        "val": "images_thermal_val",
        "test": "video_thermal_test",
    },
    {"base": "data", "raw": "analyticsData"},
)

RGB = Spectrum(
    "rgb",
    {"train": "images_rgb_train", "val": "images_rgb_val", "test": "video_rgb_test"},
    {"base": "data"},
)

SPECTRA = {s.name: s for s in (THERMAL, RGB)}


def check_root(root: Path) -> Path:
    """Fail early, and with the actual reason, if this is not a v2 download."""
    if not (root / "images_thermal_train" / "coco.json").exists():
        raise SystemExit(
            f"{root} does not look like a FLIR ADAS v2 download.\n"
            f"Expected {root / 'images_thermal_train' / 'coco.json'}."
        )
    return root


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

    Categories are matched by *name*. v2's ids are COCO's, so they are neither
    contiguous nor in the order we want (`light` is 10, `sign` is 12), and the
    thermal and visible files must produce the same YOLO indices for the
    modality comparison to mean anything.
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
    otherwise turn into NaN loss an hour into a run.
    """
    x, y, w, h = (float(v) for v in bbox)
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 1 or bh <= 1:
        return None
    cx, cy = (x0 + bw / 2) / width, (y0 + bh / 2) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}"


def frame_index(
    split_root: Path, spectrum: Spectrum, labels: dict[str, list[str]]
) -> tuple[list[str], dict[str, int]]:
    """The frames every arm of this spectrum is built from, and what was excluded.

    Intersected across *every* image source the spectrum has, not just the one
    the current arm needs. On a clean v2 download this changes nothing -- the
    8-bit and 16-bit directories match one for one. It stays because the check
    costs a directory listing and the failure it prevents is silent: two arms
    that were meant to differ only in preprocessing differing also in which
    frames they trained on.
    """
    stems = set(labels)
    counts = {"labelled": len(stems)}
    for name, subdir in spectrum.image_subdirs.items():
        present = _stems(split_root / subdir)
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
    """Write the Ultralytics data YAML for one arm."""
    out = CONFIG_DIR / "data" / f"flir_{arm}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Ultralytics data config for the '{arm}' arm.",
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


def build_split(
    split_root: Path,
    spectrum: Spectrum,
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

    source_dir = split_root / spectrum.image_subdirs["base" if render is None else "raw"]

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

    # Rendering 15k TIFFs is the one genuinely slow step in the pipeline, and
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
    spectrum: str = "thermal",
    classes: list[str] | None = None,
    render=None,
    copy: bool = False,
) -> dict:
    """Build one arm and return what was written.

    ``render`` is ``None`` for an arm that uses the shipped 8-bit frames, and a
    ``(tiff, png) -> None`` callable for one rendered from the raw 16-bit
    imagery. The frame list is computed identically either way.
    """
    check_root(root)
    spec = SPECTRA[spectrum]
    classes = classes or DEFAULT_CLASSES

    if render is not None and not spec.has_raw:
        raise SystemExit(
            f"The '{spec.name}' spectrum has no 16-bit imagery, so the '{arm}' arm "
            f"cannot be built from it."
        )

    print(f"[source ] FLIR ADAS v2, {spec.name} at {root}")
    print(f"[classes] {list(enumerate(classes))}")

    summary: dict[str, dict] = {}
    for split, subdir in spec.splits.items():
        split_root = root / subdir
        if not (split_root / "coco.json").exists():
            print(f"[{split:>5}  ] skipped -- no {split_root / 'coco.json'}")
            continue

        labels = coco_labels(split_root / "coco.json", classes)
        stems, counts = frame_index(split_root, spec, labels)
        if not stems:
            print(f"[{split:>5}  ] skipped -- no frame has both a label and an image")
            continue

        build_split(split_root, spec, split, arm, stems, labels, render, copy)

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
        raise SystemExit("Nothing was converted. Check --flir-root.")

    note = (
        "Frames are the intersection of those with a label and an image in every\n"
        "source, so each arm of a spectrum sees exactly the same frames."
    )
    data_yaml = write_data_yaml(arm, classes, list(summary), note)

    manifest = DATA_DIR / f"flir_{arm}" / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "arm": arm,
                "spectrum": spec.name,
                "source": str(root),
                "classes": classes,
                # How the pixels were produced. Inference has to reproduce it:
                # a model trained on a global window and fed 8-bit frames sees
                # a different image of the same scene, and fails quietly.
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
