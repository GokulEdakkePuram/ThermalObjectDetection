"""Inference on thermal frames.

The one trap worth handling here: a checkpoint is only valid on the
preprocessing it was trained under. A model trained on a globally-windowed
16-bit rendering, then shown FLIR's AGC JPEGs, is looking at a different image
of the same scene -- and it fails quietly, with plausible-looking boxes in the
wrong places rather than an error. So raw ``.tiff`` input is rendered through
the checkpoint's own mapping, read back from the arm it was built with.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .config import resolve_device
from .evaluate import run_config
from .paths import DATA_DIR, configure_ultralytics

RAW_SUFFIX = ".tiff"


def arm_mapping(weights: str) -> tuple[str, Any]:
    """Recover the preprocessing a checkpoint was trained under."""
    from . import radiometry

    data = run_config(weights).get("data", "")
    arm = Path(data).stem.replace("flir_", "") if data else "agc"

    manifest = DATA_DIR / f"flir_{arm}" / "manifest.json"
    if not manifest.exists():
        raise SystemExit(
            f"Cannot tell how {weights} was preprocessed: no {manifest}.\n"
            f"Rebuild the arm with `thermaldet convert --arm {arm}`, or pass 8-bit input."
        )

    spec = dict(json.loads(manifest.read_text())["mapping"])
    kind = spec.pop("kind")
    if kind == "agc":
        return arm, None
    return arm, getattr(radiometry, kind)(**spec)


def _render_sources(sources: list[Path], render, into: Path) -> Path:
    for src in sources:
        render(src, into / f"{src.stem}.png")
    return into


def predict(
    weights: str,
    source: str,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "auto",
    save: bool = True,
) -> dict[str, Any]:
    """Run a checkpoint over an image, a directory, or a glob."""
    from ultralytics import YOLO

    configure_ultralytics()
    path = Path(source)
    raw = (
        sorted(path.glob(f"*{RAW_SUFFIX}"))
        if path.is_dir()
        else ([path] if path.suffix.lower() == RAW_SUFFIX else [])
    )

    model = YOLO(weights)
    with tempfile.TemporaryDirectory(prefix="thermaldet-") as tmp:
        if raw:
            arm, render = arm_mapping(weights)
            if render is None:
                reason = (
                    "the visible-spectrum frames, which a thermal TIFF is not"
                    if arm == "rgb"
                    else "FLIR's 8-bit frames, which are gain-control output -- there is no "
                    "way to reproduce that mapping from a raw TIFF"
                )
                raise SystemExit(
                    f"{weights} was trained on {reason} ('{arm}' arm). "
                    f"Point --source at the frames in data/ instead."
                )
            print(f"[thermaldet] rendering {len(raw)} raw frames through the '{arm}' mapping")
            source = str(_render_sources(raw, render, Path(tmp)))

        results = model.predict(
            source=source,
            imgsz=imgsz,
            conf=conf,
            device=resolve_device(device),
            save=save,
        )

    total = sum(len(r.boxes) for r in results)
    return {
        "images": len(results),
        "detections": total,
        "save_dir": str(results[0].save_dir) if save and results else "",
    }
