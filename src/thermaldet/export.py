"""Model export.

A checkpoint that only runs inside a training framework is half a deliverable.
ONNX covers cross-platform serving, which for a thermal ADAS model means an
embedded runtime rather than a workstation; CoreML is the sensible target when
it has to run on Apple hardware.
"""

from __future__ import annotations

from pathlib import Path

from .paths import configure_ultralytics

SUPPORTED_FORMATS = ("onnx", "torchscript", "coreml", "openvino")


def export(weights: str, fmt: str = "onnx", imgsz: int = 640, half: bool = False) -> Path:
    """Export a checkpoint to a deployment format and return the artifact path."""
    from ultralytics import YOLO

    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format {fmt!r}. Choose from {SUPPORTED_FORMATS}.")

    configure_ultralytics()
    model = YOLO(weights)
    return Path(model.export(format=fmt, imgsz=imgsz, half=half))
