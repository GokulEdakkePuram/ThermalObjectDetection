"""Evaluate a fine-tuned YOLO detector on the FLIR thermal val split.

Reports mAP50 and mAP50-95 (overall and per class) via Ultralytics'
`DetectionValidator`. Defaults to the preserved checkpoint in `models/`.

Examples
--------
    python src/val.py                                  # eval models/flir_yolov8n_finetuned.pt
    python src/val.py --weights runs/detect/flir_thermal/weights/best.pt
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--weights", default="models/flir_yolov8n_finetuned.pt", help="checkpoint to evaluate")
    p.add_argument("--data", default="configs/flir_thermal.yaml", help="Ultralytics data yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="cuda index, 'mps', or 'cpu' (auto if unset)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    print(f"\nmAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50   : {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
