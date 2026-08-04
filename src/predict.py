"""Run a fine-tuned YOLO detector on thermal images and save annotated results.

`--source` accepts a single image, a directory, or a glob. Annotated images are
written under `runs/detect/predict/` (Ultralytics default).

Examples
--------
    python src/predict.py --source Dataset/FLIR_ADAS_1_3/val/thermal_8_bit/FLIR_08863.jpeg
    python src/predict.py --source Dataset/FLIR_ADAS_1_3/val/thermal_8_bit --conf 0.35
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--weights", default="models/flir_yolov8n_finetuned.pt", help="checkpoint to run")
    p.add_argument("--source", required=True, help="image, directory, or glob to run on")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--device", default=None, help="cuda index, 'mps', or 'cpu' (auto if unset)")
    p.add_argument("--no-save", action="store_true", help="do not write annotated images")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=not args.no_save,
    )
    total = sum(len(r.boxes) for r in results)
    print(f"\n{len(results)} image(s), {total} detection(s)")
    if not args.no_save and results:
        print(f"annotated output -> {results[0].save_dir}")


if __name__ == "__main__":
    main()
