"""Fine-tune a YOLO detector on the FLIR ADAS thermal dataset (standard path).

Thin wrapper over Ultralytics' `YOLO.train`, which is the recommended way to
fine-tune. Loading `yolov8n.pt` and pointing `--data` at a 4-class data yaml makes
Ultralytics automatically re-head the model to `nc=4` and report real mAP each
epoch. Results land under `runs/detect/<name>/` with `weights/best.pt`.

Run `python src/coco_to_yolo.py` first to build the dataset + data yaml.

Examples
--------
    # Full fine-tune (default)
    python src/train.py --epochs 100 --imgsz 640 --batch 16

    # Freeze the backbone (layers 0-9 on YOLOv8), train the head only
    python src/train.py --freeze 10

    # Quick CPU smoke test
    python src/train.py --epochs 1 --imgsz 320 --batch 4 --device cpu
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--weights", default="yolov8n.pt", help="pretrained checkpoint to fine-tune from")
    p.add_argument("--data", default="configs/flir_thermal.yaml", help="Ultralytics data yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--freeze", type=int, default=0, help="freeze first N layers (10 = backbone on v8)")
    p.add_argument("--patience", type=int, default=25, help="early-stopping patience (0 disables)")
    p.add_argument("--fraction", type=float, default=1.0, help="fraction of the train set to use (for quick runs)")
    p.add_argument("--device", default=default_device(), help="cuda index, 'mps', or 'cpu'")
    p.add_argument("--project", default=None, help="output dir (default: runs/detect)")
    p.add_argument("--name", default="flir_thermal")
    p.add_argument("--resume", action="store_true", help="resume the last run of --name")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        freeze=args.freeze,
        patience=args.patience,
        fraction=args.fraction,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        seed=args.seed,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
