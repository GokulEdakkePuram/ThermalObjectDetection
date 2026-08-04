# ThermalObjectDetection

Fine-tune a [YOLO](https://docs.ultralytics.com) detector for object detection on
**FLIR ADAS 1.3** thermal images. The pipeline converts FLIR's COCO annotations to
YOLO format and fine-tunes YOLOv8 on four classes: `person`, `bicycle`, `car`, `dog`.

## Setup

Uses [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```

## Dataset

Download the [FLIR ADAS thermal dataset](https://www.flir.com/oem/adas/adas-dataset-form/)
and place it at `Dataset/FLIR_ADAS_1_3/` with FLIR's original layout:

```
Dataset/FLIR_ADAS_1_3/
├── train/
│   ├── thermal_8_bit/FLIR_xxxxx.jpeg
│   └── thermal_annotations.json     # COCO format
└── val/
    ├── thermal_8_bit/
    └── thermal_annotations.json
```

Only the 8-bit thermal images and the two COCO annotation files are needed.
`Dataset/` is git-ignored. Note: the public download references more images than
it ships, so the converter simply skips any image not present on disk.

## Pipeline

```bash
# 1. Convert FLIR COCO -> YOLO dataset + refresh configs/flir_thermal.yaml
uv run python src/coco_to_yolo.py --flir-root Dataset/FLIR_ADAS_1_3 --out Dataset/FLIR_ADAS_1_3/yolo

# 2. Fine-tune (writes runs/detect/flir_thermal/weights/best.pt)
uv run python src/train.py --epochs 100 --imgsz 640 --batch 16

# 3. Evaluate mAP on the val split
uv run python src/val.py --weights runs/detect/flir_thermal/weights/best.pt

# 4. Run inference on an image or directory
uv run python src/predict.py --source Dataset/FLIR_ADAS_1_3/val/thermal_8_bit --conf 0.35
```

The converter builds a self-contained Ultralytics dataset under
`Dataset/FLIR_ADAS_1_3/yolo/{images,labels}/{train,val}` (images are symlinked)
and writes the absolute `path` into [`configs/flir_thermal.yaml`](configs/flir_thermal.yaml),
which every script reads by default.

## Scripts

| Script | Purpose |
| --- | --- |
| [`src/coco_to_yolo.py`](src/coco_to_yolo.py) | FLIR COCO → YOLO dataset + data yaml |
| [`src/train.py`](src/train.py) | **Primary** fine-tuning via Ultralytics `YOLO.train` |
| [`src/val.py`](src/val.py) | Evaluate mAP50 / mAP50-95 (per class) |
| [`src/predict.py`](src/predict.py) | Run inference, save annotated images |

Every script has `--help`. Device is auto-selected (CUDA → MPS → CPU); override
with `--device`. To freeze the backbone and train the head only, pass
`--freeze 10` to `src/train.py`.

## Preserved weights

[`models/flir_yolov8n_finetuned.pt`](models/) is a checkpoint from an earlier
fine-tuning run and is the default for `src/val.py` and `src/predict.py`.
