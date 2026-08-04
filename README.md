# ThermalObjectDetection

Fine-tune a [YOLO](https://docs.ultralytics.com) detector for object detection on
**FLIR ADAS v2** thermal images. The pipeline converts FLIR's COCO annotations to
YOLO format and fine-tunes YOLOv8 on three classes: `person`, `bike`, `car`.

## Setup

Uses [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```

## Dataset

Download the [FLIR ADAS v2 dataset](https://www.flir.com/oem/adas/adas-dataset-form/)
and place it at `Dataset/FLIR_ADAS_v2/`. Each split ships its own COCO `coco.json`
and a `data/` image folder:

```
Dataset/FLIR_ADAS_v2/
├── images_thermal_train/{coco.json, data/*.jpg}   # 10,742 images
├── images_thermal_val/{coco.json, data/*.jpg}     #  1,144 images
└── video_thermal_test/{coco.json, data/*.jpg}     #  3,749 images (held-out test)
```

Only the thermal splits are used (the `images_rgb_*` / `video_rgb_*` folders are
ignored). `Dataset/` is git-ignored.

## Pipeline

```bash
# 1. Convert FLIR COCO -> YOLO dataset + refresh configs/flir_thermal.yaml
uv run python src/coco_to_yolo.py --flir-root Dataset/FLIR_ADAS_v2 --out Dataset/FLIR_ADAS_v2/yolo

# 2. Fine-tune (writes runs/detect/flir_thermal/weights/best.pt)
uv run python src/train.py --epochs 100 --imgsz 640 --batch 16

# 3. Evaluate mAP on the val split
uv run python src/val.py --weights runs/detect/flir_thermal/weights/best.pt

# 4. Run inference on an image or directory
uv run python src/predict.py --source Dataset/FLIR_ADAS_v2/images_thermal_val/data --conf 0.35
```

The converter builds a self-contained Ultralytics dataset under
`Dataset/FLIR_ADAS_v2/yolo/{images,labels}/{train,val,test}` (images are symlinked)
and writes the absolute `path` into [`configs/flir_thermal.yaml`](configs/flir_thermal.yaml),
which every script reads by default. To train on a different class set, pass e.g.
`--classes person bike car motor bus truck` — ids are looked up by name from
`coco.json`, so no manual remapping is needed.

## Scripts

| Script | Purpose |
| --- | --- |
| [`src/coco_to_yolo.py`](src/coco_to_yolo.py) | FLIR COCO → YOLO dataset + data yaml |
| [`src/train.py`](src/train.py) | **Primary** fine-tuning via Ultralytics `YOLO.train` |
| [`src/train_manual.py`](src/train_manual.py) | **Advanced** — explicit PyTorch loop (per-group LRs, warmup, EMA, AMP) |
| [`src/val.py`](src/val.py) | Evaluate mAP50 / mAP50-95 (per class) |
| [`src/predict.py`](src/predict.py) | Run inference, save annotated images |

Every script has `--help`. Device is auto-selected (CUDA → MPS → CPU); override
with `--device`. To freeze the backbone and train the head only, pass
`--freeze 10` to `src/train.py`.

## Preserved weights

[`models/flir_yolov8n_finetuned.pt`](models/) is a checkpoint from an earlier
run fine-tuned on **FLIR ADAS 1.3** (4 classes: `person`, `bicycle`, `car`, `dog`).
It's still a working thermal detector, but its class set predates v2 — for v2
metrics, evaluate/predict with your freshly trained `runs/detect/flir_thermal/weights/best.pt`.
