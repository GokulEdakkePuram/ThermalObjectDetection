# thermaldet — fine-tuning YOLO for thermal object detection

Fine-tuning YOLO11 on the
[Teledyne FLIR Free ADAS Thermal Dataset v2](https://www.flir.com/oem/adas/adas-dataset-form/)
— 640×512 thermal frames from a vehicle-mounted camera, with time-synced
visible-spectrum frames alongside — for five classes: `person`, `bike`, `car`,
`light`, `sign`.

Most FLIR projects do two things without questioning them: start from
COCO-pretrained weights, and train on the 8-bit JPEGs the camera writes. Both
are choices. A third question — whether the thermal sensor is the right one at
all — is usually assumed rather than measured. This repo tests all three, with
the reasoning in [`docs/`](docs/) and the predictions written down before the
runs.

**Contents**

- [Quickstart](#quickstart)
- [The dataset](#the-dataset) — what is on disk, and what it constrains
- [What the pipeline builds](#what-the-pipeline-builds)
- [The three experiments](#the-three-experiments)
- [Results](#results)
- [Command reference](#command-reference)
- [Repository layout](#repository-layout)
- [Design decisions](#design-decisions)

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and the FLIR ADAS v2 download placed
at `Dataset/FLIR_ADAS_v2/`.

```bash
make setup      # install dependencies into .venv
make arms       # build all four arms (~2 min, ~6 GB)
make profile    # class balance, object scale, dynamic range -> reports/
make smoke      # 1-epoch run to check the pipeline end to end
make pretrained # the control run
make eval       # score on the held-out test split -> reports/results.md
```

For training on a rented GPU, see [docs/06](docs/06-running-on-rented-gpus.md).

---

## The dataset

FLIR ADAS v2: 26,442 annotated frames from a thermal and visible camera pair
on a vehicle. Six directories — three splits in each of two spectra.

| split | frames | annotations | 8-bit | 16-bit |
| --- | ---: | ---: | ---: | ---: |
| `images_thermal_train` | 10,742 | 175,040 | 10,742 | 10,742 |
| `images_thermal_val` | 1,144 | 16,696 | 1,144 | 1,144 |
| `video_thermal_test` | 3,749 | 62,317 | 3,749 | 3,749 |
| `images_rgb_train` | 10,318 | 169,174 | 10,318 | — |
| `images_rgb_val` | 1,085 | 16,909 | 1,085 | — |
| `video_rgb_test` | 3,749 | 84,786 | 3,749 | — |

Verified against the files, not the release notes: every `file_name` in every
`coco.json` resolves on disk, every thermal frame has a matching 16-bit TIFF,
and there are no duplicates. Nothing here needs working around.

Three things about it shape everything below.

**`analyticsData/` holds the raw 16-bit frames.** The release notes mention
that directory once. `data/` holds the 8-bit JPEGs everyone trains on; the
radiometric frames behind them are shipped too, which is what makes
[the second experiment](#2-whether-the-8-bit-frames-throw-away-signal)
possible.

**The test split is real.** 3,749 annotated frames from — per FLIR —
completely independent video sequences. Every number in this repo is scored on
it, not on validation, so nothing is reported on data that `best.pt` was
selected against.

**Both spectra are annotated**, with the same label map, which makes
[the third experiment](#3-whether-thermal-is-the-right-sensor) askable at all.

### Profile

```bash
uv run thermaldet profile
```

| split | frames | boxes | boxes/frame | small % | background frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 10,742 | 168,238 | 15.7 | 74.6 | 280 |
| val | 1,144 | 16,244 | 14.2 | 75.1 | 19 |
| test | 3,749 | 55,371 | 14.8 | **87.3** | 256 |

"Small" is COCO's convention: box area under 32×32 px.

| class | train boxes | share | median box height |
| --- | ---: | ---: | ---: |
| car | 73,622 | 43.8% | 22 px |
| person | 50,474 | 30.0% | 26 px |
| sign | 20,747 | 12.3% | **11 px** |
| light | 16,158 | 9.6% | **16 px** |
| bike | 7,237 | 4.3% | 30 px |

Dense scenes — 15.7 objects per frame against roughly 7 on COCO — and small
objects: an 11-pixel sign sits close to the floor of what a stride-8 detection
head can resolve. The test split is harder than training (87.3% small against
74.6%), because those frames were sampled at 30 fps from continuous footage
rather than curated for detector training.

**Five classes of sixteen.** v2 declares all 80 COCO categories and uses 16,
but `dog` has four training boxes and mAP weights it as heavily as `car`'s
73,622. `light` and `sign` are kept deliberately rather than stopping at the
obvious three — see [experiment 3](#3-whether-thermal-is-the-right-sensor).

Full audit, including the lighting metadata and the two cameras hiding in the
test split: [docs/00](docs/00-the-dataset.md).

---

## What the pipeline builds

`thermaldet convert` turns the download into one Ultralytics dataset per
**arm**:

```
Dataset/FLIR_ADAS_v2/          the download, read-only
data/                          built by `thermaldet convert`
├── flir_agc/                  thermal, FLIR's shipped 8-bit frames
├── flir_global/               thermal, 16-bit -> 8-bit, one fixed window
├── flir_p1p99/                thermal, 16-bit -> 8-bit, per-frame stretch
└── flir_rgb/                  visible spectrum
     ├── images/{train,val,test}/
     ├── labels/{train,val,test}/
     └── manifest.json         which mapping produced these pixels
configs/data/flir_*.yaml       generated Ultralytics data configs
```

The three thermal arms are built from the same frame list, so they differ only
in pixels. `manifest.json` records the mapping and its parameters, which is how
`thermaldet predict` knows to render a raw TIFF the same way its checkpoint was
trained.

---

## The three experiments

Seven runs, three axes, one shared control. Every arm differs from
`pretrained` in **exactly one line** of config, and a
[test](tests/test_config.py) asserts that.

### 1. What transfers from COCO

A thermal frame is single-channel, replicated to three so it fits an RGB
network. Every first-layer filter therefore sees the same image three times,
and its response depends on the sum of its channel slices. An achromatic edge
detector reinforces. A colour-opponent filter — red-minus-green, the kind that
finds a brake light — cancels, and outputs approximately zero on every thermal
frame it will ever see.

That is measurable without training anything:

```bash
uv run thermaldet stem-check yolo11s.pt
```

```
stem: 32 filters
  grey-input response below 0.1: 10 filters (31%)
  median response ratio: 0.93 (1.0 = channel slices reinforce, 0.0 = they cancel)
```

**31% of the first layer is inert before a single gradient step.** The median
filter sits at 0.93, so the stem does not degrade evenly — it splits into a
majority that transfers intact and a chromatic third that cannot.

That says those filters are inert. Whether the network *misses* them is what
the runs cost GPU time to find out:

| run | change from control | question |
| --- | --- | --- |
| `pretrained` | — | the control |
| `scratch` | `model: yolo11s.yaml` | what is COCO pretraining worth at all? |
| `frozen_stem` | `freeze: 2` | must layers 0–1 be relearned? |
| `frozen_backbone` | `freeze: 11` | must anything below the head be? |

`freeze: 11` is the whole backbone on YOLO11, whose layers run 0–10. The widely
copied `freeze: 10` is a YOLOv8 number; on YOLO11 it leaves the last backbone
block training while the config claims otherwise.

Predictions and falsification criteria:
[docs/01](docs/01-what-transfers-from-coco.md).

### 2. Whether the 8-bit frames throw away signal

FLIR's 8-bit JPEGs are not the sensor's output. They are the output of
**automatic gain control**: each frame's own range stretched to fill 0–255.
That is per-frame and non-stationary, so the same absolute temperature is a
different pixel value in every frame — which discards the one thing a thermal
camera uniquely measures.

The raw 16-bit frames are in `analyticsData/`, so the alternative is testable.
Measured over 400 sampled training frames:

| quantity | counts |
| --- | ---: |
| median frame span (p1–p99) | 764 |
| global window (pooled p0.5–p99.5) | 2,219 |
| span across frames | 3,097 |

A fixed window has to cover every frame, so a median frame reaches
`255 × 764 / 2219` = **88 of 255 output levels**. Keeping absolute radiometry
costs **2.9× of per-frame contrast**.

| run | mapping |
| --- | --- |
| `pretrained` (control) | FLIR's AGC, 8-bit |
| `global_map` | one fixed window over the 16-bit, `[5905, 8124]` counts |
| `p1p99_map` | per-frame 1st–99th percentile stretch of the 16-bit |

The third arm makes the comparison decidable. Without it, `global` losing to
`agc` has two explanations — *per-frame normalisation helps*, or *FLIR's
particular curve helps* — and no way to choose.
[docs/02](docs/02-radiometry-and-agc.md).

### 3. Whether thermal is the right sensor

v2 annotates both spectra, so `rgb` differs from the control in one line.

This does not measure which sensor carries more information — they are
different instruments. It measures the deployment question: **given one 640 px
input budget, which sensor detects better?**

The resize sounds like it hands the result to thermal, since thermal is native
at 640×512 and visible frames run up to 2048×1536. Measured, it does not — the
fields of view are similar (45° against 52.8°), so after scaling both to a
640 px long side the objects land at nearly the same size:

| class | thermal @640 | visible @640 |
| --- | ---: | ---: |
| person | 26 px | 30 px |
| bike | 30 px | 28 px |
| car | 22 px | 21 px |
| light | 16 px | 12 px |
| sign | 11 px | 9 px |

**`light` and `sign` are in the class set to make this a real question.** A
traffic light signals with colour and a street sign with printed contrast, and
an infrared sensor sees neither — a red light and a green light are the same
temperature. So the prediction is per class, not in aggregate: thermal ahead on
`person` and `car`, visible ahead on `light` and `sign`. **If thermal wins on
all five, something is wrong with the comparison.**

**Part of the answer is already in the labels.** The test split is 3,749
*paired* frames — the same moments through both cameras — annotated
independently:

| class | thermal boxes | visible boxes | visible / thermal |
| --- | ---: | ---: | ---: |
| person | 12,323 | 11,278 | 0.92 |
| car | 30,517 | 30,888 | 1.01 |
| light | 6,758 | 17,817 | **2.64** |
| sign | 5,660 | 17,210 | **3.04** |
| bike | 113 | 446 | **3.95** |

On identical moments, annotators found 2.6× more traffic lights and 3× more
signs in the visible frames — and essentially the same number of cars. That is
the predicted split, measured by the labelling process rather than a model.

It also means the per-class AP comparison is **unfair in a known direction**:
each detector is scored against its own spectrum's labels, so the thermal model
is never penalised for the ~11,000 lights it cannot see. `person` and `car`
compare cleanly; `light`, `sign` and `bike` do not, and the label counts are
the better evidence for those.

FLIR published a baseline on this exact test split (YOLOX-m, 640×640,
AP@IoU=0.5): thermal 75.33 person / 77.23 car against visible 51.42 / 55.79.
Different architecture, so a sanity check rather than a leaderboard row.
[docs/03](docs/03-thermal-vs-visible.md).

---

## Results

> **Status:** pipeline complete and tested end to end; the seven training runs
> have not been executed yet — they run on a rented GPU. Every number
> *elsewhere* in this README is already measured and reproducible today.

All runs: YOLO11s, 60 epochs, 640 px, `batch: 32` on one RTX 4090, scored on
the held-out `video_thermal_test` split (or `video_rgb_test` for the modality
arm).

**Transfer**

| run | change | mAP50-95 | mAP50 | vs control |
| --- | --- | ---: | ---: | ---: |
| `pretrained` (control) | — | | | — |
| `scratch` | `model: yolo11s.yaml` | | | |
| `frozen_stem` | `freeze: 2` | | | |
| `frozen_backbone` | `freeze: 11` | | | |

**Radiometry**

| run | mapping | mAP50-95 | mAP50 | vs control |
| --- | --- | ---: | ---: | ---: |
| `pretrained` (control) | FLIR AGC, 8-bit | | | — |
| `global_map` | fixed window, 16-bit | | | |
| `p1p99_map` | per-frame stretch, 16-bit | | | |

**Modality** — per class, because the aggregate can hide the result

| run | person | bike | car | light | sign | mAP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pretrained` (thermal) | | | | | | |
| `rgb` (visible) | | | | | | |

`bike` has **113 boxes** in the held-out split and carries a fifth of the mAP,
so a swing in that column is noise rather than a result. Read
[docs/04](docs/04-reading-the-metrics.md) before quoting any single number from
here.

---

## Command reference

```bash
# data
uv run thermaldet convert --arm agc|global|p1p99|rgb
uv run thermaldet profile

# analysis that needs no training
uv run thermaldet stem-check yolo11s.pt

# training
uv run thermaldet train pretrained [--profile cuda24] [--track wandb]
uv run thermaldet train-manual --backbone-lr-mult 5.0
uv run thermaldet probe pretrained rgb             # cost a run before starting it

# evaluation and deployment
uv run thermaldet eval runs/train/*/weights/best.pt --split test
uv run thermaldet predict <weights> <frame.tiff>   # raw input rendered first
uv run thermaldet export <weights> --format onnx
```

`make help` lists the equivalent Make targets. Every command takes `--help`.

---

## Repository layout

```
configs/          one YAML per experiment, composed via `extends:`
  profiles/       what each machine can hold — batch, workers, device, amp
  data/           generated: one Ultralytics data YAML per arm
src/thermaldet/
  config.py       config loading, inheritance, validation
  hardware.py     accelerator detection and profile selection
  convert.py      FLIR v2 -> YOLO, both spectra, and the shared frame list
  radiometry.py   the three 16-bit -> 8-bit mappings
  stem.py         how much of an RGB stem survives a greyscale input
  stats.py        dataset profiling — class balance, box scale, dynamic range
  train.py        fine-tuning; writes the resolved config next to the weights
  manual.py       explicit training loop, for per-depth learning rates
  probe.py        short calibration run that projects the full schedule
  evaluate.py     validation and the cross-run comparison table
  predict.py      inference, rendering raw input through the right mapping
  tracking.py     W&B / MLflow wiring
  export.py       ONNX / CoreML / TorchScript export
scripts/          provisioning and unattended sweeps for a rented GPU box
docs/             the reasoning, written as it was worked out
tests/            81 tests, none needing the dataset or a GPU (`make test`)
Dockerfile        pinned training image
```

---

## Design decisions

**Hardware is separate from the experiment.** A config says what to train; a
profile in `configs/profiles/` says what the machine can hold, and
`load_profile` rejects any profile that tries to set `epochs` or `imgsz`. The
payoff is one batch size across all seven runs, so the only difference between
two arms is the line their config changed.

**Every arm of a spectrum is built from the same frame list.** On a clean v2
download this changes nothing — the 8-bit and 16-bit directories match one for
one. It stays because the check costs a directory listing and the failure it
prevents is silent, and because it was not always a no-op: on the FLIR 1.3
download it caught 19 duplicate frames entering one arm and not another.

**Cheap evidence before expensive evidence.** Every claim here has a
measurement that costs seconds and runs before any GPU is rented: `stem-check`
for the transfer argument, `profile` for the radiometry one, the box-height
table for the modality one. The runs test whether those measurements *matter*,
which is the expensive question.

**Runs are traceable.** Each run writes its fully-resolved config — profile
overlay included — to `runs/train/<name>/thermaldet_config.json`, and
Ultralytics' global data and output directories are pinned into the repo rather
than scattered across `$HOME`. `eval` reads that file back to decide which arm
a checkpoint should be scored against.

---

## Documentation

- [00 — The dataset](docs/00-the-dataset.md) — the v2 audit, the class set, and
  the lighting metadata
- [01 — What transfers from COCO](docs/01-what-transfers-from-coco.md) — the
  stem measurement and the four transfer arms
- [02 — Radiometry and AGC](docs/02-radiometry-and-agc.md) — what per-frame
  gain control destroys, and what a fixed window costs
- [03 — Thermal against visible](docs/03-thermal-vs-visible.md) — what the
  modality comparison can and cannot show, and the per-class prediction
- [04 — Reading the metrics](docs/04-reading-the-metrics.md) — mAP50 vs
  mAP50-95, the class imbalance, and why the test split is the one reported
- [05 — Experiment log](docs/05-experiment-log.md) — running journal, with
  expectations recorded before each run
- [06 — Running on rented GPUs](docs/06-running-on-rented-gpus.md) — profiles,
  calibrating a run before paying for it, and what wastes a rental

## Preserved weights

[`models/flir_yolov8n_finetuned.pt`](models/) is a YOLOv8n checkpoint from an
earlier pass over the FLIR 1.3 release, on four classes (`person`, `bicycle`,
`car`, `dog`). It predates everything above and is not part of any ablation. It
is kept because it is a working thermal detector and a fixed reference point.

## Licence

Code: MIT. Ultralytics YOLO is AGPL-3.0 — relevant if you build on this
commercially. FLIR ADAS v2 is released under
[its own terms](https://www.flir.com/oem/adas/adas-dataset-agree/); check
upstream before redistributing anything derived from it.
