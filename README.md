# thermaldet — fine-tuning YOLO for thermal object detection

Fine-tuning YOLO11 on [FLIR ADAS](https://www.flir.com/oem/adas/adas-dataset-form/)
thermal infrared imagery — 640×512 single-channel frames from a vehicle-mounted
camera — for three classes: `person`, `bicycle`, `car`.

Most FLIR projects do two things without questioning them: start from
COCO-pretrained weights, and train on the 8-bit JPEGs the camera writes. Both
are choices, and both can be tested. This repo tests them, with the reasoning
in [`docs/`](docs/) and the predictions written down before the runs.

**Contents**

- [Quickstart](#quickstart)
- [Dataset audit](#dataset-audit) — what is actually on disk, and what is missing
- [What the pipeline builds](#what-the-pipeline-builds)
- [The two experiments](#the-two-experiments)
- [Results](#results)
- [Command reference](#command-reference)
- [Repository layout](#repository-layout)
- [Design decisions](#design-decisions)

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and the FLIR ADAS 1.3 download placed
at `Dataset/FLIR_ADAS_1_3/`.

```bash
make setup      # install dependencies into .venv
make arms       # build all three preprocessing arms (~1 min, ~3.5 GB)
make profile    # class balance, object scale, dynamic range -> reports/
make smoke      # 1-epoch run to check the pipeline end to end
make pretrained # the control run
make eval       # comparison table -> reports/results.md
```

If your download is missing its `thermal_annotations.json` files but has
converted YOLO labels (see the audit below), point the converter at them:

```bash
make arms ADOPT="--adopt-labels Dataset/FLIR_ADAS_1_3/yolo/labels"
```

For training on a rented GPU, see [docs/05](docs/05-running-on-rented-gpus.md).

---

## Dataset audit

FLIR has released this dataset twice and the two are not interchangeable.

| | **ADAS 1.3** (used here) | ADAS v2 |
| --- | --- | --- |
| splits | `train/`, `val/`, `video/` | `images_thermal_train/`, `images_thermal_val/`, `video_thermal_test/` |
| annotations | `<split>/thermal_annotations.json` | `<split>/coco.json` |
| 8-bit thermal | `thermal_8_bit/*.jpeg` | `data/*.jpg` |
| **16-bit thermal** | `thermal_16_bit/*.tiff` | **not shipped** |
| paired RGB | `RGB/*.jpg`, unannotated | annotated |
| classes | person, bicycle, car, dog | person, bike, car |

Both are COCO underneath, and the converter reads either. **1.3 is used here
because it ships the raw 16-bit imagery**, which is what the second experiment
needs. v2 has more frames and no 16-bit, so it supports the first experiment
only.

Categories are matched by **name**, never by id. 1.3 keeps COCO's original
numbering (`person=1`, `car=3`, `dog=18`); v2 renumbered and renamed them. A
hard-coded id map is silently wrong on one of the two.

### What is actually on disk

This is the audit of the specific download this repo was built against. FLIR's
own ReadMe says the release contains 8,862 train frames (ids 1–8,862), 1,366
val frames (8,863–10,228) and 4,224 video frames.

**train**

| source | files | unique | `" 2"` duplicates | verdict |
| --- | ---: | ---: | ---: | --- |
| YOLO labels | 9,515 | **8,862** | 653 | complete |
| 16-bit TIFF | 9,263 | **8,862** | 401 | complete |
| 8-bit JPEG | 7,883 | **7,543** | 340 | **1,319 missing (14.9%)** |

**val**

| source | files | unique | `" 2"` duplicates | verdict |
| --- | ---: | ---: | ---: | --- |
| YOLO labels | 1,461 | **1,366** | 95 | complete |
| 16-bit TIFF | 1,366 | **1,366** | 0 | complete |
| 8-bit JPEG | 1,091 | **1,091** | 0 | **275 missing (20.1%)** |

### Four findings

**1. There are no annotation JSONs at all.** Zero `.json` files anywhere in the
download, in any split. The converter's primary path — read COCO, emit YOLO —
cannot run on this copy.

*Handled by:* `--adopt-labels`, which reads an existing YOLO label directory
instead. The class order in those files has to be named explicitly, since
nothing in a `.txt` records it.

**2. There is no test split.** `video/thermal_annotations.json` is one of the
missing files, so the 4,224 video frames have no labels. Everything in this
repo is therefore measured on validation, with `best.pt` also *selected* on
validation. Those numbers are optimistic by an unmeasured amount.

This does not invalidate the *comparisons* — every arm is selected the same
way — but no single number here should be quoted as this model's accuracy.
Recovering that one JSON would fix it; the converter already handles the split
when it is present. See [docs/03](docs/03-reading-the-metrics.md).

**3. The 8-bit JPEGs are incomplete; the 16-bit TIFFs are not.** 1,319 train
and 275 val frames have a label and a 16-bit TIFF but no 8-bit JPEG. Since the
experiments compare 8-bit against 16-bit, every arm is built from the
**intersection** of frames present in every source — otherwise the comparison
would partly be measuring dataset size.

**4. The copy has Finder duplicates.** 653 train label files, 401 train TIFFs
and 340 train JPEGs are `"FLIR_01437 2.txt"`-style copy-collision artifacts,
created when the dataset was moved between machines.

These matter more than they look. A duplicate frame trains twice — and because
these only survive in whichever directories the copy touched, they enter some
arms and not others. Nineteen of them sat beside the JPEGs but not the TIFFs,
which is exactly the 19-frame gap that first showed up between the arms.
They are now filtered by pattern in
[`convert.py`](src/thermaldet/convert.py), with
[tests](tests/test_convert.py) holding it.

### The RGB frames cannot serve as a second modality

Comparing thermal against RGB on the same scenes is the obvious second
experiment, and it does not work on 1.3. FLIR's ReadMe says why:

> The thermal and RGB camera did not have identical placement on the vehicle
> and therefore had different viewing geometries, so the thermal annotations do
> not represent the placement of objects in the RGB image.

The RGB frames are 1800×1600 against the thermal 640×512, from a different
position, with no annotations of their own. There is no honest way to train an
RGB detector on them. v2 ships annotated RGB and would support it.

### What survives the audit

| split | frames | boxes | boxes/frame | small % | background frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 7,543 | 59,227 | 7.9 | 58.1 | 844 |
| val | 1,091 | 9,390 | 8.6 | 46.6 | 5 |

"Small" is COCO's convention: box area under 32×32 px. Every number in this
repo is measured on these frames.

**Class balance and object scale (train)**

| class | boxes | share | median box height |
| --- | ---: | ---: | ---: |
| car | 36,209 | 61.1% | 31 px |
| person | 19,931 | 33.7% | 34 px |
| bicycle | 3,087 | 5.2% | 35 px |

Two things follow. The imbalance is **12:1** between `car` and `bicycle`, and
mAP weights classes equally — so every comparison table here prints per-class
AP50 underneath the mean. And the objects are small but *not* aerial-small in a
frame that is only 640 px wide to begin with, which is why input resolution is
not one of the experiments.

**`dog` is excluded by default.** It has 244 training boxes and 16 validation
boxes. A class whose AP is decided by sixteen boxes would move the headline
number more than the variable being tested, and would move it differently
between arms for unrelated reasons. Pass `--classes person bicycle car dog` to
put it back.

**Reproduce this audit:**

```bash
uv run thermaldet convert --arm agc --adopt-labels Dataset/FLIR_ADAS_1_3/yolo/labels
uv run thermaldet profile
```

The converter prints per-split frame and box counts and how many labelled
frames had no image; `profile` writes the full table to
`reports/dataset_profile.md`. Full detail in [docs/00](docs/00-the-dataset.md).

---

## What the pipeline builds

`thermaldet convert` turns the FLIR download into one Ultralytics dataset per
**preprocessing arm**:

```
Dataset/FLIR_ADAS_1_3/        the download, read-only
└── train/, val/, video/

data/                          built by `thermaldet convert`
├── flir_agc/                  FLIR's shipped 8-bit JPEGs
├── flir_global/               16-bit -> 8-bit, one fixed window
└── flir_p1p99/                16-bit -> 8-bit, per-frame stretch
     ├── images/{train,val}/
     ├── labels/{train,val}/
     └── manifest.json         which mapping produced these pixels

configs/data/flir_*.yaml       generated Ultralytics data configs
```

All three arms are built from the same frame list, so they differ only in
pixels. `manifest.json` records the mapping and its parameters, which is how
`thermaldet predict` knows to render a raw TIFF the same way its checkpoint was
trained.

---

## The two experiments

Six runs, two axes, one shared control. Every arm differs from `pretrained` in
**exactly one line** of config, and a [test](tests/test_config.py) asserts that.

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

The 16-bit TIFFs are on disk, so the alternative is testable. Measured over 400
sampled training frames by `thermaldet profile`:

| quantity | counts |
| --- | ---: |
| median frame span (p1–p99) | 873 |
| global window (pooled p0.5–p99.5) | 2,096 |
| span across frames | 2,987 |

A fixed window has to cover every frame, so a median frame reaches
`255 × 873 / 2096` = **106 of 255 output levels**. Keeping absolute radiometry
costs **2.4× of per-frame contrast**. On one real frame:

| arm | mapping | std. dev. | p1 | p99 |
| --- | --- | ---: | ---: | ---: |
| `pretrained` | FLIR AGC, 8-bit | 63.7 | 9 | 249 |
| `global_map` | fixed window, 16-bit | 37.2 | 11 | 184 |
| `p1p99_map` | per-frame stretch, 16-bit | 54.4 | 0 | 255 |

The third arm is what makes the comparison decidable. Without it, `global`
losing to `agc` has two explanations — *per-frame normalisation helps*, or
*FLIR's particular curve helps* — and no way to choose. Details and caveats:
[docs/02](docs/02-radiometry-and-agc.md).

---

## Results

> **Status:** pipeline complete and tested end to end; the six training runs
> have not been executed yet — they run on a rented GPU. Every number
> *elsewhere* in this README is already measured and reproducible today.

All runs: YOLO11s, 60 epochs, 640 px, `batch: 32` on one RTX 4090, evaluated on
the FLIR validation split.

**Transfer**

| run | change | mAP50-95 | mAP50 | vs control | train time |
| --- | --- | ---: | ---: | ---: | ---: |
| `pretrained` (control) | — | | | — | |
| `scratch` | `model: yolo11s.yaml` | | | | |
| `frozen_stem` | `freeze: 2` | | | | |
| `frozen_backbone` | `freeze: 11` | | | | |

**Radiometry**

| run | mapping | mAP50-95 | mAP50 | vs control | train time |
| --- | --- | ---: | ---: | ---: | ---: |
| `pretrained` (control) | FLIR AGC, 8-bit | | | — | |
| `global_map` | fixed window, 16-bit | | | | |
| `p1p99_map` | per-frame stretch, 16-bit | | | | |

Read [docs/03](docs/03-reading-the-metrics.md) before quoting any single number
from here — there is no held-out split in this download.

---

## Command reference

```bash
# data
uv run thermaldet convert --arm agc|global|p1p99 [--adopt-labels DIR]
uv run thermaldet profile

# analysis that needs no training
uv run thermaldet stem-check yolo11s.pt

# training
uv run thermaldet train pretrained [--profile cuda24] [--track wandb]
uv run thermaldet train-manual --backbone-lr-mult 5.0
uv run thermaldet probe pretrained global_map      # cost a run before starting it

# evaluation and deployment
uv run thermaldet eval runs/train/*/weights/best.pt
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
  convert.py      FLIR 1.3 / v2 -> YOLO, and the shared frame list
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
tests/            78 tests, none needing the dataset or a GPU (`make test`)
Dockerfile        pinned training image
```

---

## Design decisions

**Hardware is separate from the experiment.** A config says what to train; a
profile in `configs/profiles/` says what the machine can hold, and
`load_profile` rejects any profile that tries to set `epochs` or `imgsz`. The
payoff is one batch size across all six runs, so the only difference between
two arms is the line their config changed.

**Every arm is built from the same frame list.** Not from whichever images that
arm happens to have — see finding 3 in the audit. There is a
[regression test](tests/test_convert.py) holding it.

**Cheap evidence before expensive evidence.** Both claims have a measurement
that costs seconds and runs before any GPU is rented: `stem-check` for the
transfer argument, `profile` for the radiometry one. The runs test whether
those measurements *matter*, which is the expensive question.

**Runs are traceable.** Each run writes its fully-resolved config — profile
overlay included — to `runs/train/<name>/thermaldet_config.json`, and
Ultralytics' global data and output directories are pinned into the repo rather
than scattered across `$HOME`. `eval` reads that file back to decide which arm
a checkpoint should be scored against.

---

## Documentation

- [00 — The dataset](docs/00-the-dataset.md) — FLIR 1.3 vs v2, label format,
  and the full audit
- [01 — What transfers from COCO](docs/01-what-transfers-from-coco.md) — the
  stem measurement and the four transfer arms
- [02 — Radiometry and AGC](docs/02-radiometry-and-agc.md) — what per-frame
  gain control destroys, and what a fixed window costs
- [03 — Reading the metrics](docs/03-reading-the-metrics.md) — mAP50 vs
  mAP50-95, the class imbalance, and why there is no held-out number here
- [04 — Experiment log](docs/04-experiment-log.md) — running journal, with
  expectations recorded before each run
- [05 — Running on rented GPUs](docs/05-running-on-rented-gpus.md) — profiles,
  calibrating a run before paying for it, and what wastes a rental

## Preserved weights

[`models/flir_yolov8n_finetuned.pt`](models/) is a YOLOv8n checkpoint from an
earlier pass over this dataset, on four classes (`person`, `bicycle`, `car`,
`dog`). It predates everything above and is not part of any ablation. It is
kept because it is a working thermal detector and a fixed reference point.

## Licence

Code: MIT. Ultralytics YOLO is AGPL-3.0 — relevant if you build on this
commercially. FLIR ADAS is released under its own terms for research use; check
upstream before redistributing anything derived from it.
