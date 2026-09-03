# thermaldet — what actually transfers when you fine-tune a detector for thermal

Fine-tuning YOLO11 on [FLIR ADAS](https://www.flir.com/oem/adas/adas-dataset-form/)
thermal infrared, where the input is a 640x512 single-channel radiometer and
the pretrained weights have only ever seen daylight RGB.

The interesting part of this problem is not calling `.train()`. It is that two
things everybody does without thinking — start from COCO weights, and train on
the 8-bit JPEGs the camera writes — are both *choices*, and both are testable.
This repo is a controlled attempt to test them, with the reasoning written down
in [`docs/`](docs/) and the predictions recorded before the runs.

Two claims, one experiment each:

1. **A third of a COCO-pretrained first layer cannot respond to thermal at
   all.** Thermal is single-channel, replicated across three to fit an RGB
   network, so every colour-opponent filter cancels. Measured: 10 of YOLO11s's
   32 stem filters. If that matters, freezing the backbone — standard
   fine-tuning advice — should be unusually expensive here.

2. **The 8-bit thermal frames everyone trains on are not the sensor's
   output.** They are per-frame automatic gain control, which throws away the
   absolute measurement that is a thermal camera's entire point. The raw 16-bit
   is on disk. Whether keeping it is worth the 2.4x of contrast it costs is a
   measurement, not an opinion.

## Results

> **Status:** pipeline complete, tested end to end, and calibrated. The six
> training runs have not been executed yet — they run on a rented GPU. The
> tables below are filled in from `reports/results.md` as runs finish, and the
> commands that produce them are the ones in this README.
>
> Everything stated as a number *outside* these two tables is already measured
> and reproducible today.

All runs: YOLO11s, 60 epochs, 640 px, `batch: 32` on one RTX 4090, evaluated on
the FLIR validation split.

**Transfer** (only the named line differs from the control):

| run | change | mAP50-95 | mAP50 | vs control | train time |
| --- | --- | ---: | ---: | ---: | ---: |
| `pretrained` (control) | — | | | — | |
| `scratch` | `model: yolo11s.yaml` | | | | |
| `frozen_stem` | `freeze: 2` | | | | |
| `frozen_backbone` | `freeze: 11` | | | | |

**Radiometry** (only the 16-bit → 8-bit mapping differs):

| run | mapping | mAP50-95 | mAP50 | vs control | train time |
| --- | --- | ---: | ---: | ---: | ---: |
| `pretrained` (control) | FLIR AGC, 8-bit | | | — | |
| `global_map` | fixed window, 16-bit | | | | |
| `p1p99_map` | per-frame stretch, 16-bit | | | | |

Predictions for both are recorded in [docs/01](docs/01-what-transfers-from-coco.md)
and [docs/02](docs/02-radiometry-and-agc.md), along with what would falsify
them. `best.pt` is selected on validation and there is no held-out test split
in this download — see [docs/03](docs/03-reading-the-metrics.md) before quoting
any single number from here.

## The first argument, measured without training anything

A thermal frame has one channel. Ultralytics replicates it to three, so every
first-layer filter sees the same image three times and its response is governed
by the sum of its channel slices. An achromatic edge detector reinforces. A
colour-opponent filter — red-minus-green, the kind that finds a brake light —
cancels, and outputs approximately zero on every thermal frame it will ever be
shown.

```bash
uv run thermaldet stem-check yolo11s.pt
```

```
stem: 32 filters
  grey-input response below 0.1: 10 filters (31%)
  median response ratio: 0.93 (1.0 = channel slices reinforce, 0.0 = they cancel)
```

**31% of the first layer is inert before a single gradient step**, and the
median filter sits at 0.93 — so the stem does not degrade uniformly, it splits
into a majority that transfers intact and a chromatic third that cannot.

That establishes those filters are inert. Whether the network *misses* them is
what `frozen_stem` costs GPU time to find out. Full argument in
[docs/01](docs/01-what-transfers-from-coco.md).

## The second argument, also measured first

FLIR's 8-bit JPEGs are AGC output: each frame's own range stretched to fill
0–255. The 1.3 release ships the raw 16-bit TIFFs alongside, so the
alternative is testable.

```bash
uv run thermaldet profile
```

| quantity | counts |
| --- | ---: |
| median frame span (p1–p99) | 873 |
| global window (pooled p0.5–p99.5) | 2,096 |
| span across frames | 2,987 |

A fixed window has to cover every frame, so a median frame reaches
`255 × 873 / 2096` = **106 of 255 output levels**. Absolute radiometry costs
**2.4× of per-frame contrast**. On one real frame:

| arm | std. dev. | p1 | p99 |
| --- | ---: | ---: | ---: |
| `agc` (FLIR's JPEG) | 63.7 | 9 | 249 |
| `global` (fixed window) | 37.2 | 11 | 184 |
| `p1p99` (per-frame stretch) | 54.4 | 0 | 255 |

Whether that trade is worth making is the ablation. Details and caveats in
[docs/02](docs/02-radiometry-and-agc.md).

## The dataset

FLIR ADAS **1.3** (not v2), because 1.3 is the release that ships the 16-bit
imagery. Three classes: `person`, `bicycle`, `car`.

| split | frames | boxes | boxes/frame | small % | background frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 7,543 | 59,227 | 7.9 | 58.1 | 844 |
| val | 1,091 | 9,390 | 8.6 | 46.6 | 5 |

Median box heights are 31 px (`car`), 34 px (`person`), 35 px (`bicycle`) in a
640×512 frame. `dog` is annotated but excluded: 244 training boxes against
36,209 for `car`, and mAP weights classes equally.

The download this was built against is missing its annotation JSONs and ~15% of
its 8-bit JPEGs; the numbers above are after intersecting every source. What
that means for the code — and why there is no test split — is in
[docs/00](docs/00-the-dataset.md).

## Quickstart

```bash
make setup      # uv sync
make arms       # build all three preprocessing arms (~1 min, ~3.5 GB)
make profile    # class balance, object scale, dynamic range -> reports/
make smoke      # 1-epoch run to verify the pipeline end to end
make pretrained # the control
make eval       # comparison table -> reports/results.md
```

If the download's `thermal_annotations.json` files are missing but converted
YOLO labels survived:

```bash
make arms ADOPT="--adopt-labels Dataset/FLIR_ADAS_1_3/yolo/labels"
```

On a rented GPU, provision with one command and calibrate before spending
hours — see [docs/05](docs/05-running-on-rented-gpus.md):

```bash
curl -fsSL https://raw.githubusercontent.com/GokulEdakkePuram/ThermalObjectDetection/main/scripts/setup_remote.sh | bash
uv run thermaldet probe pretrained global_map
tmux new -s train && ./scripts/run_sweep.sh
```

Everything is reachable directly:

```bash
uv run thermaldet convert --arm global          # build one arm
uv run thermaldet stem-check yolo11s.pt         # what survives greyscale input
uv run thermaldet train frozen_stem --track wandb
uv run thermaldet probe pretrained              # cost a run before starting it
uv run thermaldet eval runs/train/*/weights/best.pt
uv run thermaldet predict <weights> <frame.tiff>   # raw input is rendered first
uv run thermaldet export <weights> --format onnx
uv run thermaldet train-manual --backbone-lr-mult 5.0
```

## How it is put together

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
tests/            config, conversion, radiometry and stem tests (`make test`)
Dockerfile        pinned training image
```

Four deliberate choices:

**Hardware is not part of an experiment.** An experiment config says what to
train; a profile in `configs/profiles/` says what the machine can hold, and
`load_profile` rejects any profile that tries to set `epochs` or `imgsz`. The
payoff is one batch size across all six runs on one card, so the only thing
that differs between two arms is the line their config changed.

**Every arm is built from the same frame list.** Not from whichever images that
arm happens to have. This download is missing ~15% of the 8-bit JPEGs *and* 19
frames of 16-bit that the JPEGs have, so per-arm intersection gave 7,562 frames
against 7,543 — a difference small enough to be invisible and large enough to
make a 1% gap unattributable. There is a
[regression test](tests/test_convert.py) holding it.

**Cheap evidence before expensive evidence.** Both headline claims have a
measurement that costs seconds and runs before any GPU is rented: `stem-check`
for the transfer argument, `profile` for the radiometry one. Both are in this
README with real numbers. The training runs test whether those measurements
*matter*, which is a different and more expensive question.

**Runs are traceable.** Each run writes its fully-resolved config — profile
overlay included — to `runs/train/<name>/thermaldet_config.json`, and
Ultralytics' global data and output directories are pinned into the repo rather
than scattered across `$HOME`. `eval` reads that file back to decide which
preprocessing arm a checkpoint should be scored against.

## Notes

Written as the work was done, not reconstructed afterwards:

- [00 — The dataset](docs/00-the-dataset.md) — FLIR 1.3 vs v2, the label
  format, why the RGB frames cannot serve as a second modality, and what is
  missing from this download
- [01 — What transfers from COCO](docs/01-what-transfers-from-coco.md) — the
  stem measurement, the four transfer arms, and what would falsify them
- [02 — Radiometry and AGC](docs/02-radiometry-and-agc.md) — what per-frame
  gain control destroys, what a fixed window costs, and the confound that had
  to be removed first
- [03 — Reading the metrics](docs/03-reading-the-metrics.md) — mAP50 vs
  mAP50-95, a 12:1 class imbalance, and why there is no held-out number here
- [04 — Experiment log](docs/04-experiment-log.md) — running journal, with
  expectations recorded before each run
- [05 — Running on rented GPUs](docs/05-running-on-rented-gpus.md) — hardware
  profiles, calibrating a run before paying for it, and what wastes a rental

## Preserved weights

[`models/flir_yolov8n_finetuned.pt`](models/) is a YOLOv8n checkpoint from an
earlier pass over this dataset, on four classes (`person`, `bicycle`, `car`,
`dog`). It predates everything above and is not part of any ablation — it is
kept because it is a working thermal detector and a fixed reference point.

## Licence

Code: MIT. Ultralytics YOLO is AGPL-3.0 — relevant if you build on this
commercially. FLIR ADAS is released under its own terms for research use;
check upstream before redistributing anything derived from it.
