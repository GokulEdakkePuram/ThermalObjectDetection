# FLIR ADAS v2

[Teledyne FLIR Free ADAS Thermal Dataset v2](https://www.flir.com/oem/adas/adas-dataset-form/):
26,442 annotated frames from a thermal and visible camera pair mounted on a
vehicle, released January 2022.

Everything below was verified against the files rather than read off the
release notes.

## Layout

Six directories, three splits in each of two spectra, each self-describing:

```
Dataset/FLIR_ADAS_v2/
├── images_thermal_train/   coco.json  data/*.jpg  analyticsData/*.tiff
├── images_thermal_val/     coco.json  data/*.jpg  analyticsData/*.tiff
├── video_thermal_test/     coco.json  data/*.jpg  analyticsData/*.tiff
├── images_rgb_train/       coco.json  data/*.jpg
├── images_rgb_val/         coco.json  data/*.jpg
├── video_rgb_test/         coco.json  data/*.jpg
└── rgb_to_thermal_vid_map.json
```

`data/` holds the 8-bit frames the camera writes. **`analyticsData/` holds the
raw 16-bit radiometric frames behind them** — the release notes mention that
directory once, in the Download Contents section, and it is what
[the radiometry ablation](02-radiometry-and-agc.md) needs.

`index.json` sits alongside each `coco.json`. It is FLIR's own Conservator
format, richer but unfiltered; `coco.json` is the cleaned version FLIR
recommends for training, and the one used here. `index.json` is still worth
keeping, because it carries per-video tags that `coco.json` does not — see
[lighting](#lighting-metadata) below.

## What is on disk

| split | frames | annotations | 8-bit | 16-bit |
| --- | ---: | ---: | ---: | ---: |
| `images_thermal_train` | 10,742 | 175,040 | 10,742 | 10,742 |
| `images_thermal_val` | 1,144 | 16,696 | 1,144 | 1,144 |
| `video_thermal_test` | 3,749 | 62,317 | 3,749 | 3,749 |
| `images_rgb_train` | 10,318 | 169,174 | 10,318 | — |
| `images_rgb_val` | 1,085 | 16,909 | 1,085 | — |
| `video_rgb_test` | 3,749 | 84,786 | 3,749 | — |

**The download is clean.** Every `file_name` in every `coco.json` resolves on
disk, every thermal frame has a matching 16-bit TIFF, and there are no
copy-collision duplicates anywhere. Nothing here needs working around.

The test split is real: 3,749 annotated frames sampled — per FLIR — from
*completely independent* video sequences, which is what makes a held-out
measurement meaningful. See [doc 04](04-reading-the-metrics.md).

## Classes

`coco.json` declares all 80 COCO categories. Sixteen are used, and the tail is
unusable:

| class | train boxes | | class | train boxes |
| --- | ---: | --- | --- | ---: |
| car | 73,623 | | truck | 829 |
| person | 50,478 | | skateboard | 29 |
| sign | 20,770 | | stroller | 15 |
| light | 16,198 | | scooter | 15 |
| bike | 7,237 | | deer | 8 |
| bus | 2,245 | | train | 5 |
| other vehicle | 1,373 | | dog | 4 |
| motor | 1,116 | | | |
| hydrant | 1,095 | | | |

**Five are kept by default: `person`, `bike`, `car`, `light`, `sign`.** mAP
weights every class equally, so `dog` at four training boxes would carry the
same weight as `car` at 73,623 — and would move between ablation arms for
reasons that have nothing to do with the arms.

`light` and `sign` are in the default set deliberately, rather than stopping at
the obvious three. They are the two large classes a thermal sensor should find
*harder* than a visible one: a traffic light signals with colour and a street
sign with printed contrast, and an infrared sensor sees neither. Without them
[the modality comparison](03-thermal-vs-visible.md) only asks a question it
already knows the answer to.

Categories are matched by **name**, not id. v2 keeps COCO's original numbering,
so the ids are neither contiguous nor in the order the YOLO indices need
(`light` is 10, `sign` is 12) — and thermal and visible have to produce the
same index for the same class or the modality comparison is comparing two
different label spaces.

## Profile

```bash
uv run thermaldet convert --arm agc
uv run thermaldet profile
```

**Thermal**, five classes:

| split | frames | boxes | boxes/frame | small % | background frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 10,742 | 168,238 | 15.7 | 74.6 | 280 |
| val | 1,144 | 16,244 | 14.2 | 75.1 | 19 |
| test | 3,749 | 55,371 | 14.8 | **87.3** | 256 |

"Small" is COCO's convention: box area under 32×32 px. Three things follow.

**The scenes are dense** — 15.7 objects per frame, against 7 on COCO.

**The objects are small.** Three quarters of them are "small" by COCO's
definition, in a frame only 640 px wide to begin with.

| class | train boxes | share | median box height |
| --- | ---: | ---: | ---: |
| car | 73,622 | 43.8% | 22 px |
| person | 50,474 | 30.0% | 26 px |
| sign | 20,747 | 12.3% | **11 px** |
| light | 16,158 | 9.6% | **16 px** |
| bike | 7,237 | 4.3% | 30 px |

An 11-pixel sign sits close to the floor of what a stride-8 detection head can
resolve. That is worth knowing before reading any per-class number.

**The test split is harder than training.** 87.3% small against 74.6%, because
the test videos were sampled at 30 fps from continuous footage rather than
curated for detector training. Expect the held-out number to drop by more than
the usual selection bias alone would explain.

## Lighting metadata

Each image record carries `extra_info` with `hours`, `scene`, `weather` and
`video_id`. Coverage of `hours` is uneven, and on the test split it is absent
entirely:

| split | frames | `hours` present |
| --- | ---: | ---: |
| `images_thermal_train` | 10,742 | 3,144 (29%) |
| `images_thermal_val` | 1,144 | 236 (21%) |
| `video_thermal_test` | 3,749 | **0** |
| `images_rgb_train` | 10,318 | 10,266 (99.5%) |

`index.json` fills most of the gap: it carries per-video `tags`, and 6 of the 8
test videos, 126 of 133 train videos and 16 of 17 val videos are tagged with a
lighting condition. Since FLIR sampled each video under one condition,
propagating the video-level tag to its frames is defensible — as an
**imputation**, stated as one.

Test split by video tag: night 1,497, dawn 1,033, day 428, **untagged 791**.

Two caveats for anyone who wants to split results by lighting. Ten of the 133
training videos carry *conflicting* `hours` values across their own frames and
should be dropped rather than guessed at. And 21% of the test split has no tag
at all. No experiment in this repo currently splits by lighting; the metadata
is documented because it is there and it is the obvious next axis.

## Two things that constrain what can be compared

**The cameras differ.** Thermal is a Tau 2 at 45° horizontal field of view,
640×512. Visible is a BlackFly S at 52.8°, and its frames come in five
resolutions from 720×480 to 2048×1536. So the two spectra are **not**
pixel-aligned and their annotation counts differ. What that does and does not
allow is [doc 03](03-thermal-vs-visible.md).

**The test videos are not one camera either.** Six of the eight are Tau 2 at
45°; two are a Boson ADK at 50°. A hardware shift inside the held-out split is
worth knowing about before attributing a per-video difference to the model.

## Label format

YOLO wants `class cx cy w h`, normalised. Two details in the conversion fail
quietly if you get them wrong:

- **Boxes are clipped to the frame**, and boxes under a pixel wide after
  clipping are dropped. FLIR ships a handful of zero-width boxes; Ultralytics
  turns those into NaN loss an hour into a run.
- **A frame with no kept object gets an empty `.txt`, not no file.**
  Ultralytics reads an empty label as background, which is what it is. Omitting
  the file throws away a negative, and 280 training frames are background.
