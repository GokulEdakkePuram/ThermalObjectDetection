# FLIR ADAS, and which release you actually have

FLIR has released this dataset twice, and the two are not interchangeable.

| | ADAS 1.3 | ADAS v2 |
| --- | --- | --- |
| splits | `train/`, `val/`, `video/` | `images_thermal_train/`, `images_thermal_val/`, `video_thermal_test/` |
| annotations | `<split>/thermal_annotations.json` | `<split>/coco.json` |
| 8-bit thermal | `thermal_8_bit/*.jpeg` | `data/*.jpg` |
| **16-bit thermal** | `thermal_16_bit/*.tiff` | **not shipped** |
| paired RGB | `RGB/*.jpg`, unannotated | annotated, in `images_rgb_*` |
| classes | person, bicycle, car, dog | person, bike, car |

Both are COCO underneath, so [`convert.py`](../src/thermaldet/convert.py)
detects which one it is looking at and reads either. Categories are matched by
**name**, never by id: 1.3 keeps COCO's original numbering (`person=1`,
`car=3`, `dog=18`) and v2 renumbered and renamed them, so any hard-coded id
map is silently wrong on one of the two.

**This project uses 1.3**, for one reason: it ships the raw 16-bit radiometric
TIFFs, and the difference between those and the 8-bit JPEGs is
[the second ablation](02-radiometry-and-agc.md). v2 has more frames and no
16-bit, so it supports the transfer ablation only.

## What the frames are

640x512, single channel, from a vehicle-mounted FLIR Tau 2 on public roads.
The 8-bit JPEGs are 0-255 and the 16-bit TIFFs are raw sensor counts, which
land in a narrow band around 6,000-8,000 out of a possible 65,535.

## The RGB frames are not usable as a second modality

This was the obvious second axis -- same scenes, two sensors, measure what
thermal buys you -- and it does not work on 1.3. FLIR's own ReadMe says why:

> The thermal and RGB camera did not have identical placement on the vehicle
> and therefore had different viewing geometries, so the thermal annotations do
> not represent the placement of objects in the RGB image.

The RGB frames are 1800x1600 against the thermal 640x512, from a different
position, with no annotations of their own. There is no honest way to train an
RGB detector on them, so there is no RGB arm here. v2 ships annotated RGB and
would support it.

## The class set

1.3 annotates four classes. Three are used:

| class | train boxes | val boxes |
| --- | ---: | ---: |
| car | 36,209 | — |
| person | 19,931 | — |
| bicycle | 3,087 | — |
| **dog** | **244** | **16** |

`dog` is excluded by default. mAP weights every class equally, so a class
whose AP is decided by sixteen validation boxes would move the headline number
by more than the thing being ablated -- and it would move it differently
between arms, for reasons that have nothing to do with the arms. Pass
`--classes person bicycle car dog` to put it back.

## The label format, and the trap in it

YOLO wants `class cx cy w h`, normalised. Two details in the conversion are
worth stating because getting them wrong fails quietly:

- **Boxes that overhang the sensor edge** are clipped, and boxes narrower than
  a pixel after clipping are dropped. FLIR ships a handful of zero-width
  boxes; Ultralytics turns those into NaN loss somewhere in the first epoch.
- **A frame with no surviving object gets an empty `.txt`, not no file.**
  Ultralytics reads an empty label as a background frame, which is exactly
  what it is. Omitting the file instead throws away a negative example, and
  844 of the 7,543 training frames are background.

## A download with pieces missing

The copy this repo was developed against is incomplete, in a way worth
documenting because it shaped the code:

- **No `thermal_annotations.json` at all**, in any split. The converter's
  primary path cannot run.
- **Pre-converted YOLO labels survived** for train and val, covering the full
  id range: 8,862 and 1,366 unique stems, matching FLIR's own counts.
- **The 16-bit TIFFs are complete** — 8,862 train, 1,366 val.
- **The 8-bit JPEGs are short**: 7,543 unique of 8,862 in train (14.9%
  missing), 1,091 of 1,366 in val (20.1% missing).
- **The copy is littered with Finder duplicates** — `"FLIR_01437 2.jpeg"` and
  friends, created by moving the dataset between machines. 653 train labels,
  401 train TIFFs and 340 train JPEGs; 95 val labels.

Per source, counted:

| split | source | files | unique | `" 2"` copies |
| --- | --- | ---: | ---: | ---: |
| train | YOLO labels | 9,515 | 8,862 | 653 |
| train | 16-bit TIFF | 9,263 | 8,862 | 401 |
| train | 8-bit JPEG | 7,883 | 7,543 | 340 |
| val | YOLO labels | 1,461 | 1,366 | 95 |
| val | 16-bit TIFF | 1,366 | 1,366 | 0 |
| val | 8-bit JPEG | 1,091 | 1,091 | 0 |

The duplicates are not cosmetic. A duplicate frame trains twice, and one that
survives in only some directories trains twice in only some arms -- which is
precisely what happened before they were filtered (see
[doc 02](02-radiometry-and-agc.md)). Nothing legitimate in FLIR ends in a
space and a number, so they are dropped by pattern in `_stems` and
`adopted_labels`.

Three features exist because of this. `--adopt-labels` reads an existing YOLO
label directory instead of a COCO JSON, remapping class ids to the requested
set (the source order has to be named, since nothing in a `.txt` records it).
Copy-collision duplicates are filtered by pattern. And the frame list is
intersected across *every* image source rather than the one the current arm
needs -- see [`frame_index`](../src/thermaldet/convert.py).

Net result, and what every number in this repo is measured on:

| split | frames | boxes | background frames |
| --- | ---: | ---: | ---: |
| train | 7,543 | 59,227 | 844 |
| val | 1,091 | 9,390 | 5 |

**There is no test split**, because `video/thermal_annotations.json` is one of
the missing files. That is a real limitation and it is called out in
[doc 03](03-reading-the-metrics.md): every number here is validation, and
`best.pt` is selected on validation, so every number is optimistic. Recovering
that one JSON would fix it, and the converter already handles it.

## Profile

```bash
uv run thermaldet profile
```

| split | frames | boxes | boxes/frame | small % | medium % | large % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 7,543 | 59,227 | 7.9 | 58.1 | 35.8 | 6.1 |
| val | 1,091 | 9,390 | 8.6 | 46.6 | 44.9 | 8.5 |

"Small" is COCO's convention: box area under 32x32 px. Median box heights are
31 px for `car`, 34 px for `person`, 35 px for `bicycle` -- so the objects are
small but not *aerial*-small, and unlike a drone dataset the frames are 640 px
wide to begin with. **Input resolution is therefore not the story here**, which
is what makes the two axes this repo does test the interesting ones.
