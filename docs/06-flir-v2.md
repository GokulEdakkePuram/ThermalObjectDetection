# FLIR ADAS v2: feasibility assessment

Audit of the v2 download, and what it changes for the experiments in this repo.

**Verdict: v2 is better than 1.3 on every axis that matters here, and it fixes
the one real limitation of the current setup — the missing test split.** It
also makes the RGB-vs-thermal experiment possible, which 1.3 could not
support. Two things need care: the class set, and the uneven day/night
labelling.

## What is on disk

Every count below was verified against the files, not read off the release
notes.

| split | frames | annotations | 8-bit JPG | 16-bit TIFF |
| --- | ---: | ---: | ---: | ---: |
| `images_thermal_train` | 10,742 | 175,040 | 10,742 | 10,742 |
| `images_thermal_val` | 1,144 | 16,696 | 1,144 | 1,144 |
| `video_thermal_test` | 3,749 | 62,317 | 3,749 | 3,749 |
| `images_rgb_train` | 10,318 | 169,174 | 10,318 | — |
| `images_rgb_val` | 1,085 | 16,909 | 1,085 | — |
| `video_rgb_test` | 3,749 | 84,786 | 3,749 | — |

**The download is clean.** Zero missing files: every `file_name` in every
`coco.json` resolves on disk, and every thermal frame has a matching 16-bit
TIFF. Zero copy-collision duplicates. This is the opposite of the 1.3 copy,
which was [missing its annotations entirely](00-the-dataset.md).

`analyticsData/` is where v2 keeps the 16-bit radiometric frames. The release
notes mention it once, in the Download Contents section, which is why this
repo initially recorded v2 as shipping no 16-bit data. That was wrong.

## What it fixes

**A real held-out test split.** 3,749 annotated frames from eight video
sequences, sampled — per FLIR — from *completely independent* footage. This is
the single biggest gain. Every number this repo currently produces is
validation-only with `best.pt` also selected on validation
([doc 03](03-reading-the-metrics.md)); v2 removes that caveat entirely.

**More training data, cleanly.** 10,742 thermal training frames against 7,543
usable on the 1.3 copy — **+42%** — with no intersection games, because
nothing is missing.

**The converter already reads it.** No code change was needed for the 8-bit
path; `detect_layout` recognises the layout and all three splits convert:

```
train  images_thermal_train   frames=10,742  boxes=131,333
val    images_thermal_val     frames= 1,144  boxes= 11,773
test   video_thermal_test     frames= 3,749  boxes= 42,953
```

(box counts for the three-class subset `person`, `bike`, `car`)

## What it makes newly possible

### RGB vs thermal — the experiment 1.3 could not support

On 1.3 the RGB frames are unannotated and from a different viewing geometry,
so there was [no honest way](00-the-dataset.md) to train an RGB detector.
v2 annotates both spectra with the same 15-class label map, and the same
splits.

The cameras are still different — thermal is a Tau 2 at 45° HFOV, RGB a
BlackFly S at 52.8° — so frames are **not** pixel-aligned and the annotation
counts differ (175,040 thermal vs 169,174 RGB in train). That rules out a
paired per-object comparison on the image splits, but not a detector-level
one: train the same architecture on each spectrum, evaluate each on its own
test split, compare.

For the *video* test split it is better than that.
`rgb_to_thermal_vid_map.json` gives 3,749 time-synced RGB↔thermal frame pairs,
and all 3,749 resolve on both sides. That is a genuinely paired comparison on
identical moments.

FLIR also publishes a baseline to sit beside: YOLOX-m, COCO-pretrained,
AP@IoU=0.5 on this test set.

| | person | car |
| --- | ---: | ---: |
| RGB | 51.42 | 55.79 |
| Thermal | **75.33** | **77.23** |

A published external reference point is worth a lot for a portfolio repo — it
turns "my number is 0.7" into "my number is 0.7 against a published 0.75 under
the same protocol".

### Day and night — feasible, with an imputation

This is the one that needs care, so the numbers are given in full.

Each image record carries `extra_info` with `hours`, `scene`, `weather` and
`video_id`. Coverage of `hours` is uneven:

| split | frames | `hours` present |
| --- | ---: | ---: |
| `images_thermal_train` | 10,742 | 3,144 (29%) |
| `images_thermal_val` | 1,144 | 236 (21%) |
| `video_thermal_test` | 3,749 | **0 (0%)** |
| `images_rgb_train` | 10,318 | 10,266 (99.5%) |

Taken at face value, day/night is impossible on the test split. But
`index.json` carries per-video `tags`, and those do cover it: 6 of the 8 test
videos are tagged, as are 126 of 133 train videos and 16 of 17 val videos.
Since FLIR sampled each video under one lighting condition, propagating the
video-level tag to its frames is defensible — and it has to be *stated* as an
imputation rather than presented as a label.

Test split, by video tag:

| lighting | frames |
| --- | ---: |
| night | 1,497 |
| dawn | 1,033 |
| day | 428 |
| untagged | 791 |

Two caveats. **10 of the 133 training videos carry conflicting `hours` values
across their own frames**, so the propagation is not clean everywhere and
those videos should be dropped from any day/night split rather than guessed
at. And **791 test frames (21%) have no lighting tag at all**, so a day/night
result on the test split is computed on 79% of it.

### The radiometry ablation gets sharper

v2's 16-bit is complete, so both arms trivially see the same frames — no
intersection required. Measured over 400 sampled training frames:

| | 1.3 | v2 |
| --- | ---: | ---: |
| median frame span (p1–p99) | 873 | 764 |
| global window (pooled p0.5–p99.5) | 2,096 | 2,219 |
| levels a median frame gets under the global map | 106 / 255 | **88 / 255** |
| implied contrast loss | 2.4× | **2.9×** |

So the trade the ablation tests is *harder* on v2 — more contrast given up for
the same absolute radiometry. The prediction in
[doc 02](02-radiometry-and-agc.md) is unchanged in direction and stronger in
magnitude.

## What needs deciding

**The class set.** `coco.json` declares all 80 COCO categories but only 16 are
used, and the tail is unusable:

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

Three options, in order of what I would pick:

1. **`person`, `bike`, `car`** — keeps continuity with the 1.3 results and
   matches FLIR's own baseline reporting (person and car). 131,333 train
   boxes.
2. **Add `light` and `sign`** — the two large classes thermal ought to find
   *harder* than RGB, since a traffic light's signal is colour and a sign's is
   printed contrast, neither of which survives a thermal sensor. That makes
   the RGB-vs-thermal comparison much more interesting than person-and-car
   alone, where thermal obviously wins.
3. All 16 — not advisable. Four classes with under 30 boxes would each carry
   1/16th of the mAP.

Option 2 is the one that turns the modality experiment from a confirmation
into a question with a real answer either way.

**Whether to keep 1.3 support.** The converter reads both and the cost of
keeping it is one dictionary entry. The 1.3 results are also the only ones
that exist right now. Recommend keeping both readable and moving the
experiments to v2.

**What has to be redone.** Migrating means re-running everything, since the
dataset, class set and splits all change: the arms, the profile, the two
prediction docs' measured numbers, the dataset section of the README, and the
audit above becomes the primary audit rather than an appendix. The `stem-check`
result is unaffected — it is a property of the pretrained weights, not the
data.

## Summary

| | 1.3 (current) | v2 |
| --- | --- | --- |
| annotations | **missing entirely** | complete, all 6 splits |
| thermal train frames | 7,543 usable | 10,742 |
| held-out test split | **none** | 3,749 annotated frames |
| 16-bit thermal | complete | complete |
| annotated RGB | no | yes, both splits + paired video |
| day/night metadata | none | partial, recoverable per video |
| file integrity | ~1,500 duplicates, 15% of JPEGs missing | clean |
| classes | 4 (3 usable) | 16 used (5–6 usable) |
| published baseline to compare against | no | yes (YOLOX-m) |
