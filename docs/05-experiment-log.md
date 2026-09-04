# Experiment log

A running journal. The point is to record what was expected *before* each run,
so that being wrong stays visible instead of getting quietly rewritten into a
tidy narrative afterwards.

Template:

---

## YYYY-MM-DD — <short title>

**Question.** What am I trying to find out?

**Expectation.** What I think will happen, and why. Written before running.

**Setup.** Config, commit SHA, hardware, wall-clock time.

**Result.** Numbers. Link the run directory.

**Reading.** Did the expectation hold? If not, what is the most likely
explanation, and what would distinguish between the candidates?

**Next.** The single most informative experiment to run after this one.

---

## 2026-09-03 — Dataset audit (FLIR 1.3)

**Question.** Is the download usable as it stands?

**Result.** Partly. No `thermal_annotations.json` in any split, ~15% of the
8-bit JPEGs missing, ~1,500 Finder `" 2"` duplicates spread unevenly across
directories, and no annotated test split. Usable at 7,543 train / 1,091 val
frames after filtering and intersecting.

**Reading.** The missing `video/` annotations were the real cost -- no
held-out measurement at all. Superseded three days later; kept here because
the duplicate-filtering and frame-list code it forced still earns its place.

**Next.** Measure the radiometric dynamic range.

---

## 2026-09-03 — Stem transferability

**Question.** How much of a COCO-pretrained first layer can respond at all to a
single-channel input replicated across three channels?

**Expectation.** Some meaningful fraction dead, because colour-opponent filters
must cancel. No prior on the size of it.

**Result.** 10 of 32 YOLO11s stem filters (31%) below a 0.1 grey-input
response ratio. Median filter 0.93.

**Reading.** The bimodality is the interesting part and was not predicted: the
stem splits into an achromatic majority that transfers intact and a chromatic
third that is inert, rather than degrading uniformly. This is what makes
`frozen_stem` worth a separate arm from `frozen_backbone`.

Note this establishes only that those filters are *inert*, not that the network
*needs* them. `frozen_stem` is what tests the second claim.

**Next.** `smoke`, then probe, then the six real runs.

---

## 2026-09-04 — Moved to FLIR ADAS v2

**Question.** Does v2 support the experiments better than 1.3?

**Expectation.** Better on data volume, worse or absent on 16-bit -- I had
recorded v2 as shipping no radiometric imagery.

**Result.** Wrong on the 16-bit, and better than expected everywhere else. v2
keeps the raw frames under `analyticsData/`, complete and one-for-one with the
8-bit. It also ships a genuine held-out test split (3,749 frames from
independent sequences), 10,742 thermal training frames against 7,543 usable on
1.3, annotated visible imagery, and a clean download -- no missing files, no
duplicates.

**Reading.** The v2 release notes mention `analyticsData` once, in a contents
list, and I took the absence of a `thermal_16_bit/` directory as the answer
without checking. That claim had been sitting in the README as fact.

Two things follow. The held-out split removes the largest caveat on every
number this repo will produce. And the modality experiment -- thermal against
visible, which 1.3 could not support because its RGB frames are unannotated --
becomes possible, so it is back in as a third axis.

**Setup.** Five classes now: `person`, `bike`, `car`, `light`, `sign`. The last
two are there to make the modality comparison a question rather than a
formality.

**Next.** Re-measure the radiometry on v2, then the smoke test, then the sweep.

---

## 2026-09-04 — Radiometric dynamic range on v2

**Question.** Does the AGC trade look the same on v2 as on 1.3?

**Expectation.** Similar. Same sensor family, same T-linear mode.

**Result.** Sharper. Median frame span 764 counts against a 2,219-count global
window, so a median frame reaches 88 of 255 levels -- a 2.9x contrast loss,
against 2.4x on 1.3.

**Reading.** v2 draws from more locations (England, France, Michigan, Idaho)
than 1.3's Santa Barbara footage, so the spread *across* frames is wider while
each individual frame is no wider. That is exactly the shape that makes a
fixed global window expensive, and it strengthens the prediction rather than
changing it.

**Next.** The seven runs.

---

## <pending> — The transfer ablation

**Question.** How much of COCO pretraining survives the move to thermal, and
where does it stop surviving?

**Expectation.** Recorded in [doc 01](01-what-transfers-from-coco.md), in
advance: `pretrained` > `scratch`, but by less than the RGB rule of thumb;
`frozen_stem` costs disproportionately more than freezing two of twenty-three
layers should; `frozen_backbone` worse again; and possibly `scratch` beating
`frozen_backbone`, which would be the striking result.

**Falsified if** `frozen_stem` lands within noise of `pretrained`.

**Result.** _to fill in_

---

## <pending> — The radiometry ablation

**Question.** Is FLIR's per-frame AGC discarding usable signal, and is what it
contributes the normalisation or the particular curve?

**Expectation.** Recorded in [doc 02](02-radiometry-and-agc.md): `global`
loses to `agc`; `p1p99` lands close to `agc`.

**Result.** _to fill in_

---

## <pending> — The modality ablation

**Question.** Given one 640 px input budget, which sensor detects better?

**Expectation.** Recorded in [doc 03](03-thermal-vs-visible.md), per class
rather than in aggregate: thermal clearly ahead on `person` and `car`, visible
clearly ahead on `light` and `sign`, `bike` unclear. FLIR's own YOLOX-m
baseline puts thermal +24 AP50 on person and +21 on car.

**Falsified if** thermal wins on all five. That would point at the resize or
the 4% frame-count difference rather than at the sensor.

**Result.** _to fill in_
