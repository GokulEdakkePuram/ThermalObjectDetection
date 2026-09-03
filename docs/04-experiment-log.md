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

## 2026-09-03 — Dataset audit

**Question.** Is the FLIR download usable as it stands?

**Result.** Partly. No `thermal_annotations.json` in any split; the 8-bit
JPEGs are 14.9% short in train and 20.1% in val; the 16-bit TIFFs and the
pre-converted YOLO labels are complete. The copy also carries ~1,500 Finder
`" 2"` duplicates spread unevenly across the source directories. Usable via
`--adopt-labels`, at 7,543 train / 1,091 val frames after filtering duplicates
and intersecting the sources. No test split.

**Reading.** The missing `video/` annotations are the real cost -- it removes
the held-out measurement entirely. Everything downstream is validation-only
and should be read as optimistic.

The duplicates were nearly missed, and would have shown up later as an
unexplained 19-frame gap between two arms. Worth remembering that a dataset
which has been copied around is not the dataset that was published.

**Next.** Measure the radiometric dynamic range, which decides whether the
`global` arm is worth building at all.

---

## 2026-09-03 — Radiometric dynamic range

**Question.** What does a fixed global 16-bit to 8-bit window cost in contrast?

**Expectation.** A large fraction of the output range. Thermal scenes occupy a
narrow band of raw counts, and the band moves between frames.

**Result.** Median frame span 873 counts; global window 2,096; so a median
frame reaches 106 of 255 levels. A 2.4x contrast loss. On `FLIR_00001`,
per-frame std. dev. falls from 63.7 (AGC) to 37.2 (global).

**Reading.** Smaller than expected -- 106 levels is a real loss but not a
crippling one, so the radiometry ablation is a genuine trade rather than a
foregone conclusion. Worth the GPU time.

**Next.** The stem measurement, same logic: cheap evidence before expensive
evidence.

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

## 2026-09-03 — Pipeline smoke test

**Question.** Does the whole path -- convert, profile, train, evaluate -- run
end to end before any real compute is committed to it?

**Expectation.** Yes, with meaningless metrics (1 epoch, 2% of the data).

**Setup.** `configs/smoke.yaml`, YOLO11n, 640 px, MPS on an M2 Pro.

**Result.** Ran. mAP50 0.096, mAP50-95 0.047 after one epoch on 150 frames;
`car` AP50 0.284 against `person` 0.003 and `bicycle` 0.000. Resolved config
written next to the weights; `eval` recovered the arm from it.

**Reading.** Plumbing confirmed. The per-class spread is what one epoch on 2%
of the data looks like -- `car` is 61% of the boxes and the largest object, so
it is the only class with any signal yet. Nothing here says anything about the
model.

**Next.** `thermaldet probe pretrained global_map` on the rented box, then the
sweep.

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

**Expectation.** Recorded in [doc 02](02-radiometry-and-agc.md), in advance:
`global` loses to `agc`; `p1p99` lands close to `agc`.

**Result.** _to fill in_
