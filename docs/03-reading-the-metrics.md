# Reading the metrics honestly

## What mAP averages

**AP** for one class is the area under its precision-recall curve, swept by
varying the confidence threshold. **mAP** averages that over classes. The two
numbers reported everywhere differ only in the IoU threshold used to decide
whether a prediction counts as a hit:

- **mAP50** — a box counts if IoU >= 0.5. *"Did you find the thing?"*
- **mAP50-95** — averaged over thresholds 0.5, 0.55, ... 0.95. *"Did you find
  it and draw the box precisely?"*

Report both. Quoting only mAP50 is the standard way to make a detector look
better than it is.

## Three traps specific to this dataset

**The class imbalance is 12:1 and mAP does not care.** `car` has 36,209
training boxes, `bicycle` has 3,087, and mAP weights them equally. So a
headline that moved 2% between two ablation arms can be one rare class getting
slightly luckier, with nothing to do with the variable being tested. Every
comparison table this repo writes puts per-class AP50 underneath the mean for
exactly that reason, and reading the mean alone across arms is how a
preprocessing difference gets mistaken for a real effect.

**There is no held-out test split.** This is the significant limitation and it
is worth stating plainly rather than burying. FLIR 1.3 ships a third split
(`video/`, 4,224 frames from a continuous recording) but its annotation JSON is
missing from the download this repo was built against -- see
[doc 00](00-the-dataset.md). So:

- every number here is **validation**, and
- `best.pt` is **selected** on validation.

Numbers produced that way are optimistic by construction, and the amount by
which they are optimistic is unmeasured. That does not invalidate the
*comparisons* -- all arms are selected the same way, so the bias applies
roughly uniformly -- but it does mean no single number in this repo should be
quoted as this model's accuracy. Recovering `video/thermal_annotations.json`
would fix it; the converter already handles that split when it is present.

**Train and validation come from the same collection.** FLIR split the sampled
frames by identifier -- 1-8,862 to train, 8,863-10,228 to validation -- so the
two are not randomly interleaved, which is good. They are still frames from the
same vehicle, the same camera and the same campaign. Validation here measures
generalisation to new frames, not to a new deployment.

## Precision will look bad, and mostly should

844 of the 7,543 training frames contain no annotated object; validation has 5
out of 1,091. A detector trained with 11% background and validated with 0.5% of
it sees far more empty scenes in training than the metric ever rewards it for
handling. Low precision on this validation split is partly that asymmetry
rather than a purely model-side failure.

## What a good result looks like here

Published FLIR numbers are close to uncomparable across papers, because the
class set is not fixed: 1.3 annotates four classes, v2 annotates fifteen, and
papers variously report three, four, or all of them, sometimes after merging
`person` and `bicycle` differently. A number lifted from one paper and set
beside a number from another is usually measuring two different problems.

So the defensible claim for this repo is not a leaderboard position. It is:
*here is a controlled ablation, here is the mechanism predicted in advance,
and here is whether the data agreed.*

## The comparison this repo produces

```bash
uv run thermaldet eval runs/train/*/weights/best.pt
```

Writes both a JSON record and a markdown table to `reports/`. Each checkpoint
is evaluated against **the arm it was trained on**, read back from the
`thermaldet_config.json` written next to its weights. Evaluating a
globally-windowed checkpoint against the AGC arm is a legitimate domain-shift
experiment, but it is not what `eval` is being asked for, and doing it by
accident would put an unexplained row in the table.
