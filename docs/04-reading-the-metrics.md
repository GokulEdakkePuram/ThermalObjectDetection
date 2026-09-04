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

**The class imbalance is 10:1 and mAP does not care.** `car` has 73,622
training boxes, `bike` has 7,237, and mAP weights them equally. So a headline
that moved 2% between two ablation arms can be one small class getting
slightly luckier, with nothing to do with the variable being tested. Every
comparison table this repo writes puts per-class AP50 underneath the mean for
exactly that reason, and reading the mean alone across arms is how a
preprocessing difference gets mistaken for a real effect.

That is doubly true here because the per-class split is not decoration -- it
is the [modality prediction](03-thermal-vs-visible.md). Thermal is expected to
win on `person` and `car` and lose on `light` and `sign`, and the aggregate
mAP could come out flat while both halves of that are true.

**`bike` is the class to distrust.** It has 7,237 training boxes but only 170
in validation and **113 in the held-out test split**, out of 55,371. Its AP is
decided by a hundred-odd boxes and carries a fifth of the headline mAP, so it
will move between arms for reasons that have nothing to do with the arms --
exactly the failure that got `dog` excluded, one order of magnitude up.

It stays in the class set because it has enough *training* data to be learned
and because dropping a class after seeing its numbers is how a result gets
tuned into existence. The right response is to read it as noise and say so:
quote mAP over the five, and note that a `bike` swing is not evidence. If two
arms differ only through `bike`, they do not differ.

Per-class counts in the thermal splits:

| class | train | val | test |
| --- | ---: | ---: | ---: |
| car | 73,622 | 7,133 | 30,517 |
| person | 50,474 | 4,470 | 12,323 |
| sign | 20,747 | 2,471 | 5,660 |
| light | 16,158 | 2,000 | 6,758 |
| bike | 7,237 | **170** | **113** |

**Report the test split, not validation.** v2 ships 3,749 annotated frames in
`video_thermal_test`, sampled -- per FLIR -- from completely independent video
sequences. `thermaldet eval` defaults to `--split test` for that reason: `best.pt`
is *selected* on validation, so a validation number is optimistic by
construction and there is no reason to quote one when a held-out split exists.

Validation is still worth watching during a run. It is just not the number that
goes in a table.

**Expect the test number to be well below validation, and not only from
selection bias.** The test frames are 87.3% small by COCO's definition against
74.6% in training, because they were sampled at 30 fps from continuous footage
rather than curated for detector training. Part of the drop is the split being
genuinely harder. Part of it is `best.pt` having been chosen on val. Those two
are not separable from a single pair of numbers, so report both splits and
resist explaining the gap.

**Two cameras hide inside the test split.** Six of the eight test videos are a
Tau 2 at 45° HFOV; two are a Boson ADK at 50°. A per-video breakdown will show
that as a difference the model did not cause.

**The scenes are dense.** 15.7 annotated objects per training frame, against
roughly 7 on COCO. Ultralytics validates at `conf=0.001` and caps predictions
at `max_det=300`; ground truth never approaches that here, but at a
near-zero confidence threshold a detector emits far more candidates than there
are objects, and the discarded tail is exactly what the recall sweep needs.
Worth measuring rather than assuming in either direction -- re-run `eval` with
`max_det` raised and see whether mAP moves.

## What a good result looks like here

Published FLIR numbers are close to uncomparable across papers, because the
class set is not fixed: v2 annotates sixteen classes and papers variously
report three, five, or all of them. A number lifted from one paper and set
beside a number from another is usually measuring two different problems.

There is one exception worth using. FLIR published a baseline on *this* test
split -- YOLOX-m, COCO-pretrained, 640×640 -- at AP@IoU=0.5 of 75.33 (person)
and 77.23 (car) on thermal, 51.42 and 55.79 on visible. Different architecture
and an unstated class set, so not a leaderboard row, but a useful sanity check:
a number far from those in either direction means something is wrong before it
means something is interesting.

Beyond that, the defensible claim for this repo is not a leaderboard position.
It is: *here is a controlled ablation, here is the mechanism predicted in
advance, and here is whether the data agreed.*

## The comparison this repo produces

```bash
uv run thermaldet eval runs/train/*/weights/best.pt --split test
```

Writes both a JSON record and a markdown table to `reports/`. Each checkpoint
is evaluated against **the arm it was trained on**, read back from the
`thermaldet_config.json` written next to its weights. Evaluating a
globally-windowed checkpoint against the AGC arm is a legitimate domain-shift
experiment, but it is not what `eval` is being asked for, and doing it by
accident would put an unexplained row in the table.
