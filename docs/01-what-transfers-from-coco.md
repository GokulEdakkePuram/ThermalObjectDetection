# What transfers from COCO, and what does not

Nearly every thermal detection paper starts by fine-tuning COCO-pretrained
weights, and nearly none of them ask what that pretraining is actually
contributing. The assumption is imported wholesale from RGB transfer learning,
where it is well earned: pretrain on ImageNet or COCO, fine-tune on your
smaller in-domain set, and you get most of the accuracy for a fraction of the
data.

Thermal is a different kind of shift. It is not a new set of scenes drawn from
the same sensor -- it is a **different sensor**, and the difference shows up in
the very first layer.

## A third of the stem is dead on arrival

A thermal frame is single-channel. Ultralytics replicates it across three
channels to fit a network built for RGB, so every first-layer filter sees the
*same* image three times. That makes its response depend entirely on the sum of
its channel slices:

```
y = sum_c W[:, c] * x        because x is identical in every channel
```

An **achromatic** filter -- an edge detector that ignores colour -- has three
near-identical slices, and the sum is three times one of them. Nothing is lost.
A **colour-opponent** filter -- red-minus-green, the kind that separates a
brake light from foliage -- has slices of opposite sign, and the sum cancels.
It outputs approximately zero on every thermal frame it will ever be shown,
regardless of what its weights say.

That is measurable directly, with no training at all:

```bash
uv run thermaldet stem-check yolo11s.pt
```

```
stem: 32 filters
  grey-input response below 0.1: 10 filters (31%)
  median response ratio: 0.93 (1.0 = channel slices reinforce, 0.0 = they cancel)
```

**Ten of thirty-two filters, ~31% of the first layer, are inert on thermal
input before a single gradient step.** The median filter sits at 0.93, which
matters just as much: the stem does not degrade gracefully across the board, it
**splits** into a majority that transfers intact and a chromatic minority that
cannot transfer at all.

Deeper layers are less exposed, because they consume feature maps rather than
pixels. Whatever survives layer 0 is a valid input to layer 1. So the
prediction is not "COCO pretraining is useless on thermal" -- it is that the
damage is concentrated at the very front, and that the front is exactly the
part conventional fine-tuning advice tells you to freeze.

## What the ablation tests

Four runs. Everything -- schedule, data, augmentation, seed, batch -- is
identical except the one line named:

| run | change | what it isolates |
| --- | --- | --- |
| `pretrained` | — | the control |
| `scratch` | `model: yolo11s.yaml` | what COCO pretraining is worth at all |
| `frozen_stem` | `freeze: 2` | whether layers 0-1 must be relearned |
| `frozen_backbone` | `freeze: 11` | whether anything below the head must be |

`freeze: 11` is the whole backbone on YOLO11, whose layers run 0-10. The
widely copied `freeze: 10` is a YOLOv8 number; carried over unchanged it leaves
the last backbone block training while the config claims otherwise.

## Predictions, written before the runs

Recorded here so that being wrong stays visible rather than getting quietly
rewritten afterwards.

1. **`pretrained` beats `scratch`, but by less than the RGB rule of thumb.**
   Shape, texture and part-structure priors are modality-independent and
   should survive. Colour is a third of the stem, and gone.

2. **`frozen_stem` costs disproportionately more than freezing two layers
   costs on an RGB fine-tune.** This is the sharp one. Freezing two of
   twenty-three layers is normally almost free. If a third of layer 0 is inert
   and cannot be repaired, it should not be.

3. **`frozen_backbone` is much worse than `frozen_stem`**, which is expected
   anywhere and is not evidence for anything thermal-specific on its own.

4. **The interesting possibility: `scratch` beats `frozen_backbone`.** That
   would say COCO features, held fixed, are worse than no COCO features at all
   -- a strong statement about how far out of domain thermal is, and one that
   would make the standard freeze-the-backbone recipe actively harmful here.

## What would falsify it

If **`frozen_stem` lands within noise of `pretrained`**, the argument above is
wrong. It would mean the 31% of inert filters were carrying nothing the network
needed, that the surviving achromatic majority is sufficient, and that a
thermal fine-tune is an ordinary fine-tune after all. That is a perfectly
plausible outcome and it is the one this arm exists to detect.

The stem measurement is *not* itself evidence that those filters mattered. It
establishes that they are inert; whether the network misses them is what the
GPU time buys.

## The follow-up, if the prediction holds

If the stem has to be relearned rather than adjusted, it wants a **higher**
learning rate than the layers above it -- which is the opposite of what
discriminative fine-tuning conventionally does, where early layers get the
*smallest* LR because they are assumed most general.

Ultralytics' Trainer offers no hook to scale learning rate by depth, which is
what [`manual.py`](../src/thermaldet/manual.py) is for:

```bash
uv run thermaldet train-manual --backbone-lr-mult 5.0 --ema
```

`--backbone-lr-mult 1.0` reproduces the stock three-group layout exactly, so
any difference is attributable to the multiplier rather than to the loop.
