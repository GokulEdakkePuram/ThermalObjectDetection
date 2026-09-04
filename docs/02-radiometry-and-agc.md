# The 8-bit frames are not what the sensor saw

Every FLIR ADAS result in the literature is trained on the frames in `data/`.
Those files are not the camera's output. They are the output of **automatic
gain control**: each frame's own intensity range is stretched to fill 0-255
before it is written.

That is a per-frame, non-stationary transform, and it destroys the one thing a
thermal sensor uniquely offers. A thermal camera measures an absolute quantity.
After AGC, a 310 K person maps to whatever grey level the rest of *that
particular frame* happened to leave available. The same person, in the same
pose, is a different pixel value in the next frame if a hot exhaust enters the
scene.

v2 ships the raw 16-bit frames alongside, under `analyticsData/`, which makes
this testable rather than arguable.

## What the raw data looks like

640x512, uint16, raw sensor counts. A typical frame occupies a narrow band:

| quantity | counts |
| --- | ---: |
| median frame span (1st-99th percentile) | 764 |
| span across frames (nothing clipped) | 3,097 |
| the global window this repo uses (pooled p0.5-p99.5) | 2,219 |

Measured over 400 sampled training frames by `thermaldet profile`.

## The trade, before any GPU time

A single fixed window has to be wide enough for every frame in the dataset. A
median frame spans 764 counts inside a 2,219-count window, so it reaches:

```
255 * 764 / 2219  =  88 of 255 output levels
```

**Absolute radiometry costs 2.9x of per-frame contrast.** That number is
available for the price of reading 400 TIFFs, and it is what makes this a real
trade rather than an obvious win for either side.

## The three arms

Identical in every respect except how 16 bits become 8:

**`agc`** — FLIR's shipped JPEGs. The control, and what everyone else uses.

**`global`** — one fixed linear window for the whole dataset, taken from pooled
0.5th/99.5th percentiles of the training split. Not min/max: a single
sun-facing radiator at 15,000 counts would otherwise set the top of the window
for every frame. Absolute counts survive, so the same temperature is the same
grey level everywhere.

**`p1p99`** — per-frame 1st-99th percentile stretch. AGC re-implemented in
eight lines, with published parameters instead of a proprietary curve.

The third arm is what makes the comparison decidable. Without it, `global`
losing to `agc` has two explanations — *per-frame normalisation helps*, or
*FLIR's particular normalisation helps* — and no way to choose between them.

## Predictions

1. **`global` loses to `agc`.** 2.4x of contrast is a lot to pay, and a
   detector has no obvious mechanism for exploiting absolute temperature that
   it could not get from relative contrast within the frame.

2. **`p1p99` lands close to `agc`.** If it does, what AGC contributes is the
   normalisation, and FLIR's curve is not doing anything a linear stretch
   cannot.

3. If instead **`global` wins**, absolute radiometry is carrying real signal
   and the entire 8-bit convention is discarding it -- which would be the more
   interesting result, and would make the 16-bit files worth the disk they
   take.

## The frames are the same frames

The arms are built from different image *files*, so the first thing to check is
that they are built from the same *frames*. On v2 they are: the `data/` and
`analyticsData/` directories match one for one in every split, and all three
thermal arms come out at 10,742 / 1,144 / 3,749 frames with 168,238 / 16,244 /
55,371 boxes.

The converter still intersects the frame list across every image source rather
than trusting that, because the check costs a directory listing and the failure
it prevents is silent -- two arms meant to differ only in preprocessing
differing also in what they trained on. There is a
[regression test](../tests/test_convert.py) holding it.

## Caveats

- **The mapping is fixed before training and never revisited.** A learned or
  adaptive normalisation is a different and probably better idea; it is also a
  different experiment, and it would stop this one being single-variable.
- **JPEG vs PNG.** The `agc` arm is JPEG because that is what FLIR ships; the
  rendered arms are PNG. Re-encoding the rendered arms as JPEG to match would
  add a lossy codec on top of the variable being tested. The asymmetry is
  real, and it slightly favours the rendered arms.
- **`global`'s window is measured on the training split only**, which is
  correct -- measuring it on validation too would leak.
