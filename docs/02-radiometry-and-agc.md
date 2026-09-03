# The 8-bit frames are not what the sensor saw

Every FLIR ADAS result in the literature is trained on `thermal_8_bit/*.jpeg`.
Those files are not the camera's output. They are the output of **automatic
gain control**: each frame's own intensity range is stretched to fill 0-255
before it is written.

That is a per-frame, non-stationary transform, and it destroys the one thing a
thermal sensor uniquely offers. A thermal camera measures an absolute quantity.
After AGC, a 310 K person maps to whatever grey level the rest of *that
particular frame* happened to leave available. The same person, in the same
pose, is a different pixel value in the next frame if a hot exhaust enters the
scene.

The 1.3 release ships the raw 16-bit TIFFs alongside, which makes this
testable rather than arguable.

## What the raw data looks like

640x512, uint16, raw sensor counts. A typical frame occupies a narrow band:

| quantity | counts |
| --- | ---: |
| median frame span (1st-99th percentile) | 873 |
| span across frames (nothing clipped) | 2,987 |
| the global window this repo uses (pooled p0.5-p99.5) | 2,096 |

Measured over 400 sampled training frames by `thermaldet profile`.

## The trade, before any GPU time

A single fixed window has to be wide enough for every frame in the dataset. A
median frame spans 873 counts inside a 2,096-count window, so it reaches:

```
255 * 873 / 2096  =  106 of 255 output levels
```

**Absolute radiometry costs 2.4x of per-frame contrast.** That number is
available for the price of reading 400 TIFFs, and it is what makes this a real
trade rather than an obvious win for either side.

Checked on a real frame (`FLIR_00001`), the prediction holds:

| arm | std. dev. | p1 | p99 |
| --- | ---: | ---: | ---: |
| `agc` (FLIR's JPEG) | 63.7 | 9 | 249 |
| `global` | 37.2 | 11 | 184 |
| `p1p99` | 54.4 | 0 | 255 |

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

## The confound that had to be removed first

The arms are built from *different image files*, so the first thing to check is
that they are built from the same *frames*.

They were not. The download is missing ~15% of the 8-bit JPEGs while the
16-bit TIFFs are complete, so taking the intersection per-arm -- over only the
sources each arm needed -- produced 7,562 training frames for `agc` against
7,543 for `p1p99`.

Nineteen frames out of seven and a half thousand will not move mAP much. They
will move it a little, in an unknown direction, and that is worse: a 1% gap
between two arms would have had no attributable cause. So the frame list is
now intersected across every source the release ships, whichever arm is being
built, with a [regression test](../tests/test_convert.py) holding it.

Chasing the 19 down found something worse than an inconsistency. All of them
were `"FLIR_01437 2.jpeg"`-style Finder copies, made when the dataset was
moved between machines, which happened to exist beside the JPEGs and not
beside the TIFFs. Left in, a duplicate frame trains **twice** -- and one that
survives in only some directories trains twice in only some arms. They are now
dropped by pattern at source; nothing legitimate in FLIR ends in a space and a
number.

Both fixes are kept. The intersection is the half that works without having
noticed the cause, and there will be another cause.

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
