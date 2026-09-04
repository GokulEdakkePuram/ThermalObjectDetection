# Thermal against visible, on the same drives

v2 annotates both spectra with the same label map, which makes the obvious
question askable: **what does a thermal sensor actually buy you?**

The `rgb` arm differs from the control in one line — `data:`. Same
architecture, same COCO initialisation, same schedule, same augmentation.

## What this measures, and what it does not

It is tempting to read the result as "thermal carries more information than
visible". It does not measure that, and it cannot: the two cameras are
different instruments pointed at the same road.

What it measures is narrower and more useful: **given one 640 px input budget,
which sensor detects better?** A deployed detector has a fixed input size, so
that is the question that decides what to put on the vehicle. It is also the
question FLIR's own published baseline answers, which makes the numbers
comparable.

## The resize, which sounds like a confound and mostly is not

Thermal frames are 640×512. Visible frames come in five resolutions up to
2048×1536. Training both at `imgsz: 640` downscales the visible frames a lot
and the thermal ones not at all, which looks like it hands the comparison to
thermal before it starts.

Measured, it does not. The two cameras have similar fields of view — 45° for
the Tau 2, 52.8° for the BlackFly — so after each frame is scaled to a 640 px
long side, an object of a given real-world size lands at nearly the same
pixel height in both:

| class | thermal, native | thermal @640 | visible, native | visible @640 |
| --- | ---: | ---: | ---: | ---: |
| person | 26 px | **26** | 71 px | **30** |
| bike | 30 px | **30** | 71 px | **28** |
| car | 22 px | **22** | 51 px | **21** |
| light | 16 px | **16** | 31 px | **12** |
| sign | 11 px | **11** | 22 px | **9** |

Median box height in the training split, by class.

The residual gap runs **against** visible on the two small classes: a sign is
9 px on the visible arm against 11 on thermal, a light 12 against 16. That is
the wider field of view spending pixels on more scene.

This matters for how a result is read, and the direction is convenient. If
visible *wins* on `sign` and `light`, it wins despite ~25% less resolution on
them, and the modality effect is larger than measured. If visible *loses* on
them, part of that loss is resolution rather than sensing, and the result is
confounded. So one outcome is clean and the other needs a caveat — which is
worth knowing in advance rather than deciding afterwards.

## The augmentation handicap, taken deliberately

[`base.yaml`](../configs/base.yaml) sets `hsv_h: 0.0` and `hsv_s: 0.0`. On a
single-channel thermal sensor replicated across three channels, hue and
saturation jitter do nothing — there is only a grey axis to shift.

On the visible arm they would do a great deal, and they stay off anyway.

This is a real handicap to RGB and it is the price of the comparison being
single-variable. An arm that changed both the sensor *and* the augmentation
recipe would not isolate either. The honest way to report it is: the visible
arm is trained without the colour augmentation it would normally get, so its
number is a floor rather than its best achievable. A follow-up `rgb_hsv` arm
would measure how much that costs, and is the first thing to run if the
modality result is close.

## The classes chosen to make this a question

Restricted to `person` and `car`, this experiment is a formality — thermal
detects warm bodies at night and that is the entire point of the sensor.

`light` and `sign` are in the default class set to make it a real question.
A traffic light signals with **colour**; a street sign with **printed
contrast**. Neither survives an infrared sensor, which sees only emitted heat
— a red light and a green light are the same temperature. So the prediction
splits:

| class | expected winner | why |
| --- | --- | --- |
| person | thermal, clearly | body heat against ambient, day or night |
| car | thermal | engine and exhaust are strong emitters |
| bike | unclear | mostly metal, warm only when ridden |
| light | **visible, clearly** | the signal *is* the colour |
| sign | **visible** | printed contrast, no thermal signature |

**If thermal wins on all five, something is wrong with the comparison** — most
likely the resize, or the fact that the thermal arm has 4% more training
frames. A per-class split that matches the table above is the outcome that
would make the aggregate number trustworthy.

## The paired video split

For the image splits the two spectra are separate frame sets, so the
comparison is detector-level: train each on its own data, score each on its
own test split.

The test split is better than that. `rgb_to_thermal_vid_map.json` gives 3,749
time-synced RGB↔thermal frame pairs, and all 3,749 resolve on both sides. Both
detectors can be scored on the *same moments*, which removes scene difficulty
as a variable entirely.

That pairing is not used by `thermaldet eval` yet — the test splits are scored
independently, which is the weaker comparison. Using the map is the obvious
next improvement and needs no new data.

## A published baseline to sit beside

FLIR trained YOLOX-m, COCO-pretrained, at 640×640 on both spectra and reported
AP@IoU=0.5 on this test split:

| | person | car |
| --- | ---: | ---: |
| visible | 51.42 | 55.79 |
| thermal | **75.33** | **77.23** |

Worth having for two reasons. It is an external check that the pipeline here
is not producing nonsense — a number wildly off these, in either direction,
means something is wrong before it means something is interesting. And it sets
the expected size of the modality effect on `person` and `car`: roughly +24 and
+21 AP50 for thermal.

Note the protocol is not identical — YOLOX-m against YOLO11s, and FLIR do not
state their class set or augmentation. It is a reference point, not a
leaderboard row.
