"""Mapping raw 16-bit thermal counts onto the 8 bits a detector expects.

The 8-bit JPEGs everyone trains on are not what the sensor produced. FLIR's
camera applies automatic gain control before writing them: each frame's own
intensity range is stretched to fill 0-255. That is a *per-frame,
non-stationary* transform, and it throws away the one thing a thermal sensor
uniquely offers -- an absolute measurement. After AGC, a 310 K person maps to
whatever grey level the rest of that particular frame left available.

FLIR ships the raw 16-bit TIFFs alongside, under ``analyticsData/``, which
makes the alternative testable. Three mappings, identical in every other respect:

``agc``
    FLIR's shipped 8-bit JPEGs. The control, and what every FLIR paper uses.

``global``
    One fixed linear map for the entire dataset, from pooled percentiles of
    the training split. Absolute counts survive: the same temperature is the
    same grey level in every frame.

``p1p99``
    Per-frame 1st-99th percentile stretch. AGC re-implemented in eight lines,
    so that "per-frame normalisation" can be separated from "FLIR's particular
    per-frame normalisation".

What this costs is measurable before any GPU time is spent, and
:func:`frame_spans` measures it. Over 400 sampled v2 training frames: a median
frame's 1st-99th percentile span is **764 counts**, while the span across
frames is **3,097**. So a single fixed window wide enough for the dataset
leaves a median frame using roughly **88 of the 255 output levels** -- a 2.9x
contrast loss, paid on every frame, in exchange for absolute radiometry.

The prediction, written before running any of it: ``global`` loses to ``agc``,
and ``p1p99`` lands close to ``agc``. That combination would say the useful
content of AGC is the per-frame normalisation rather than FLIR's particular
curve, and that absolute temperature is not worth 2.9x of contrast to a
detector. If instead ``global`` wins, absolute radiometry is carrying real
signal and the entire 8-bit convention is leaving it on the table.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Pooled percentiles for the global map. Not min/max: a single sun-facing
# radiator at 15,000 counts would otherwise set the top of the window for
# every frame in the dataset.
GLOBAL_LO_PCT = 0.5
GLOBAL_HI_PCT = 99.5

# Frames sampled when measuring the global window. The pooled distribution is
# stable well before this, and reading 10k TIFFs to place two numbers is not a
# good use of four minutes.
SAMPLE_FRAMES = 400


def read_raw(path: Path) -> np.ndarray:
    """Read one 16-bit radiometric TIFF as a uint16 array."""
    from PIL import Image

    with Image.open(path) as img:
        return np.array(img)


def _to_uint8(counts: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linearly map ``[lo, hi]`` counts onto 0-255, clipping outside."""
    if hi <= lo:
        return np.zeros(counts.shape, dtype=np.uint8)
    scaled = (counts.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class GlobalLinear:
    """One fixed window for the whole dataset. Absolute counts are preserved."""

    lo: float
    hi: float

    def __call__(self, src: Path, dst: Path) -> None:
        _save(_to_uint8(read_raw(src), self.lo, self.hi), dst)


@dataclass(frozen=True)
class PercentileStretch:
    """Per-frame window from that frame's own percentiles. AGC, reproducibly."""

    lo_pct: float = 1.0
    hi_pct: float = 99.0

    def __call__(self, src: Path, dst: Path) -> None:
        counts = read_raw(src)
        lo, hi = np.percentile(counts, [self.lo_pct, self.hi_pct])
        _save(_to_uint8(counts, float(lo), float(hi)), dst)


def _save(image: np.ndarray, dst: Path) -> None:
    from PIL import Image

    # PNG, not JPEG: the arms already differ in one deliberate way, and adding
    # a lossy codec to two of the three would be a second difference.
    Image.fromarray(image, mode="L").save(dst, optimize=False, compress_level=1)


MAPPINGS = {"global": GlobalLinear, "p1p99": PercentileStretch}


def measure_window(
    raw_dir: Path,
    sample: int = SAMPLE_FRAMES,
    lo_pct: float = GLOBAL_LO_PCT,
    hi_pct: float = GLOBAL_HI_PCT,
    seed: int = 0,
) -> tuple[float, float]:
    """Pool pixels from a sample of frames and return the global window.

    Sampling is seeded so that rebuilding the dataset reproduces the same
    window -- an arm whose preprocessing shifts between builds is not an arm.
    """
    paths = sorted(raw_dir.glob("*.tiff"))
    if not paths:
        raise SystemExit(f"No 16-bit TIFFs under {raw_dir}.")

    chosen = random.Random(seed).sample(paths, min(sample, len(paths)))
    pooled = np.concatenate([read_raw(p).ravel() for p in chosen])
    lo, hi = np.percentile(pooled, [lo_pct, hi_pct])
    return float(lo), float(hi)


def frame_spans(raw_dir: Path, sample: int = SAMPLE_FRAMES, seed: int = 0) -> dict[str, float]:
    """How much dynamic range a single frame uses, against the whole dataset.

    This is the measurement that predicts the ablation's outcome, so it is
    worth having before spending the GPU hours rather than after.
    """
    paths = sorted(raw_dir.glob("*.tiff"))
    chosen = random.Random(seed).sample(paths, min(sample, len(paths)))

    spans, los, his = [], [], []
    for path in chosen:
        counts = read_raw(path)
        lo, hi = np.percentile(counts, [1, 99])
        spans.append(hi - lo)
        los.append(lo)
        his.append(hi)

    median_span = float(np.median(spans))
    window_lo, window_hi = measure_window(raw_dir, sample=sample, seed=seed)
    window_span = window_hi - window_lo

    return {
        "frames_sampled": len(chosen),
        "median_frame_span": median_span,
        # The naive window -- every frame covered, nothing clipped. Reported
        # because it is what "one fixed map" sounds like it should mean.
        "dataset_span": float(np.max(his) - np.min(los)),
        "window_lo": window_lo,
        "window_hi": window_hi,
        "window_span": window_span,
        # What the global arm actually costs: the output levels a median frame
        # gets to use, measured against the window that arm really applies.
        "levels_under_global_map": 255.0 * median_span / max(window_span, 1.0),
    }
