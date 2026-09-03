"""Short calibration run before a long one.

Renting a GPU makes a wrong batch size expensive: an OOM, or an estimate that
turns out to be five times too optimistic, is much cheaper to find in three
minutes than an hour in.

The naive way to do this -- time one short run and divide by the fraction of
data it used -- is wrong, and wrong in a way that looks plausible. Epoch time
is not proportional to dataset size::

    epoch_seconds = overhead + rate * n_images

``overhead`` is dataloader spin-up, cuDNN autotuning and epoch teardown, and
it does not shrink with the data. Dividing a small run's time by ``fraction``
scales that overhead up along with everything else, so at ``fraction=0.1`` the
estimate carries **ten times** the real overhead. That matters more here than
on a larger dataset: 7.5k frames at 640x512 make for short epochs, so fixed
overhead is a large share of each one.

So this measures at two fractions and solves for both terms. The extra run
costs a couple of minutes and is the difference between an estimate and a
guess.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config

DEFAULT_EPOCHS = 3
DEFAULT_FRACTIONS = (0.1, 0.3)


@dataclass
class ProbeResult:
    """What a calibration run measured, and what it implies for the real one."""

    name: str
    profile: str
    imgsz: int
    batch: int
    train_images: int
    overhead_seconds: float
    seconds_per_image: float
    target_epochs: int
    peak_memory_gb: float | None

    @property
    def full_epoch_seconds(self) -> float:
        return self.overhead_seconds + self.seconds_per_image * self.train_images

    @property
    def estimated_hours(self) -> float:
        return self.full_epoch_seconds * self.target_epochs / 3600

    def summary(self) -> str:
        mem = f"{self.peak_memory_gb:.1f} GB peak" if self.peak_memory_gb else "peak memory n/a"
        return (
            f"{self.name} [{self.profile or 'no profile'}] imgsz={self.imgsz} batch={self.batch}\n"
            f"  fitted   : {self.overhead_seconds:.1f}s overhead "
            f"+ {self.seconds_per_image * 1000:.1f}ms/image\n"
            f"  projected: {self.full_epoch_seconds / 60:.1f} min/epoch "
            f"on {self.train_images:,} images\n"
            f"  {self.target_epochs} epochs -> {self.estimated_hours:.1f} h   ({mem})"
        )


def _steady_epoch_seconds(results_csv: Path) -> float:
    """Fastest per-epoch delta, as the best available proxy for steady state.

    The first epoch always carries label-cache building and warmup, so a mean
    would be biased upward by costs the real run pays only once.
    """
    rows = list(csv.DictReader(results_csv.open()))
    if not rows:
        raise RuntimeError(f"{results_csv} has no rows; the probe run produced no epochs.")

    times = [float(r["time"]) for r in rows]
    deltas = [b - a for a, b in zip(times, times[1:], strict=False)]
    return min(deltas) if deltas else times[0]


def _count_train_images(data: str) -> int:
    """How many images a full epoch actually iterates over."""
    from ultralytics.data.utils import check_det_dataset

    from .paths import configure_ultralytics

    configure_ultralytics()
    train_dir = Path(check_det_dataset(data)["train"])
    return sum(1 for p in train_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def _fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Solve ``t = overhead + rate * n`` through two or more measurements.

    Returns ``(overhead_seconds, seconds_per_image)``. A non-positive fitted
    rate means the two runs were indistinguishable through the noise; the
    caller is told rather than handed a nonsense extrapolation.
    """
    (n1, t1), (n2, t2) = points[0], points[-1]
    if n2 == n1:
        raise ValueError("Probe fractions must differ to separate overhead from rate.")

    rate = (t2 - t1) / (n2 - n1)
    if rate <= 0:
        raise RuntimeError(
            f"Fitted a non-positive rate ({rate:.6f}s/image) from {t1:.1f}s at {n1} images "
            f"and {t2:.1f}s at {n2}. The runs were too short to measure through noise -- "
            f"re-probe with larger --fractions."
        )
    return t1 - rate * n1, rate


def probe(
    config: str,
    profile: str | None = "auto",
    epochs: int = DEFAULT_EPOCHS,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
) -> ProbeResult:
    """Time short runs at two data fractions and project the full schedule."""
    import torch

    from .train import train

    base = load_config(config, profile=profile)
    target_epochs = base.epochs
    train_images = _count_train_images(base.data)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    points: list[tuple[int, float]] = []
    for fraction in fractions:
        cfg = load_config(config, profile=profile)
        cfg.name = f"probe_{cfg.name}_{fraction:g}"
        cfg.epochs = epochs
        cfg.patience = 0
        cfg.tracker = "none"
        cfg.train_args = {
            **cfg.train_args,
            "fraction": fraction,
            "plots": False,
            # Validation is a fixed per-epoch cost that does not scale with the
            # training fraction, so including it would corrupt the fit.
            "val": False,
        }
        result = train(cfg)
        seconds = _steady_epoch_seconds(Path(result["save_dir"]) / "results.csv")
        points.append((int(train_images * fraction), seconds))

    overhead, rate = _fit(points)
    # Reserved, not allocated. PyTorch's caching allocator holds freed blocks,
    # so reserved exceeds allocated -- and reserved is what runs the card out
    # of memory. Ultralytics' own GPU_mem column reports reserved for the same
    # reason.
    peak = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None

    return ProbeResult(
        name=config,
        profile=base.profile,
        imgsz=base.imgsz,
        batch=base.batch,
        train_images=train_images,
        overhead_seconds=overhead,
        seconds_per_image=rate,
        target_epochs=target_epochs,
        peak_memory_gb=peak,
    )


def probe_all(configs: list[str], **kwargs: Any) -> list[ProbeResult]:
    """Probe several configs and report what the whole sweep would cost."""
    results = [probe(c, **kwargs) for c in configs]

    print("\n" + "=" * 68)
    for r in results:
        print(r.summary())
    print("-" * 68)
    print(f"  full sweep: {sum(r.estimated_hours for r in results):.1f} h")
    print("=" * 68)
    return results
