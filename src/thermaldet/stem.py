"""How much of an RGB-pretrained first layer survives a single-channel input.

A thermal frame has one channel. Ultralytics replicates it across three to fit
a network built for RGB, which means every first-layer filter sees the *same*
image three times. A filter's response to that input is governed by the sum of
its three channel slices::

    y = sum_c W[:, c] * x        (because x is identical in every channel)

For an achromatic filter -- one whose three slices are near-identical, an edge
detector that ignores colour -- the sum is three times one slice and nothing is
lost. For a colour-opponent filter -- red-minus-green, the kind that fires on a
brake light against foliage -- the slices have opposite signs and the sum very
nearly cancels. Those filters output approximately zero on every thermal frame
ever shown to them, whatever their weights say.

So this measures, per filter::

    ratio = ||sum_c W[:, c]||_F / sum_c ||W[:, c]||_F

near 1 when the channel slices reinforce, near 0 when they cancel.

On COCO-pretrained YOLO11s, **10 of 32 stem filters fall below 0.1**: roughly a
third of the first layer is dead on arrival on thermal input, before a single
gradient step. The median filter sits at 0.93, so the layer is not uniformly
damaged -- it splits cleanly into a majority that transfers intact and a
chromatic minority that cannot.

That is the concrete form of the claim the transfer ablation tests. It is also
why `frozen_stem` is a separate arm from `frozen_backbone`: if a third of the
stem has to be rebuilt, freezing it should cost far more here than the same
freeze costs on an RGB fine-tune.
"""

from __future__ import annotations

from dataclasses import dataclass

DEAD_THRESHOLD = 0.1


@dataclass(frozen=True)
class StemAnalysis:
    """Per-filter grey-input response for one first convolution."""

    n_filters: int
    ratios: list[float]
    threshold: float = DEAD_THRESHOLD

    @property
    def n_dead(self) -> int:
        return sum(1 for r in self.ratios if r < self.threshold)

    @property
    def dead_fraction(self) -> float:
        return self.n_dead / self.n_filters if self.n_filters else 0.0

    @property
    def median_ratio(self) -> float:
        ordered = sorted(self.ratios)
        return ordered[len(ordered) // 2] if ordered else 0.0

    def summary(self) -> str:
        return (
            f"stem: {self.n_filters} filters\n"
            f"  grey-input response below {self.threshold}: "
            f"{self.n_dead} filters ({100 * self.dead_fraction:.0f}%)\n"
            f"  median response ratio: {self.median_ratio:.2f} "
            f"(1.0 = channel slices reinforce, 0.0 = they cancel)"
        )


def stem_weight(model) -> object:
    """The first convolution's weight tensor, whatever the model wraps it in."""
    for name, param in model.named_parameters():
        if param.ndim == 4 and param.shape[1] == 3:
            return name, param.detach()
    raise SystemExit("No 3-channel convolution found; is this an RGB-input model?")


def analyse(weights: str = "yolo11s.pt", threshold: float = DEAD_THRESHOLD) -> StemAnalysis:
    """Measure the grey-input response of every filter in a checkpoint's stem."""
    from ultralytics import YOLO

    _, w = stem_weight(YOLO(weights).model)

    # Response to a replicated single channel is governed by the sum over the
    # channel dimension; normalise by the energy that was there to begin with,
    # so the ratio is about cancellation rather than about filter magnitude.
    combined = w.sum(dim=1).flatten(1).norm(dim=1)
    separate = w.flatten(2).norm(dim=2).sum(dim=1)
    ratios = (combined / separate.clamp(min=1e-12)).tolist()

    return StemAnalysis(n_filters=len(ratios), ratios=ratios, threshold=threshold)
