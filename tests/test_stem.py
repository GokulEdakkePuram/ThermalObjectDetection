"""The grey-input response measurement behind the transfer thesis."""

from __future__ import annotations

import pytest
import torch

from thermaldet.stem import StemAnalysis, stem_weight


class _Model:
    """Minimal stand-in: `analyse` only ever asks for named_parameters."""

    def __init__(self, weight):
        self._weight = weight

    def named_parameters(self):
        yield "model.0.conv.bias", torch.zeros(len(self._weight))
        yield "model.0.conv.weight", self._weight


def _ratios(weight):
    combined = weight.sum(dim=1).flatten(1).norm(dim=1)
    separate = weight.flatten(2).norm(dim=2).sum(dim=1)
    return (combined / separate.clamp(min=1e-12)).tolist()


def test_an_achromatic_filter_survives_replication():
    """Three identical slices reinforce: nothing is lost on a grey input."""
    slice_ = torch.randn(1, 1, 3, 3)
    weight = slice_.repeat(1, 3, 1, 1)

    assert _ratios(weight)[0] == pytest.approx(1.0, abs=1e-5)


def test_a_colour_opponent_filter_cancels_completely():
    """Red-minus-green fires on a brake light and outputs zero on thermal."""
    slice_ = torch.randn(1, 1, 3, 3)
    weight = torch.cat([slice_, -slice_, torch.zeros_like(slice_)], dim=1)

    assert _ratios(weight)[0] == pytest.approx(0.0, abs=1e-6)


def test_stem_weight_finds_the_first_three_channel_convolution():
    weight = torch.randn(8, 3, 3, 3)
    name, found = stem_weight(_Model(weight))

    assert name.endswith("conv.weight")
    assert torch.equal(found, weight)


def test_a_model_with_no_rgb_input_fails_loudly():
    class Mono:
        def named_parameters(self):
            yield "model.0.conv.weight", torch.randn(8, 1, 3, 3)

    with pytest.raises(SystemExit, match="RGB-input"):
        stem_weight(Mono())


class TestAnalysis:
    def test_counts_filters_below_the_threshold(self):
        analysis = StemAnalysis(n_filters=4, ratios=[0.02, 0.05, 0.9, 0.95])

        assert analysis.n_dead == 2
        assert analysis.dead_fraction == 0.5

    def test_median_is_reported_alongside_the_count(self):
        """The count alone would suggest uniform damage. On real weights the
        median sits near 0.93, so the layer splits rather than degrades."""
        analysis = StemAnalysis(n_filters=5, ratios=[0.02, 0.05, 0.93, 0.95, 0.97])

        assert analysis.median_ratio == pytest.approx(0.93)
        assert analysis.dead_fraction == pytest.approx(0.4)

    def test_an_empty_stem_does_not_divide_by_zero(self):
        assert StemAnalysis(n_filters=0, ratios=[]).dead_fraction == 0.0
