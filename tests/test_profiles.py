"""Hardware profiles, and the projection maths that decides what a run costs."""

from __future__ import annotations

import pytest

from thermaldet.config import load_config, load_profile
from thermaldet.hardware import CPU, CUDA_LARGE, CUDA_MEDIUM, CUDA_SMALL, Hardware, auto_profile

ARMS = ["pretrained", "scratch", "frozen_stem", "frozen_backbone", "global_map", "p1p99_map"]


def test_profile_overrides_experiment_batch():
    """The machine's limits win over whatever the experiment asked for."""
    laptop = load_config("pretrained", profile="mps")
    rented = load_config("pretrained", profile="cuda24")

    assert laptop.batch == 8
    assert rented.batch == 32
    assert laptop.imgsz == rented.imgsz == 640  # the experiment is unchanged


def test_profile_gives_one_batch_across_the_whole_sweep():
    """The point of profiles: a constant batch removes the ablation's confound."""
    batches = {load_config(name, profile="cuda24").batch for name in ARMS}
    assert batches == {32}, "the arms must not vary batch size on one machine"


def test_profile_is_recorded_on_the_config():
    assert load_config("pretrained", profile="cuda48").profile == "cuda48"


@pytest.mark.parametrize("name", ["mps", "cpu", "cuda12", "cuda24", "cuda48"])
def test_every_profile_sets_only_hardware_keys(name):
    """A profile that could set epochs or imgsz would make two runs with the
    same label mean different things."""
    assert set(load_profile(name)) <= {"batch", "workers", "device", "train_args"}


def test_unknown_profile_lists_the_alternatives():
    with pytest.raises(FileNotFoundError, match="cuda24"):
        load_profile("cuda9000")


@pytest.mark.parametrize(
    ("hw", "expected"),
    [
        (Hardware("cpu", "CPU", 32), CPU),
        (Hardware("0", "RTX 4070", 12), CUDA_SMALL),
        (Hardware("0", "RTX 4090", 24), CUDA_MEDIUM),
        (Hardware("0", "A6000", 48), CUDA_LARGE),
        (Hardware("0", "A100", 80), CUDA_LARGE),
    ],
)
def test_auto_profile_matches_hardware(hw, expected):
    assert auto_profile(hw) == expected


def test_auto_profile_is_conservative_at_boundaries():
    """A card just under a threshold must not get the bigger profile: an OOM
    two hours into a rented run costs more than unused memory."""
    assert auto_profile(Hardware("0", "card", 19.9)) == CUDA_SMALL
    assert auto_profile(Hardware("0", "card", 39.9)) == CUDA_MEDIUM


class TestProbeFit:
    """Timing a run on 10% of the data and dividing by 0.1 assumes epoch time
    is proportional to dataset size. It is not -- there is a fixed per-epoch
    overhead, and that naive scaling multiplies it by ten."""

    def test_recovers_known_overhead_and_rate(self):
        from thermaldet.probe import _fit

        # Ground truth: 4s overhead, 10ms per image.
        overhead, rate = _fit([(754, 4.0 + 0.010 * 754), (2262, 4.0 + 0.010 * 2262)])

        assert overhead == pytest.approx(4.0, abs=1e-6)
        assert rate == pytest.approx(0.010, abs=1e-9)

    def test_naive_scaling_would_have_overestimated(self):
        from thermaldet.probe import _fit

        overhead, rate = _fit([(754, 11.5), (2262, 11.5 + 0.006 * 1508)])
        fitted_full = overhead + rate * 7543
        naive_full = 11.5 / 0.1

        assert naive_full > fitted_full
        # The error is (1/fraction - 1) x overhead -- 9x here, not a rounding
        # difference.
        assert naive_full - fitted_full == pytest.approx(9 * overhead, rel=0.01)

    def test_indistinguishable_runs_raise_instead_of_extrapolating(self):
        """Two runs lost in the noise must not yield a confident wrong number."""
        from thermaldet.probe import _fit

        with pytest.raises(RuntimeError, match="too short to measure"):
            _fit([(754, 11.5), (2262, 11.2)])

    def test_identical_fractions_rejected(self):
        from thermaldet.probe import _fit

        with pytest.raises(ValueError, match="must differ"):
            _fit([(754, 11.5), (754, 11.7)])
