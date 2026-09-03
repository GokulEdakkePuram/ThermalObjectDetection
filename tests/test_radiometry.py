"""The 16-bit to 8-bit mappings, which are the second ablation's only variable."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from thermaldet.radiometry import (
    GlobalLinear,
    PercentileStretch,
    _to_uint8,
    frame_spans,
    measure_window,
    read_raw,
)


def _write_tiff(path, counts):
    Image.fromarray(np.asarray(counts, dtype=np.uint16)).save(path)
    return path


def _ramp(lo, hi, shape=(32, 32)):
    return np.linspace(lo, hi, num=shape[0] * shape[1]).reshape(shape).astype(np.uint16)


class TestToUint8:
    def test_maps_the_window_onto_the_full_range(self):
        out = _to_uint8(np.array([6000, 7000, 8000]), 6000, 8000)
        assert out.tolist() == [0, 127, 255]

    def test_clips_outside_the_window(self):
        out = _to_uint8(np.array([5000, 9000]), 6000, 8000)
        assert out.tolist() == [0, 255]

    def test_a_degenerate_window_gives_black_rather_than_dividing_by_zero(self):
        """A frame the sensor recorded as uniform is rare and real."""
        assert _to_uint8(np.full((4, 4), 6000), 6000, 6000).max() == 0


class TestGlobalLinear:
    def test_the_same_counts_map_to_the_same_grey_in_every_frame(self, tmp_path):
        """The whole point of the global arm: absolute radiometry survives.

        Two frames at different temperatures must come out at different
        brightness, because that difference is the signal being preserved.
        """
        cold = _write_tiff(tmp_path / "cold.tiff", np.full((8, 8), 6200))
        hot = _write_tiff(tmp_path / "hot.tiff", np.full((8, 8), 7400))
        render = GlobalLinear(6000, 8000)

        render(cold, tmp_path / "cold.png")
        render(hot, tmp_path / "hot.png")

        cold_out = np.array(Image.open(tmp_path / "cold.png"))
        hot_out = np.array(Image.open(tmp_path / "hot.png"))
        assert cold_out.mean() < hot_out.mean()
        assert cold_out.mean() == pytest.approx(255 * 200 / 2000, abs=1)

    def test_a_narrow_frame_uses_only_part_of_the_output_range(self, tmp_path):
        """The measured cost of the global arm, as a unit test.

        A frame spanning 873 counts inside a 2,096-count window reaches ~106
        of 255 levels. That is the trade the ablation is testing.
        """
        src = _write_tiff(tmp_path / "narrow.tiff", _ramp(6500, 7373))
        GlobalLinear(5963, 8059)(src, tmp_path / "narrow.png")

        out = np.array(Image.open(tmp_path / "narrow.png"))
        assert out.max() - out.min() == pytest.approx(106, abs=3)


class TestPercentileStretch:
    def test_fills_the_range_whatever_the_offset(self, tmp_path):
        """And that is exactly what it destroys: after this, a cold frame and a
        hot frame are indistinguishable by brightness."""
        for name, base in (("cold", 6000), ("hot", 7400)):
            src = _write_tiff(tmp_path / f"{name}.tiff", _ramp(base, base + 800))
            PercentileStretch()(src, tmp_path / f"{name}.png")

        cold = np.array(Image.open(tmp_path / "cold.png"))
        hot = np.array(Image.open(tmp_path / "hot.png"))
        assert cold.mean() == pytest.approx(hot.mean(), abs=1)
        assert cold.min() == 0 and cold.max() == 255


class TestMeasurement:
    def test_read_raw_keeps_16_bit_precision(self, tmp_path):
        src = _write_tiff(tmp_path / "a.tiff", np.array([[6000, 40000]]))
        counts = read_raw(src)
        assert counts.dtype == np.uint16
        assert counts.max() == 40000

    def test_window_ignores_a_single_hot_outlier(self, tmp_path):
        """One sun-facing radiator must not set the window for the dataset."""
        frame = np.full((32, 32), 6500, dtype=np.uint16)
        frame[0, 0] = 60000
        _write_tiff(tmp_path / "a.tiff", frame)

        _, hi = measure_window(tmp_path, sample=1)
        assert hi < 10000

    def test_window_is_reproducible_across_builds(self, tmp_path):
        """An arm whose preprocessing shifts between rebuilds is not an arm."""
        for i in range(12):
            _write_tiff(tmp_path / f"{i}.tiff", _ramp(6000 + 40 * i, 7000 + 40 * i))

        assert measure_window(tmp_path, sample=5) == measure_window(tmp_path, sample=5)

    def test_frame_spans_reports_what_a_median_frame_loses(self, tmp_path):
        for i in range(12):
            _write_tiff(tmp_path / f"{i}.tiff", _ramp(6000 + 100 * i, 6800 + 100 * i))

        spans = frame_spans(tmp_path, sample=12)

        assert spans["median_frame_span"] < spans["window_span"] <= spans["dataset_span"]
        assert 0 < spans["levels_under_global_map"] < 255

    def test_an_empty_directory_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="No 16-bit"):
            measure_window(tmp_path)
