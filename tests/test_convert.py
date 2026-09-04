"""Dataset conversion: label parsing, class mapping and the frame list."""

from __future__ import annotations

import json

import pytest

from thermaldet.convert import (
    RGB,
    THERMAL,
    _yolo_line,
    check_root,
    coco_labels,
    frame_index,
)


def _touch(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")


class TestSpectra:
    def test_thermal_has_the_16_bit_directory(self):
        """v2 keeps the raw frames under `analyticsData`, which its release
        notes mention once. Getting it wrong makes the radiometry ablation
        look impossible when it is not."""
        assert THERMAL.has_raw
        assert THERMAL.image_subdirs["raw"] == "analyticsData"

    def test_rgb_has_no_raw_source(self):
        assert not RGB.has_raw

    def test_both_spectra_cover_the_same_three_splits(self):
        """The modality comparison needs a matching split on each side."""
        assert set(THERMAL.splits) == set(RGB.splits) == {"train", "val", "test"}

    def test_a_download_that_is_not_v2_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="v2"):
            check_root(tmp_path)


class TestYoloLine:
    def test_converts_a_centred_box(self):
        line = _yolo_line(0, [160, 128, 320, 256], 640, 512)
        assert line == "0 0.500000 0.500000 0.500000 0.500000"

    def test_clips_a_box_that_runs_off_the_frame(self):
        """FLIR boxes overhang the sensor edge; Ultralytics wants them inside."""
        cls, cx, _, w, _ = _yolo_line(1, [-20, -20, 60, 60], 640, 512).split()

        assert cls == "1"
        assert float(w) == pytest.approx(40 / 640)
        assert float(cx) == pytest.approx(20 / 640)

    @pytest.mark.parametrize("bbox", [[10, 10, 0, 40], [10, 10, 40, 0.5]])
    def test_drops_degenerate_boxes(self, bbox):
        """A zero-width box becomes NaN loss an hour into a run."""
        assert _yolo_line(0, bbox, 640, 512) is None


class TestCocoLabels:
    @staticmethod
    def _write(tmp_path, name="coco.json"):
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    # v2 keeps COCO's ids, so they are neither contiguous nor
                    # in the order the YOLO indices need.
                    "categories": [
                        {"id": 1, "name": "person"},
                        {"id": 3, "name": "car"},
                        {"id": 10, "name": "light"},
                        {"id": 17, "name": "dog"},
                    ],
                    "images": [
                        {"id": 7, "width": 640, "height": 512, "file_name": "data/frame-1.jpg"}
                    ],
                    "annotations": [
                        {"image_id": 7, "category_id": 3, "bbox": [0, 0, 64, 64]},
                        {"image_id": 7, "category_id": 1, "bbox": [0, 0, 64, 64]},
                        {"image_id": 7, "category_id": 10, "bbox": [0, 0, 64, 64]},
                        {"image_id": 7, "category_id": 17, "bbox": [0, 0, 64, 64]},
                        {"image_id": 7, "category_id": 1, "bbox": [0, 0, 64, 64], "iscrowd": True},
                    ],
                }
            )
        )
        return path

    def test_matches_categories_by_name_not_id(self, tmp_path):
        labels = coco_labels(self._write(tmp_path), ["person", "car"])
        assert sorted(line.split()[0] for line in labels["frame-1"]) == ["0", "1"]

    def test_yolo_index_follows_the_requested_order(self, tmp_path):
        """Thermal and RGB must map the same class to the same index, or the
        modality comparison is comparing different label spaces."""
        labels = coco_labels(self._write(tmp_path), ["light", "person", "car"])
        by_class = {line.split()[0] for line in labels["frame-1"]}
        assert by_class == {"0", "1", "2"}

    def test_unkept_classes_and_crowds_are_dropped(self, tmp_path):
        labels = coco_labels(self._write(tmp_path), ["person", "car"])
        assert len(labels["frame-1"]) == 2  # dog and light excluded, iscrowd excluded

    def test_a_frame_with_no_kept_object_still_gets_an_entry(self, tmp_path):
        """Ultralytics reads an empty label as background, which is what it is.
        Omitting the frame instead throws away a negative."""
        labels = coco_labels(self._write(tmp_path), ["light"])
        assert labels["frame-1"] and all(line.startswith("0") for line in labels["frame-1"])

    def test_a_missing_class_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="bike"):
            coco_labels(self._write(tmp_path), ["bike"])


class TestFrameIndex:
    def test_intersects_across_every_source_not_just_the_one_in_use(self, tmp_path):
        """Two arms that differ only in preprocessing must not also differ in
        which frames they trained on."""
        _touch(tmp_path / "data", ["a.jpg", "b.jpg"])
        _touch(tmp_path / "analyticsData", ["a.tiff", "c.tiff"])

        stems, counts = frame_index(tmp_path, THERMAL, {"a": [], "b": [], "c": []})

        assert stems == ["a"]
        assert counts["labelled"] == 3
        assert counts["kept"] == 1

    def test_rgb_is_indexed_on_its_single_source(self, tmp_path):
        _touch(tmp_path / "data", ["a.jpg", "b.jpg"])

        stems, _ = frame_index(tmp_path, RGB, {"a": [], "b": [], "ghost": []})
        assert stems == ["a", "b"]

    def test_a_label_with_no_image_anywhere_is_excluded(self, tmp_path):
        _touch(tmp_path / "data", ["a.jpg"])
        _touch(tmp_path / "analyticsData", ["a.tiff"])

        stems, _ = frame_index(tmp_path, THERMAL, {"a": [], "ghost": []})
        assert stems == ["a"]


class TestCopyCollisionDuplicates:
    """A dataset copied between machines collects "frame-1 2.jpg" files.

    They duplicate a frame already in the set, so left in they train twice --
    and because they only survive in whichever directories the copy touched,
    they land in some arms and not others.
    """

    def test_duplicate_images_are_not_counted_as_frames(self, tmp_path):
        _touch(tmp_path / "data", ["a.jpg", "a 2.jpg"])
        _touch(tmp_path / "analyticsData", ["a.tiff", "a 2.tiff"])

        stems, _ = frame_index(tmp_path, THERMAL, {"a": [], "a 2": []})
        assert stems == ["a"]

    def test_a_real_stem_ending_in_a_digit_is_untouched(self, tmp_path):
        """v2 stems end in a random id, but frame-000108 must still survive."""
        _touch(tmp_path / "data", ["video-x-frame-000108.jpg"])
        _touch(tmp_path / "analyticsData", ["video-x-frame-000108.tiff"])

        stems, _ = frame_index(tmp_path, THERMAL, {"video-x-frame-000108": []})
        assert stems == ["video-x-frame-000108"]
