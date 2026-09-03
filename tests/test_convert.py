"""Dataset conversion: label parsing, class remapping and the frame list."""

from __future__ import annotations

import json

import pytest

from thermaldet.convert import (
    Layout,
    _yolo_line,
    adopted_labels,
    coco_labels,
    detect_layout,
    frame_index,
)

LAYOUT = Layout("1.3", {}, {"agc": "thermal_8_bit", "raw": "thermal_16_bit"})


def _touch(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")


class TestYoloLine:
    def test_converts_a_centred_box(self):
        assert (
            _yolo_line(0, [160, 128, 320, 256], 640, 512) == "0 0.500000 0.500000 0.500000 0.500000"
        )

    def test_clips_a_box_that_runs_off_the_frame(self):
        """FLIR boxes overhang the sensor edge; Ultralytics wants them inside."""
        line = _yolo_line(1, [-20, -20, 60, 60], 640, 512)
        cls, cx, cy, w, h = line.split()

        assert cls == "1"
        assert float(w) == pytest.approx(40 / 640)
        assert float(cx) == pytest.approx(20 / 640)

    @pytest.mark.parametrize("bbox", [[10, 10, 0, 40], [10, 10, 40, 0.5]])
    def test_drops_degenerate_boxes(self, bbox):
        """A zero-width box becomes NaN loss several hours later."""
        assert _yolo_line(0, bbox, 640, 512) is None


class TestCocoLabels:
    @staticmethod
    def _write(tmp_path):
        path = tmp_path / "thermal_annotations.json"
        path.write_text(
            json.dumps(
                {
                    # 1.3 keeps COCO's original ids, which are neither
                    # contiguous nor in the order we want them.
                    "categories": [
                        {"id": 1, "name": "person"},
                        {"id": 3, "name": "car"},
                        {"id": 18, "name": "dog"},
                    ],
                    "images": [
                        {"id": 7, "width": 640, "height": 512, "file_name": "x/FLIR_1.jpeg"}
                    ],
                    "annotations": [
                        {"image_id": 7, "category_id": 3, "bbox": [0, 0, 64, 64]},
                        {"image_id": 7, "category_id": 1, "bbox": [0, 0, 64, 64]},
                        {"image_id": 7, "category_id": 18, "bbox": [0, 0, 64, 64]},
                        {"image_id": 7, "category_id": 1, "bbox": [0, 0, 64, 64], "iscrowd": 1},
                    ],
                }
            )
        )
        return path

    def test_matches_categories_by_name_not_id(self, tmp_path):
        """Reading by id works on exactly one FLIR release."""
        labels = coco_labels(self._write(tmp_path), ["person", "car"])
        assert sorted(line.split()[0] for line in labels["FLIR_1"]) == ["0", "1"]

    def test_class_order_follows_the_requested_list(self, tmp_path):
        labels = coco_labels(self._write(tmp_path), ["car", "person"])
        by_class = {line.split()[0] for line in labels["FLIR_1"]}
        assert by_class == {"0", "1"}

    def test_unkept_classes_and_crowds_are_dropped(self, tmp_path):
        labels = coco_labels(self._write(tmp_path), ["person", "car"])
        assert len(labels["FLIR_1"]) == 2  # dog excluded, iscrowd excluded

    def test_an_image_with_no_kept_object_still_gets_an_entry(self, tmp_path):
        """Ultralytics treats an empty label file as background, which is what
        it should be -- silently omitting the frame loses a negative."""
        labels = coco_labels(self._write(tmp_path), ["person"])
        assert labels["FLIR_1"] and all(line.startswith("0") for line in labels["FLIR_1"])

    def test_a_missing_class_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="bicycle"):
            coco_labels(self._write(tmp_path), ["bicycle"])


class TestAdoptedLabels:
    def test_remaps_class_ids_and_drops_the_rest(self, tmp_path):
        """The escape hatch for a download whose annotation JSONs are gone."""
        (tmp_path / "FLIR_1.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n2 0.5 0.5 0.1 0.1\n3 0.5 0.5 0.1 0.1\n"
        )

        labels = adopted_labels(tmp_path, ["person", "bicycle", "car", "dog"], ["car", "person"])

        # car was 2 and becomes 0; person was 0 and becomes 1; dog is dropped.
        assert labels["FLIR_1"] == ["1 0.5 0.5 0.1 0.1", "0 0.5 0.5 0.1 0.1"]


class TestFrameIndex:
    def test_intersects_across_every_source_not_just_the_one_in_use(self, tmp_path):
        """The regression this exists for.

        Taking the intersection per-arm left the 8-bit arm with 19 frames the
        16-bit arm did not have, so two runs that were meant to differ only in
        preprocessing also differed in their training set.
        """
        _touch(tmp_path / "thermal_8_bit", ["a.jpeg", "b.jpeg"])
        _touch(tmp_path / "thermal_16_bit", ["a.tiff", "c.tiff"])

        stems, counts = frame_index(tmp_path, LAYOUT, {"a": [], "b": [], "c": []})

        assert stems == ["a"]
        assert counts["labelled"] == 3
        assert counts["kept"] == 1

    def test_a_label_with_no_image_anywhere_is_excluded(self, tmp_path):
        _touch(tmp_path / "thermal_8_bit", ["a.jpeg"])
        _touch(tmp_path / "thermal_16_bit", ["a.tiff"])

        stems, _ = frame_index(tmp_path, LAYOUT, {"a": [], "ghost": []})
        assert stems == ["a"]


class TestDetectLayout:
    def test_recognises_1_3(self, tmp_path):
        _touch(tmp_path / "train" / "thermal_8_bit", [])
        layout = detect_layout(tmp_path)
        assert layout.version == "1.3"
        assert layout.has_raw

    def test_recognises_v2_and_knows_it_has_no_raw(self, tmp_path):
        """v2 ships more frames and no 16-bit, so it supports the transfer
        ablation and not the radiometry one."""
        _touch(tmp_path / "images_thermal_train", [])
        layout = detect_layout(tmp_path)
        assert layout.version == "v2"
        assert not layout.has_raw

    def test_an_unrecognised_tree_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="neither"):
            detect_layout(tmp_path)


class TestCopyCollisionDuplicates:
    """A dataset copied between machines collects "FLIR_00009 2.jpeg" files.

    They are duplicates of a frame already in the set. Left in, they enter
    training twice -- and because they only survive in whichever directories
    the copy touched, they land in some arms and not others.
    """

    def test_duplicate_images_are_not_counted_as_frames(self, tmp_path):
        _touch(tmp_path / "thermal_8_bit", ["a.jpeg", "a 2.jpeg"])
        _touch(tmp_path / "thermal_16_bit", ["a.tiff", "a 2.tiff"])

        stems, _ = frame_index(tmp_path, LAYOUT, {"a": [], "a 2": []})
        assert stems == ["a"]

    def test_duplicate_labels_are_skipped_when_adopting(self, tmp_path):
        (tmp_path / "FLIR_1.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (tmp_path / "FLIR_1 2.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        assert sorted(adopted_labels(tmp_path, ["person"], ["person"])) == ["FLIR_1"]

    def test_a_real_stem_ending_in_a_digit_is_untouched(self, tmp_path):
        """FLIR_00009 must survive; only "FLIR_00009 2" is a copy."""
        _touch(tmp_path / "thermal_8_bit", ["FLIR_00009.jpeg"])
        _touch(tmp_path / "thermal_16_bit", ["FLIR_00009.tiff"])

        stems, _ = frame_index(tmp_path, LAYOUT, {"FLIR_00009": []})
        assert stems == ["FLIR_00009"]
