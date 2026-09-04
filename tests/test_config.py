"""Config loading, inheritance and validation."""

from __future__ import annotations

import pytest

from thermaldet.config import load_config, resolve_device

TRANSFER_ARMS = ["pretrained", "scratch", "frozen_stem", "frozen_backbone"]
RADIOMETRY_ARMS = ["pretrained", "global_map", "p1p99_map"]
MODALITY_ARMS = ["pretrained", "rgb"]


def test_extends_deep_merges_train_args():
    """A child overriding one hyperparameter must not drop its siblings."""
    smoke = load_config("smoke")

    assert smoke.train_args["fraction"] == 0.02  # set by the child
    assert smoke.train_args["plots"] is False  # overridden
    assert smoke.train_args["mosaic"] == 1.0  # inherited from base


def test_unknown_top_level_key_is_rejected(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("name: bad\nlr0: 0.01\n")  # lr0 belongs under train_args

    with pytest.raises(ValueError, match="train_args"):
        load_config(cfg)


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")


def test_to_train_kwargs_resolves_device_and_flattens():
    kwargs = load_config("pretrained").to_train_kwargs()

    assert kwargs["imgsz"] == 640
    assert kwargs["device"] in {"cpu", "mps", "0"}
    assert kwargs["mosaic"] == 1.0  # train_args flattened in


def test_explicit_device_passes_through():
    assert resolve_device("cpu") == "cpu"


class TestAblationsStayComparable:
    """The README's claims depend on these staying true.

    An ablation is only an ablation while its arms differ in one place. These
    assertions are what stops a hyperparameter tweak meant for one arm from
    quietly making the comparison meaningless.
    """

    @pytest.mark.parametrize("name", TRANSFER_ARMS)
    def test_transfer_arms_share_the_schedule_and_the_data(self, name):
        control, arm = load_config("pretrained"), load_config(name)

        assert arm.data == control.data
        assert arm.epochs == control.epochs
        assert arm.imgsz == control.imgsz
        assert arm.seed == control.seed

    @pytest.mark.parametrize("name", RADIOMETRY_ARMS + MODALITY_ARMS)
    def test_pixel_arms_share_everything_but_the_pixels(self, name):
        control, arm = load_config("pretrained"), load_config(name)

        assert arm.model == control.model
        assert arm.epochs == control.epochs
        assert arm.train_args == control.train_args

    def test_each_arm_differs_from_the_control_in_exactly_one_place(self):
        control = load_config("pretrained")
        expected = {
            "scratch": "model",
            "frozen_stem": "train_args",
            "frozen_backbone": "train_args",
            "global_map": "data",
            "p1p99_map": "data",
            "rgb": "data",
        }
        for name, key in expected.items():
            arm = load_config(name)
            differing = {
                field
                for field in ("model", "data", "epochs", "imgsz", "batch", "seed", "train_args")
                if getattr(arm, field) != getattr(control, field)
            }
            assert differing == {key}, f"{name} differs in {sorted(differing)}, expected {{{key}}}"

    def test_the_rgb_arm_keeps_the_thermal_augmentation(self):
        """Hue and saturation jitter stay off on the visible arm too. It is a
        handicap, and it is the price of the comparison being single-variable."""
        control, rgb = load_config("pretrained"), load_config("rgb")

        assert rgb.train_args == control.train_args
        assert rgb.train_args["hsv_s"] == 0.0

    def test_the_two_frozen_arms_freeze_different_depths(self):
        assert load_config("frozen_stem").train_args["freeze"] == 2
        # YOLO11's backbone is layers 0-10, so the whole of it is 11, not the
        # YOLOv8-era 10 that gets copied around.
        assert load_config("frozen_backbone").train_args["freeze"] == 11
