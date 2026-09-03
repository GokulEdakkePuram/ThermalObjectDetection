"""The explicit loop's parameter grouping, which is the reason it exists."""

from __future__ import annotations

import pytest
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

from thermaldet.manual import build_optimizer, build_param_groups, freeze_layers, layer_index

BACKBONE_LAYERS = 11  # YOLO11: layers 0-10


@pytest.fixture(scope="module")
def model():
    """A real YOLO11s, built from its architecture spec so no download is needed."""
    return DetectionModel("yolo11s.yaml", ch=3, nc=3, verbose=False)


def test_layer_index_reads_the_top_level_index():
    assert layer_index("model.7.conv.weight") == 7
    assert layer_index("model.bias") is None


class TestParamGroups:
    def test_default_reproduces_the_stock_three_group_layout(self, model):
        """`--backbone-lr-mult 1.0` must be the stock trainer, or the
        comparison against it means nothing."""
        groups = build_param_groups(model, 5e-4, BACKBONE_LAYERS, 1.0)

        assert len(groups) == 3
        assert {g["lr_mult"] for g in groups} == {1.0}

    def test_splitting_by_depth_separates_backbone_from_head(self, model):
        """Five groups, not six: YOLO11's backbone convolutions carry no bias
        at all -- every one is followed by a BatchNorm that supplies the
        shift. So the backbone has weight and norm groups and nothing else.
        """
        groups = build_param_groups(model, 5e-4, BACKBONE_LAYERS, 5.0)

        assert len(groups) == 5
        assert sorted({g["lr_mult"] for g in groups}) == [1.0, 5.0]
        assert not any(g["is_bias"] for g in groups if g["lr_mult"] == 5.0)

    def test_every_trainable_parameter_lands_in_exactly_one_group(self, model):
        groups = build_param_groups(model, 5e-4, BACKBONE_LAYERS, 5.0)

        seen = [id(p) for g in groups for p in g["params"]]
        trainable = [id(p) for p in model.parameters() if p.requires_grad]
        assert len(seen) == len(set(seen))
        assert set(seen) == set(trainable)

    def test_only_weights_carry_weight_decay(self, model):
        """Decaying a BatchNorm scale toward zero is a way to quietly break a
        network, and it is the default if groups are built carelessly."""
        decayed = {
            id(p) for g in build_param_groups(model, 5e-4) if g["weight_decay"] for p in g["params"]
        }
        norms = {
            id(p)
            for m in model.modules()
            if isinstance(m, nn.BatchNorm2d)
            for p in m.parameters(recurse=False)
        }
        assert decayed.isdisjoint(norms)

    def test_freezing_the_backbone_leaves_only_head_groups(self, model):
        frozen = DetectionModel("yolo11s.yaml", ch=3, nc=3, verbose=False)
        freeze_layers(frozen, BACKBONE_LAYERS)

        groups = build_param_groups(frozen, 5e-4, BACKBONE_LAYERS, 5.0)
        assert {g["lr_mult"] for g in groups} == {1.0}

    def test_no_trainable_parameters_fails_loudly(self, model):
        frozen = DetectionModel("yolo11s.yaml", ch=3, nc=3, verbose=False)
        for p in frozen.parameters():
            p.requires_grad_(False)

        with pytest.raises(RuntimeError, match="No trainable parameters"):
            build_param_groups(frozen, 5e-4)


class TestOptimizer:
    def test_the_multiplier_reaches_the_optimizer(self, model):
        """torch.optim keeps unknown keys in a group dict, which is what lets
        lr_mult ride along -- worth pinning, since it is undocumented."""
        groups = build_param_groups(model, 5e-4, BACKBONE_LAYERS, 5.0)
        optimizer = build_optimizer("AdamW", groups, lr=1e-3, momentum=0.937)

        lrs = sorted({g["lr"] for g in optimizer.param_groups})
        assert lrs == pytest.approx([1e-3, 5e-3])
        assert all("is_bias" in g for g in optimizer.param_groups)

    def test_an_unknown_optimizer_is_rejected(self, model):
        with pytest.raises(ValueError):
            build_optimizer("Adagrad", build_param_groups(model, 5e-4), 1e-3, 0.9)
