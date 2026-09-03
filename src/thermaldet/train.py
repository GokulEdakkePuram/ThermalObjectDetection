"""Training entry point."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, load_config
from .hardware import detect
from .paths import RUNS_DIR, configure_ultralytics
from .tracking import configure as configure_tracking


def train(
    config: str | ExperimentConfig,
    profile: str | None = None,
    tracker: str | None = None,
) -> dict[str, Any]:
    """Fine-tune a YOLO model according to an experiment config.

    ``profile`` overlays a hardware profile (or ``"auto"`` to detect one), so
    the same experiment runs unchanged on a laptop or a rented GPU. The
    resolved config is written next to the run's weights, so a result in
    ``runs/`` can always be traced back to the exact settings that produced it
    -- including which machine profile it ran under.
    """
    from ultralytics import YOLO

    configure_ultralytics()
    cfg = load_config(config, profile=profile) if isinstance(config, str) else config
    if tracker is not None:
        cfg.tracker = tracker

    hw = detect()
    active = configure_tracking(cfg.tracker, run_name=cfg.name)
    print(f"[thermaldet] hardware: {hw.describe()}")
    print(f"[thermaldet] profile : {cfg.profile or '(none, using config defaults)'}")
    print(f"[thermaldet] tracking: {active}")

    model = YOLO(cfg.model)
    if cfg.pretrained:
        # Matching layers transfer by name and shape; the rest stay as built.
        print(f"[thermaldet] transferring weights from {cfg.pretrained}")
        model = model.load(cfg.pretrained)

    kwargs = cfg.to_train_kwargs()
    kwargs.setdefault("project", str(RUNS_DIR / "train"))
    print(f"[thermaldet] training '{cfg.name}' on {kwargs['device']} ({cfg.model})")

    # Record the resolved config before training rather than after, so that a
    # failure in any post-training step cannot leave a run whose weights and
    # metrics are fine but which is unidentifiable.
    #
    # It goes to a staging file rather than straight into the run directory:
    # creating that directory early makes Ultralytics think the name is taken,
    # so it silently trains into `<name>2` and every downstream path that
    # expects `<name>` breaks.
    payload = json.dumps(asdict(cfg), indent=2)
    staging = RUNS_DIR / ".pending"
    staging.mkdir(parents=True, exist_ok=True)
    pending = staging / f"{cfg.name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    pending.write_text(payload)

    results = model.train(**kwargs)

    save_dir = Path(results.save_dir)
    (save_dir / "thermaldet_config.json").write_text(payload)
    pending.unlink(missing_ok=True)  # landed safely; no need for the copy

    return {
        "name": cfg.name,
        "save_dir": str(save_dir),
        "best_weights": str(save_dir / "weights" / "best.pt"),
    }
