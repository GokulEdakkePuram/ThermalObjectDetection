"""Experiment tracking.

Ultralytics ships callbacks for Weights & Biases and MLflow; they activate
when the corresponding Ultralytics setting is on *and* the package imports.
This makes that a config choice rather than a global toggle someone has to
remember to flip, and fails loudly when a tracker is asked for but missing --
training for an hour on a rented GPU and logging nothing is the failure mode
worth preventing.

W&B is the default here because its run pages are shareable, so the README
can point a reader at real training curves. MLflow is the option when the data
should stay on the machine; with no tracking URI set it writes to a local file
store under ``runs/mlflow``.
"""

from __future__ import annotations

import os

# Ultralytics setting name per tracker, and the import it needs.
TRACKERS = {
    "wandb": ("wandb", "wandb"),
    "mlflow": ("mlflow", "mlflow"),
    "tensorboard": ("tensorboard", "torch.utils.tensorboard"),
}

DEFAULT_PROJECT = "thermaldet"


def configure(
    tracker: str = "none",
    project: str = DEFAULT_PROJECT,
    run_name: str = "",
    entity: str = "",
) -> str:
    """Enable one tracker and disable the rest. Returns the active tracker.

    ``entity`` selects which W&B account or team owns the run. It matters when
    an account's default entity is an organisation: a project created there
    inherits the org's visibility rules, so a run intended to be linked
    publicly may not be yours to publish. Set it explicitly, or via
    ``WANDB_ENTITY``.
    """
    from ultralytics.utils import SETTINGS

    tracker = (tracker or "none").lower()
    if tracker not in {*TRACKERS, "none"}:
        raise ValueError(f"Unknown tracker {tracker!r}. Choose from {sorted(TRACKERS)} or 'none'.")

    # Turn everything off first, so a previous run's setting cannot leak in.
    SETTINGS.update({name: False for name, _ in TRACKERS.values()})

    if tracker == "none":
        return "none"

    setting, module = TRACKERS[tracker]
    try:
        __import__(module)
    except ImportError as exc:
        raise RuntimeError(
            f"Tracker {tracker!r} requested but {module!r} is not installed. "
            f"Install it with: uv sync --extra {tracker}"
        ) from exc

    SETTINGS.update({setting: True})

    if tracker == "wandb":
        os.environ.setdefault("WANDB_PROJECT", project)
        if entity:
            os.environ.setdefault("WANDB_ENTITY", entity)
        if run_name:
            os.environ.setdefault("WANDB_NAME", run_name)
    elif tracker == "mlflow":
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", project)
        if run_name:
            os.environ.setdefault("MLFLOW_RUN", run_name)

    return tracker
