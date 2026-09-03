"""Canonical project paths.

Everything the pipeline writes lands under the repo root, so a run is
reproducible from a fresh clone without depending on global Ultralytics state
left behind by some earlier session.

Two directories hold data and they are not the same thing. ``Dataset/`` is the
FLIR download exactly as it ships, treated as read-only. ``data/`` is what the
converter builds out of it -- one Ultralytics-shaped tree per preprocessing
arm. Keeping them apart means a botched conversion is one ``rm -rf data/``
away from fixed, rather than a re-download.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

CONFIG_DIR = PROJECT_ROOT / "configs"
DATASET_DIR = PROJECT_ROOT / "Dataset"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_dirs() -> None:
    """Create the output directories that runs write into."""
    for d in (DATA_DIR, RUNS_DIR, WEIGHTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def configure_ultralytics() -> None:
    """Redirect Ultralytics' global directories into the repo.

    Ultralytics persists ``datasets_dir``/``runs_dir`` in a user-level settings
    file. Left alone it scatters artifacts across the home directory, which
    makes results impossible to reproduce or clean up. We pin them per-process.
    """
    from ultralytics.utils import SETTINGS

    ensure_dirs()
    SETTINGS.update(
        {
            "datasets_dir": str(DATA_DIR),
            "runs_dir": str(RUNS_DIR),
            "weights_dir": str(WEIGHTS_DIR),
        }
    )
