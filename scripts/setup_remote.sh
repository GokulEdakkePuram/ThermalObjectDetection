#!/usr/bin/env bash
# Provision a freshly rented GPU box for training.
#
#   curl -fsSL https://raw.githubusercontent.com/GokulEdakkePuram/ThermalObjectDetection/main/scripts/setup_remote.sh | bash
#
# Assumes an Ubuntu image with an NVIDIA driver already present (any of the
# PyTorch or CUDA templates on vast.ai / RunPod). Everything Python-side comes
# from uv.lock, so the environment matches the one the results were produced
# on.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/GokulEdakkePuram/ThermalObjectDetection.git}"
WORKDIR="${WORKDIR:-$HOME/thermaldet}"

echo "==> Checking for a GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "    nvidia-smi not found. This box has no usable GPU driver." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "==> Checking the driver is new enough"
# The pinned torch wheels bundle their own CUDA runtime, which needs a host
# driver that supports that CUDA major version. An older driver fails with
# "NVIDIA driver on your system is too old" -- but only after uv sync has spent
# five paid minutes pulling 5 GB, so check it up front.
REQUIRED_CUDA_MAJOR="${REQUIRED_CUDA_MAJOR:-12}"
HOST_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
if [ -z "${HOST_CUDA}" ]; then
    echo "    WARNING: could not read the driver's CUDA version; continuing anyway." >&2
elif [ "${HOST_CUDA%%.*}" -lt "${REQUIRED_CUDA_MAJOR}" ]; then
    echo "    This host supports CUDA ${HOST_CUDA}, but the locked torch build needs" >&2
    echo "    CUDA ${REQUIRED_CUDA_MAJOR}.0 or newer. Destroy this instance and rent one" >&2
    echo "    filtered to CUDA ${REQUIRED_CUDA_MAJOR}.0+ -- cheaper than working around it," >&2
    echo "    and it keeps the environment identical to uv.lock." >&2
    exit 1
else
    echo "    driver supports CUDA ${HOST_CUDA}"
fi

echo "==> Checking disk space"
# The FLIR 1.3 download is ~18 GB, the three preprocessing arms add ~6 GB, and
# the CUDA venv is ~8 GB. Rentals default to 10 GB of disk, which is the most
# common way an instance gets wasted.
AVAIL_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "    ${AVAIL_GB} GB available"
if [ "${AVAIL_GB}" -lt 60 ]; then
    echo "    WARNING: under 60 GB free. The dataset alone is ~18 GB and the three" >&2
    echo "    arms add ~6 GB. Re-rent with more disk before pulling anything." >&2
fi

echo "==> Installing uv"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "==> Cloning into ${WORKDIR}"
if [ -d "${WORKDIR}/.git" ]; then
    git -C "${WORKDIR}" pull --ff-only
else
    git clone "${REPO_URL}" "${WORKDIR}"
fi
cd "${WORKDIR}"

echo "==> Installing dependencies (CUDA wheels come from uv.lock)"
uv sync --extra wandb

echo "==> Verifying CUDA is visible to torch"
uv run python -c "
import torch
assert torch.cuda.is_available(), 'torch cannot see the GPU'
p = torch.cuda.get_device_properties(0)
print(f'    {p.name}, {p.total_memory / 1024**3:.0f} GB, torch {torch.__version__}')
"

cat <<'NEXT'

Ready, except for the dataset -- FLIR ADAS is behind a registration form, so
it cannot be fetched from here. Put the download at Dataset/FLIR_ADAS_1_3/
(scp, or an rclone remote), then:

  make arms
      Builds all three preprocessing arms. ~2 minutes, ~6 GB.
      Add ADOPT="--adopt-labels Dataset/FLIR_ADAS_1_3/yolo/labels" if the
      download's thermal_annotations.json files are missing.

  uv run thermaldet probe pretrained global_map
      Calibrate before committing rental hours: confirms the profile's batch
      size fits, and projects how long the real runs take. Two fractions, so
      fixed per-epoch overhead is separated from the per-image rate.

  export WANDB_API_KEY=<key from wandb.ai/authorize>
      Preferred over an interactive login on an instance you will destroy.
      Every CLI here lives in the project venv, so the interactive form is
      `uv run wandb login` -- a bare `wandb` is not on PATH.

  tmux new -s train
  ./scripts/run_sweep.sh
      Inside tmux, so a dropped SSH connection does not kill six runs.

NEXT
