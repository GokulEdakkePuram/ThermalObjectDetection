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
# Some templates mount the rented disk somewhere other than $HOME -- /workspace
# is the usual one -- and the container's own filesystem can be a few GB. Set
# WORKDIR to wherever `df -h` says the space actually is.
WORKDIR="${WORKDIR:-$HOME/thermaldet}"

echo "==> Checking for a GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "    nvidia-smi not found. This box has no usable GPU driver." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "==> Checking the driver is new enough"
# uv.lock pins torch 2.11 against a CUDA 13 runtime (nvidia-cudnn-cu13,
# nvidia-nccl-cu13, cuda-toolkit 13.x), so the host driver has to support CUDA
# 13.0 or newer -- roughly driver 580+. A 12.x driver fails with "NVIDIA driver
# on your system is too old", but only once torch is first imported: five paid
# minutes and 5 GB after uv sync started. Hence checking it before anything
# else. Most vast.ai listings are still 12.x; filter for it when renting.
REQUIRED_CUDA_MAJOR="${REQUIRED_CUDA_MAJOR:-13}"
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
# Budget: the zip is ~13 GB and has to coexist with its ~13 GB unpacked copy
# during extraction, the four arms add ~6 GB, and the CUDA 13 venv is ~12 GB.
# Peak is therefore around 45 GB before any run writes a checkpoint. vast.ai
# defaults to 10 GB unless you move the slider, which is the single most
# common way an instance gets wasted.
# Measured on the filesystem WORKDIR will live on, not on $HOME, so that
# overriding WORKDIR onto the big volume is actually checked.
mkdir -p "$(dirname "${WORKDIR}")"
AVAIL_GB=$(df -Pk "$(dirname "${WORKDIR}")" | awk 'NR==2 {print int($4/1048576)}')
echo "    ${AVAIL_GB} GB available"
if [ "${AVAIL_GB}" -lt 60 ]; then
    echo "    WARNING: under 60 GB free. Peak usage is ~45 GB (zip + unpacked" >&2
    echo "    dataset + venv), before arms or checkpoints. Re-rent with more disk" >&2
    echo "    now rather than discovering this three hours in." >&2
    echo "    If the rented volume is mounted elsewhere, re-run with" >&2
    echo "    WORKDIR=/workspace/thermaldet (check \`df -h\`)." >&2
fi

echo "==> Installing prerequisites"
# Some vast.ai images are minimal enough to lack git. Cheaper to check than to
# have the clone fail after the driver and disk checks have already passed.
for tool in git unzip curl; do
    command -v "${tool}" >/dev/null 2>&1 || {
        echo "    installing ${tool}"
        apt-get update -qq && apt-get install -y -qq "${tool}"
    }
done
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
# Before the dataset, not after. If this fails the instance is unusable, and
# there is no reason to have spent thirteen gigabytes finding that out.
uv run python -c "
import torch
assert torch.cuda.is_available(), 'torch cannot see the GPU'
p = torch.cuda.get_device_properties(0)
print(f'    {p.name}, {p.total_memory / 1024**3:.0f} GB, torch {torch.__version__}')
"

if [ -n "${GDRIVE_ID:-}" ]; then
    echo "==> Fetching the dataset from Google Drive"
    ./scripts/fetch_dataset.sh
fi

if [ -d "Dataset/FLIR_ADAS_v2/images_thermal_train" ]; then
cat <<'NEXT'

Ready, dataset included. Next:

  make arms
      Builds all four arms: three thermal, one visible. ~2 minutes, ~6 GB.

  uv run thermaldet probe pretrained rgb
      Calibrate before committing rental hours: confirms the profile's batch
      size fits, and projects how long the real runs take. Two fractions, so
      fixed per-epoch overhead is separated from the per-image rate.

  export WANDB_API_KEY=<key from wandb.ai/authorize>
      Preferred over an interactive login on an instance you will destroy.
      Every CLI here lives in the project venv, so the interactive form is
      `uv run wandb login` -- a bare `wandb` is not on PATH.

  tmux new -s train
  ./scripts/run_sweep.sh
      Inside tmux, so a dropped SSH connection does not kill seven runs.

NEXT
else
cat <<'NEXT'

Ready, except for the dataset. FLIR is behind a registration form, so it
cannot be fetched from source here. Either:

  GDRIVE_ID=<file id> ./scripts/fetch_dataset.sh
      Pull your own copy from Google Drive. The file has to be shared as
      "Anyone with the link"; the script checks what it actually got, because
      a sign-in page saves quite happily under a .zip name.

  scp -P <port> flir_adas_v2.zip root@<host>:~/thermaldet/Dataset/
      The fallback when Drive's public-download quota bites.

Then `make arms`, and see docs/06 for the rest.

NEXT
fi
