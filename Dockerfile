# Reproducible training image.
#
# Deliberately built on plain Ubuntu rather than an nvidia/cuda base: the torch
# wheels pinned in uv.lock ship their own CUDA runtime (nvidia-cudnn, nccl,
# cublas ...), so a CUDA base image would mean two copies of the toolkit and a
# chance of them disagreeing. The only host requirement is an NVIDIA driver new
# enough for the CUDA version those wheels were built against -- check with
# `nvidia-smi` before pulling.
#
#   docker build -t thermaldet .
#   docker run --gpus all \
#       -v $PWD/Dataset:/app/Dataset:ro -v $PWD/runs:/app/runs \
#       thermaldet uv run thermaldet train pretrained --track wandb
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:/app/.venv/bin:$PATH"

# libgl and libglib are OpenCV's runtime dependencies; without them the import
# succeeds at build time and fails on the first frame read.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Dependencies resolve from the lockfile alone, so this layer stays cached
# until the lock actually changes -- source edits do not trigger a 5 GB
# reinstall.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --extra wandb

COPY . .
RUN uv sync --frozen --extra wandb

# The FLIR download is mounted read-only; the built arms and runs are written.
VOLUME ["/app/Dataset", "/app/data", "/app/runs"]

CMD ["uv", "run", "thermaldet", "--help"]
