# Running on rented GPUs

Development happens on an M2 Pro; training happens on a rented 4090. The
awkward part of that split is not the SSH — it is that the two machines cannot
hold the same batch size, and batch size changes results.

## The confound profiles fix

On a laptop, batch has to shrink whenever anything else grows. Once it does,
two runs labelled `pretrained` and `frozen_stem` differ in the freeze depth
*and* in the effective gradient noise, and the ablation is no longer an
ablation.

So an experiment config says **what to train** and a hardware profile says
**what the machine can hold**:

```
configs/pretrained.yaml     model, epochs, imgsz, augmentation, seed
configs/profiles/cuda24.yaml   batch, workers, device, amp, cache
```

`load_profile` rejects any profile that tries to set `epochs` or `imgsz`, and
[a test](../tests/test_profiles.py) asserts every arm resolves to the same
batch under one profile. The payoff is that all six runs share `batch: 32` on
one card, so the only thing that differs between them is the line their config
changed.

The profile is recorded on the resolved config written next to each run's
weights, so a row in a results table can always be traced back to the machine
that produced it. Runs from different profiles should not be compared.

| profile | batch | for |
| --- | ---: | --- |
| `cpu` | 2 | CI, and proving the pipeline runs anywhere |
| `mps` | 8 | Apple Silicon; unified memory, so the GPU competes with the OS |
| `cuda12` | 16 | 12-16 GB: 3060, 4070, T4 |
| `cuda24` | 32 | 24 GB: 3090, 4090, A5000 — the common rental tier |
| `cuda48` | 48 | 48 GB+: A6000, L40S, A100 |

`--profile auto` detects the card and picks one. Thresholds are conservative:
a profile that OOMs two hours into a rented run costs more than one that
leaves memory unused.

## Measure before paying

```bash
uv run thermaldet probe pretrained global_map
```

Two short runs per config at different data fractions, then solve

```
epoch_seconds = overhead + rate * n_images
```

for both terms. **Do not time one short run and divide by the fraction.**
`overhead` — dataloader spin-up, cuDNN autotuning, epoch teardown — does not
shrink with the data, so dividing by `fraction` scales it up along with
everything else. At `fraction=0.1` the estimate carries ten copies of it.

That error bites harder here than on a large dataset. 7,543 frames at 640x512
make short epochs, so fixed overhead is a large share of each one, and a naive
estimate can rank two configs in the wrong order. The
[test suite](../tests/test_profiles.py) pins the arithmetic.

`probe` reports **reserved** GPU memory, not allocated. PyTorch's caching
allocator holds freed blocks, and reserved is the number that actually runs a
card out of memory.

## Expect this to be dataloader-bound

640x512 greyscale frames are small, and mosaic augmentation composites four of
them per sample. That work happens on the CPU. On a 4090 the GPU is likely to
spend a good fraction of each step waiting for it, which has two consequences
worth knowing before reading any timing:

- **`cache: ram` matters more than the card does.** The whole dataset is a few
  GB decoded; keeping it in RAM removes JPEG decode from the loop entirely.
  It is set in every CUDA profile for that reason.
- **Renting a bigger card may buy nothing.** If the bottleneck is CPU-side
  augmentation, a 48 GB A6000 finishes at nearly the same wall clock as a 4090
  and costs more per hour. Check the vCPU count on the instance before paying
  for the GPU tier above it.

Whether this actually holds here is a measurement, not an assumption — `probe`
across two model sizes would show it, and if per-image time barely moves with
model size, the pipeline is input-bound.

## Choosing a box

- **CUDA version.** The pinned torch wheels bundle their own CUDA runtime and
  need a host driver new enough for it. An old driver fails only once torch is
  imported — five paid minutes and 5 GB after `uv sync` started.
  `setup_remote.sh` checks the driver first, before installing anything.
- **Disk.** The FLIR download is ~18 GB, the three preprocessing arms add ~6
  GB, and the CUDA venv is ~8 GB. Rentals commonly default to 10 GB total,
  which is the single most common way an instance gets wasted.
- **The dataset cannot be scripted.** FLIR is behind a registration form, so
  it has to be moved onto the box by hand (`scp`, or an rclone remote). Plan
  for that; on a slow uplink it is the longest step in the whole process.

## Provisioning

```bash
curl -fsSL https://raw.githubusercontent.com/GokulEdakkePuram/ThermalObjectDetection/main/scripts/setup_remote.sh | bash
```

Checks the GPU, the driver's CUDA version and the disk *before* installing
anything, then clones and syncs from `uv.lock` so the environment matches the
one the results came from.

Then, inside tmux so a dropped connection does not kill six runs:

```bash
make arms ADOPT="--adopt-labels Dataset/FLIR_ADAS_1_3/yolo/labels"
uv run thermaldet probe pretrained global_map
tmux new -s train
./scripts/run_sweep.sh
```

`run_sweep.sh` deliberately does not `set -e`. A config that dies must not
cancel the five after it — the entire point of running unattended is to come
back to as much finished work as possible. It logs each run separately and
prints a success/failure summary at the end.

## Tracking

```bash
export WANDB_API_KEY=<key from wandb.ai/authorize>
uv run thermaldet train pretrained --track wandb
```

Export the key rather than logging in interactively on a box you are going to
destroy. Note that every CLI here lives in the project venv, so the
interactive form is `uv run wandb login` — a bare `wandb` is not on `PATH`.

A tracker that is requested but not installed raises at startup rather than
being skipped. Finishing an hour of paid training and discovering nothing was
logged is the specific failure that guards against.

## Before destroying the instance

Pull `runs/` and `reports/`. A stopped instance still bills for its disk, and
a destroyed one takes the only copy of six runs with it.
