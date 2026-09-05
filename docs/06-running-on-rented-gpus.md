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
batch under one profile. The payoff is that all seven runs share `batch: 32` on
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
uv run thermaldet probe pretrained rgb
```

Two short runs per config at different data fractions, then solve

```
epoch_seconds = overhead + rate * n_images
```

for both terms. **Do not time one short run and divide by the fraction.**
`overhead` — dataloader spin-up, cuDNN autotuning, epoch teardown — does not
shrink with the data, so dividing by `fraction` scales it up along with
everything else. At `fraction=0.1` the estimate carries ten copies of it.

That error bites less on v2 than it would on a small dataset -- 10,742 frames
at 640x512 still make fairly short epochs -- but it is the kind of error that
ranks two configs in the wrong order, which is worse than being uniformly
optimistic. The
[test suite](../tests/test_profiles.py) pins the arithmetic.

`probe` reports **reserved** GPU memory, not allocated. PyTorch's caching
allocator holds freed blocks, and reserved is the number that actually runs a
card out of memory.

## Expect this to be dataloader-bound

640x512 greyscale frames are small, and mosaic augmentation composites four of
them per sample. That work happens on the CPU. On a 4090 the GPU is likely to
spend a good fraction of each step waiting for it, which has two consequences
worth knowing before reading any timing:

- **`cache: ram` matters more than the card does.** 10,742 640x512 greyscale
  frames are a few GB decoded; keeping them in RAM removes JPEG decode from
  the loop entirely. It is set in every CUDA profile for that reason. The
  `rgb` arm is the exception worth watching -- its source frames are up to
  2048x1536, so it caches larger and decodes slower.
- **Renting a bigger card may buy nothing.** If the bottleneck is CPU-side
  augmentation, a 48 GB A6000 finishes at nearly the same wall clock as a 4090
  and costs more per hour. Check the vCPU count on the instance before paying
  for the GPU tier above it.

Whether this actually holds here is a measurement, not an assumption — `probe`
across two model sizes would show it, and if per-image time barely moves with
model size, the pipeline is input-bound.

## Choosing a box

- **CUDA 13, specifically.** `uv.lock` pins torch 2.11 against a CUDA 13
  runtime — `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `cuda-toolkit` 13.x — so
  the host needs a driver supporting CUDA 13.0 or newer, roughly **580+**. A
  12.x driver fails only once torch is first imported, which is five paid
  minutes and 5 GB after `uv sync` started. Most vast.ai listings are still on
  12.x, so **filter for it before renting**; `setup_remote.sh` checks the
  driver before installing anything, but by then you have already paid for the
  instance.
- **Disk: ask for 75 GB.** The zip is ~13 GB and has to coexist with its
  unpacked copy during extraction, the four arms add ~6 GB, and the CUDA 13
  venv is ~12 GB. Peak is around 45 GB before a single checkpoint is written.
  vast.ai defaults the disk slider to 10 GB, which is the single most common
  way an instance gets wasted.
- **vCPU count matters more than you would think.** This pipeline is
  dataloader-bound (see above), and mosaic composites four frames per sample.
  A 4090 behind 4 vCPUs will idle. Prefer listings with 8+ vCPUs; it is
  usually cheaper than moving up a GPU tier for the same wall clock.

## Getting the dataset onto the box

FLIR is behind a registration form, so the dataset cannot be fetched from
source on an unattended machine. Keep your own copy of the zip in Google Drive
and pull it from there:

```bash
GDRIVE_ID=<file id> ./scripts/fetch_dataset.sh
# or paste the share URL:
./scripts/fetch_dataset.sh 'https://drive.google.com/file/d/<id>/view?usp=sharing'
```

The file must be shared as **"Anyone with the link"**. Three things this
routinely goes wrong on, all handled:

- **A file that still requires sign-in** returns an HTML login page, and the
  downloader saves it under the `.zip` name without complaint. The script
  checks the size and prints the first few hundred bytes rather than letting
  `unzip` fail confusingly 20 minutes later.
- **Drive's public-download quota.** "Too many users have viewed or downloaded
  this file recently" is a per-file counter that clears in about a day. The
  reliable workaround is to copy the file inside your own Drive (right-click →
  Make a copy) and use the new id, which starts a fresh counter.
- **The archive's shape.** Some exports put the six split directories at the
  root, some nest them under a folder. The script finds whichever level holds
  `images_thermal_train/` and moves that.

It verifies what it extracted — all six `coco.json` files plus
`analyticsData/`, without which the 16-bit arms cannot be built — and deletes
the zip afterwards to get 13 GB back. `KEEP_ZIP=1` to keep it.

`setup_remote.sh` calls this automatically when `GDRIVE_ID` is set in the
environment.

If Drive is being difficult, `scp` is the fallback:

```bash
scp -P <port> flir_adas_v2.zip root@<host>:~/thermaldet/Dataset/
```

## The vast.ai run, start to finish

**1. Rent.** Filter on, in order of how expensive the mistake is:

| filter | value | why |
| --- | --- | --- |
| CUDA version | **≥ 13.0** | the lockfile's torch needs it; a 12.x box is wasted |
| disk | **75 GB** | the slider defaults to 10; peak need is ~45 GB |
| vCPUs | **8+** | the pipeline is dataloader-bound, not GPU-bound |
| GPU | RTX 4090 / 3090 | 24 GB, the `cuda24` profile's tier |

A 4090 with 4 vCPUs will finish no faster than a 3090 with 12, and costs more.

**2. Provision.** One command, which checks the driver, the disk and the GPU
*before* installing 5 GB of wheels:

```bash
export GDRIVE_ID=<your file id>
curl -fsSL https://raw.githubusercontent.com/GokulEdakkePuram/ThermalObjectDetection/main/scripts/setup_remote.sh | bash
cd ~/thermaldet
```

Run `df -h` first. Some templates mount the rented volume away from `$HOME` —
`/workspace` is the usual one — and the container's own filesystem can be only
a few GB. If that is the case here, set `WORKDIR=/workspace/thermaldet` before
the curl; the disk check measures whichever filesystem `WORKDIR` will live on.

**3. Build the arms and calibrate.**

```bash
make arms                                  # ~2 min, ~6 GB
uv run thermaldet probe pretrained rgb     # ~5 min, tells you what the sweep costs
```

Read the probe output before starting the sweep. If the projected total is
wildly more than you budgeted, that is the moment to change something — not
three runs in.

**4. Train.** Inside tmux, so a dropped SSH connection does not kill seven
runs:

```bash
export WANDB_API_KEY=<key from wandb.ai/authorize>
tmux new -s train
./scripts/run_sweep.sh
```

Detach with `ctrl-b d`, reattach with `tmux attach -t train`. The sweep logs
each run separately under `logs/`, scores every checkpoint that finished on
the held-out test split, and prints a success/failure summary at the end.

**5. Get the results off before destroying anything.**

```bash
# from your laptop
scp -P <port> -r root@<host>:~/thermaldet/reports .
scp -P <port> -r root@<host>:~/thermaldet/runs/train .
```

`reports/` is small and holds the tables. `runs/train/` is a few hundred MB
with the checkpoints and each run's resolved config. If you tracked with W&B
the metrics are already safe, but the weights are not.

**6. Destroy the instance.** A stopped instance still bills for its disk.

## Provisioning, in one command

```bash
GDRIVE_ID=<file id> bash <(curl -fsSL https://raw.githubusercontent.com/GokulEdakkePuram/ThermalObjectDetection/main/scripts/setup_remote.sh)
```

Checks the GPU, the driver's CUDA version and the disk *before* installing
anything, then clones, syncs from `uv.lock` so the environment matches the one
the results came from, fetches the dataset if `GDRIVE_ID` is set, and confirms
torch can actually see the card.

`run_sweep.sh` deliberately does not `set -e`. A config that dies must not
cancel the six after it — the entire point of running unattended is to come
back to as much finished work as possible.

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
a destroyed one takes the only copy of seven runs with it. W&B keeps the
metrics but not the weights.
