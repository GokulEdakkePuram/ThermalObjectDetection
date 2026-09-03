"""Hardware detection and profile selection.

An experiment config describes *what* to train; a hardware profile describes
*what the machine can hold*. Keeping them apart is what lets the same
`pretrained` config run on a laptop and on a rented 4090 without editing the
experiment -- and it is what makes a constant batch size across an ablation
possible, which removes the biggest confound from the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

# Profile names map to configs/profiles/<name>.yaml.
CUDA_LARGE = "cuda48"
CUDA_MEDIUM = "cuda24"
CUDA_SMALL = "cuda12"
APPLE = "mps"
CPU = "cpu"


@dataclass(frozen=True)
class Hardware:
    """What we detected about the machine we are running on."""

    device: str
    name: str
    memory_gb: float

    def describe(self) -> str:
        return f"{self.name} ({self.memory_gb:.0f} GB, device={self.device})"


def detect() -> Hardware:
    """Identify the accelerator and how much memory it has.

    On Apple Silicon the GPU shares system RAM, so the reported figure is total
    unified memory -- of which a training run can realistically use maybe half.
    The MPS profile accounts for that rather than pretending 16 GB is usable.
    """
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return Hardware("0", props.name, props.total_memory / 1024**3)

    import psutil

    total_gb = psutil.virtual_memory().total / 1024**3
    if torch.backends.mps.is_available():
        return Hardware("mps", "Apple Silicon (unified memory)", total_gb)
    return Hardware("cpu", "CPU", total_gb)


def auto_profile(hw: Hardware | None = None) -> str:
    """Pick the profile that matches the detected hardware.

    Thresholds are deliberately conservative: a profile that OOMs two hours
    into a rented run costs more than one that leaves some memory unused.
    """
    hw = hw or detect()

    if hw.device == "cpu":
        return CPU
    if hw.device == "mps":
        return APPLE
    if hw.memory_gb >= 40:
        return CUDA_LARGE
    if hw.memory_gb >= 20:
        return CUDA_MEDIUM
    return CUDA_SMALL
