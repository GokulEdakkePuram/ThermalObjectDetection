"""
Manual fine-tuning loop for Ultralytics YOLO (v8/v11 DetectionModel) — ADVANCED.

This is an alternative to the standard `src/train.py`. It replaces the stock
Trainer with an explicit loop so custom logic (per-group LRs, discriminative
fine-tuning, extra logging) can be injected. Prefer `src/train.py` unless you
need that control — note `validate()` here reports mean val loss, not mAP
(swap in DetectionValidator for real mAP).

Run `python src/coco_to_yolo.py` first to build the dataset + data yaml.

Usage:
    python src/train_manual.py \\
        --weights yolov8n.pt \\
        --data configs/flir_thermal.yaml \\
        --epochs 50 --imgsz 640 --batch 16 \\
        --freeze 10 --optimizer AdamW --lr0 1e-3 --ema \\
        --device mps        # or cuda, or cpu
"""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.build import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import ModelEMA


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data / model
    p.add_argument("--weights", type=str, default="yolov8n.pt")
    p.add_argument("--data", type=str, default="configs/flir_thermal.yaml", help="path to data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)

    # optimization
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--optimizer", choices=["SGD", "AdamW"], default="AdamW")
    p.add_argument("--lr0", type=float, default=1e-3, help="initial LR")
    p.add_argument("--lrf", type=float, default=0.01, help="final LR = lr0 * lrf")
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--warmup-momentum", type=float, default=0.8)
    p.add_argument("--warmup-bias-lr", type=float, default=0.1)

    # regularization / stability
    p.add_argument("--freeze", type=int, default=0, help="freeze first N top-level layers (backbone is 0..9 on v8)")
    p.add_argument("--accumulate", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--grad-clip", type=float, default=10.0, help="max grad norm (0 disables)")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA or MPS; no-op on CPU)")
    p.add_argument("--ema", action="store_true")

    # loss gains
    p.add_argument("--box", type=float, default=7.5)
    p.add_argument("--cls", type=float, default=0.5)
    p.add_argument("--dfl", type=float, default=1.5)

    # io
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    p.add_argument("--project", type=str, default="runs/manual")
    p.add_argument("--name", type=str, default="exp")
    p.add_argument("--resume", type=str, default=None, help="path to checkpoint")
    p.add_argument("--save-period", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_layers(model: nn.Module, freeze_n: int) -> None:
    """Freeze params whose top-level layer index (model.<idx>.*) < freeze_n."""
    if freeze_n <= 0:
        return
    frozen = 0
    for name, param in model.named_parameters():
        parts = name.split(".")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        if int(parts[1]) < freeze_n:
            param.requires_grad_(False)
            frozen += 1
    LOGGER.info(f"Froze {frozen} parameter tensors in first {freeze_n} layers")


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Three groups: conv/linear weights (wd), biases (no wd), norm weights (no wd).

    Enumerates from named_parameters() to guarantee full coverage; membership
    in the 'norm' group is decided by pointer identity against each norm
    module's direct parameters.
    """
    g_weights, g_biases, g_norm = [], [], []
    norm_types = (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)

    norm_ids = set()
    for m in model.modules():
        if isinstance(m, norm_types):
            norm_ids.update(id(p) for p in m.parameters(recurse=False))

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in norm_ids:
            g_norm.append(p)
        elif name.endswith(".bias"):
            g_biases.append(p)
        else:
            g_weights.append(p)

    LOGGER.info(f"Param groups: weights={len(g_weights)}, biases={len(g_biases)}, norm={len(g_norm)}")
    if not (g_weights or g_biases or g_norm):
        raise RuntimeError("No trainable parameters — check --freeze value")
    return [
        {"params": g_weights, "weight_decay": weight_decay},
        {"params": g_biases, "weight_decay": 0.0},
        {"params": g_norm, "weight_decay": 0.0},
    ]


def build_optimizer(name: str, groups: list[dict], lr: float, momentum: float) -> torch.optim.Optimizer:
    if name == "AdamW":
        return AdamW(groups, lr=lr, betas=(momentum, 0.999))
    if name == "SGD":
        return SGD(groups, lr=lr, momentum=momentum, nesterov=True)
    raise ValueError(name)


def cosine_lr_lambda(lrf: float, epochs: int):
    return lambda epoch: ((1 - math.cos(epoch * math.pi / epochs)) / 2) * (lrf - 1) + 1


def warmup_step(
    optimizer,
    *,
    ni: int,
    nw: int,
    epoch: int,
    warmup_bias_lr: float,
    warmup_momentum: float,
    momentum: float,
    base_lrs: list[float],
    lr_lambda,
) -> None:
    xi = [0, nw]
    for j, pg in enumerate(optimizer.param_groups):
        start_lr = warmup_bias_lr if j == 1 else 0.0
        pg["lr"] = _interp(ni, xi, [start_lr, base_lrs[j] * lr_lambda(epoch)])
        if "momentum" in pg:
            pg["momentum"] = _interp(ni, xi, [warmup_momentum, momentum])


def _interp(x: float, xp: list[float], fp: list[float]) -> float:
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    t = (x - xp[0]) / (xp[-1] - xp[0])
    return fp[0] + t * (fp[-1] - fp[0])


def batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move every tensor in the batch dict to device, normalize image to float."""
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
    batch["img"] = batch["img"].float() / 255.0
    return batch


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_dataloader(cfg, data_dict, split: str, args) -> torch.utils.data.DataLoader:
    img_path = data_dict[split]
    dataset = build_yolo_dataset(
        cfg,
        img_path,
        args.batch,
        data_dict,
        mode="train" if split == "train" else "val",
        rect=(split != "train"),
        stride=32,
    )
    shuffle = split == "train"
    return build_dataloader(dataset, args.batch, args.workers, shuffle=shuffle, rank=-1)


# --------------------------------------------------------------------------- #
# Train / val
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    *,
    epoch: int,
    epochs: int,
    nw: int,
    nb: int,
    device,
    args,
    base_lrs: list[float],
    lr_lambda,
    ema=None,
) -> torch.Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    mloss = torch.zeros(3, device=device)

    for i, batch in enumerate(loader):
        ni = i + nb * epoch

        if ni <= nw:
            warmup_step(
                optimizer,
                ni=ni,
                nw=nw,
                epoch=epoch,
                warmup_bias_lr=args.warmup_bias_lr,
                warmup_momentum=args.warmup_momentum,
                momentum=args.momentum,
                base_lrs=base_lrs,
                lr_lambda=lr_lambda,
            )

        batch = batch_to_device(batch, device)

        with autocast(device_type=device.type, enabled=args.amp):
            loss, loss_items = model.loss(batch, preds=None)
            loss = loss.sum() / args.accumulate

        scaler.scale(loss).backward()

        if (i + 1) % args.accumulate == 0 or (i + 1) == nb:
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        mloss = (mloss * i + loss_items.detach()) / (i + 1)
        if i % 50 == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            LOGGER.info(
                f"Epoch {epoch + 1}/{epochs}  batch {i + 1}/{nb}  "
                f"box={mloss[0]:.4f} cls={mloss[1]:.4f} dfl={mloss[2]:.4f}  lr={cur_lr:.2e}"
            )

    return mloss


@torch.no_grad()
def validate(model, loader, device, args) -> float:
    """Mean total loss on val set. Swap for DetectionValidator for real mAP."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        with autocast(device_type=device.type, enabled=args.amp):
            loss, _ = model.loss(batch, preds=None)
        total += loss.sum().item() * batch["img"].size(0)
        n += batch["img"].size(0)
    return total / max(n, 1)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Saving to {save_dir.resolve()} (device={device})")

    # --- Data spec (resolves relative paths, validates nc/names) ---
    data_dict = check_det_dataset(args.data)

    # --- Model: load pretrained, rehead if nc mismatch ---
    yolo = YOLO(args.weights)
    src_nc = yolo.model.yaml.get("nc", None)
    tgt_nc = data_dict["nc"]
    if src_nc != tgt_nc:
        LOGGER.info(f"Re-heading detection model: {src_nc} -> {tgt_nc} classes")
        new_model = DetectionModel(yolo.model.yaml, ch=3, nc=tgt_nc, verbose=False)
        new_model.load(yolo.model)  # copies matching-shape tensors, re-inits head
        yolo.model = new_model
    model: nn.Module = yolo.model.to(device)
    model.nc = tgt_nc
    model.names = (
        {i: n for i, n in enumerate(data_dict["names"])}
        if isinstance(data_dict["names"], (list, tuple))
        else data_dict["names"]
    )

    # Loss gains live on model.args
    cfg = get_cfg(
        overrides={
            "data": args.data,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "box": args.box,
            "cls": args.cls,
            "dfl": args.dfl,
        }
    )
    model.args = cfg

    freeze_layers(model, args.freeze)

    # --- Data loaders ---
    train_loader = make_dataloader(cfg, data_dict, "train", args)
    val_loader = make_dataloader(cfg, data_dict, "val", args)
    nb = len(train_loader)
    nw = max(round(args.warmup_epochs * nb), 100)
    LOGGER.info(f"Train batches/epoch={nb}, warmup iters={nw}")

    # --- Optimizer + scheduler ---
    groups = build_param_groups(model, args.weight_decay)
    optimizer = build_optimizer(args.optimizer, groups, args.lr0, args.momentum)
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    lr_lambda = cosine_lr_lambda(args.lrf, args.epochs)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # --- AMP + EMA ---
    # GradScaler is only meaningful for fp16 on CUDA; elsewhere it's a no-op.
    scaler = GradScaler(device=device.type, enabled=(args.amp and device.type == "cuda"))
    ema = ModelEMA(model) if args.ema else None

    # --- Resume ---
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # `model`/`ema` are stored as module objects (see checkpoint save below).
        model.load_state_dict(ckpt["model"].float().state_dict())
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        if ema is not None and ckpt.get("ema") is not None:
            ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            ema.updates = ckpt.get("ema_updates", 0)
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        LOGGER.info(f"Resumed from epoch {start_epoch}, best_val={best_val:.4f}")

    # --- Train ---
    for epoch in range(start_epoch, args.epochs):
        mloss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            epoch=epoch,
            epochs=args.epochs,
            nw=nw,
            nb=nb,
            device=device,
            args=args,
            base_lrs=base_lrs,
            lr_lambda=lr_lambda,
            ema=ema,
        )
        scheduler.step()

        val_model = ema.ema if ema is not None else model
        val_loss = validate(val_model, val_loader, device, args)
        LOGGER.info(f"[Epoch {epoch + 1}] train box/cls/dfl={mloss.tolist()}  val_loss={val_loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "best_val": min(best_val, val_loss),
            # Store the module objects (not just state_dicts) so the checkpoint is
            # loadable by ultralytics `YOLO(...)` for val/predict/export, while the
            # optimizer/scheduler/scaler below still make it resumable.
            "model": deepcopy(model).half(),
            "ema": deepcopy(ema.ema).half() if ema else None,
            "ema_updates": ema.updates if ema else 0,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, save_dir / "best.pt")
            LOGGER.info(f"  -> new best (val_loss={val_loss:.4f})")
        if args.save_period > 0 and (epoch + 1) % args.save_period == 0:
            torch.save(ckpt, save_dir / f"epoch{epoch + 1}.pt")

    LOGGER.info(f"Done. best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
