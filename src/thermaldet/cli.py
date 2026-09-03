"""Command line interface.

Every stage of the project is reachable as ``thermaldet <verb>``, so the
commands in the README are the same ones that produced the numbers in it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .paths import DATASET_DIR, REPORTS_DIR


def _raw_dir(root: Path) -> Path | None:
    """The 16-bit training frames, if this release ships them."""
    candidate = root / "train" / "thermal_16_bit"
    return candidate if candidate.is_dir() else None


def _cmd_convert(args: argparse.Namespace) -> int:
    from .convert import convert
    from .radiometry import GlobalLinear, PercentileStretch, measure_window

    render = None
    if args.arm == "p1p99":
        render = PercentileStretch()
    elif args.arm == "global":
        raw = _raw_dir(args.flir_root)
        if raw is None:
            print(f"No 16-bit frames under {args.flir_root}.", file=sys.stderr)
            return 1
        lo, hi = measure_window(raw)
        print(f"[window ] global map fixed at [{lo:.0f}, {hi:.0f}] counts")
        render = GlobalLinear(lo, hi)

    convert(
        args.flir_root,
        arm=args.arm,
        classes=args.classes,
        adopt_from=args.adopt_labels,
        render=render,
        copy=args.copy,
    )
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    from ultralytics.data.utils import check_det_dataset

    from .paths import configure_ultralytics
    from .radiometry import frame_spans
    from .stats import profile_dataset, write_report

    configure_ultralytics()
    stats = profile_dataset(args.data)
    if not stats:
        print("No splits found. Run 'thermaldet convert' first.", file=sys.stderr)
        return 1

    raw = _raw_dir(args.flir_root)
    names = check_det_dataset(args.data)["names"]
    report = write_report(stats, names, REPORTS_DIR, frame_spans(raw) if raw else None)

    for split, s in stats.items():
        print(
            f"{split:>5}: {s.n_images:,} frames, {s.n_boxes:,} boxes, "
            f"{100 * s.small_fraction:.1f}% small"
        )
    print(f"\nWrote {report}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import train

    print(json.dumps(train(args.config, profile=args.profile, tracker=args.track), indent=2))
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    from .probe import probe_all

    probe_all(
        args.configs, profile=args.profile, epochs=args.epochs, fractions=tuple(args.fractions)
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import evaluate, write_comparison

    results = [
        evaluate(w, data=args.data, imgsz=args.imgsz, split=args.split, device=args.device)
        for w in args.weights
    ]
    print(write_comparison(results).read_text())
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from .predict import predict

    out = predict(
        args.weights,
        args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=not args.no_save,
    )
    print(f"{out['images']} frame(s), {out['detections']} detection(s)")
    if out["save_dir"]:
        print(f"annotated output -> {out['save_dir']}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from .export import export

    print(f"Exported to {export(args.weights, fmt=args.format, imgsz=args.imgsz)}")
    return 0


def _cmd_train_manual(args: argparse.Namespace) -> int:
    from .manual import main as manual_main

    return manual_main(args.args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thermaldet", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="build one preprocessing arm from the FLIR download")
    p.add_argument(
        "--arm",
        default="agc",
        choices=["agc", "global", "p1p99"],
        help="agc: FLIR's 8-bit JPEGs; global/p1p99: rendered from the 16-bit TIFFs",
    )
    p.add_argument("--flir-root", type=Path, default=DATASET_DIR / "FLIR_ADAS_1_3")
    p.add_argument("--classes", nargs="+", default=None, help="class names, in YOLO index order")
    p.add_argument(
        "--adopt-labels",
        type=Path,
        default=None,
        help="directory of existing YOLO labels (<dir>/train, <dir>/val), for a "
        "download whose annotation JSONs are missing",
    )
    p.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    p.set_defaults(func=_cmd_convert)

    p = sub.add_parser("profile", help="measure class balance, object scale and dynamic range")
    p.add_argument("--data", default="configs/data/flir_agc.yaml")
    p.add_argument("--flir-root", type=Path, default=DATASET_DIR / "FLIR_ADAS_1_3")
    p.set_defaults(func=_cmd_profile)

    p = sub.add_parser("train", help="fine-tune a model from a config")
    p.add_argument("config", help="config name (e.g. 'pretrained') or path")
    p.add_argument(
        "--profile",
        default="auto",
        help="hardware profile: auto (default), mps, cpu, cuda12, cuda24, cuda48",
    )
    p.add_argument(
        "--track",
        default=None,
        choices=["wandb", "mlflow", "tensorboard", "none"],
        help="experiment tracker (default: whatever the config says)",
    )
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser(
        "train-manual",
        help="train through the explicit loop (per-depth learning rates)",
        description="Everything after the verb is passed through; see "
        "`thermaldet train-manual --help`.",
    )
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(func=_cmd_train_manual)

    p = sub.add_parser("probe", help="time a short run and project the full one")
    p.add_argument("configs", nargs="+")
    p.add_argument("--profile", default="auto")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument(
        "--fractions",
        type=float,
        nargs=2,
        default=[0.1, 0.3],
        metavar=("SMALL", "LARGE"),
        help="two data fractions; the gap between them separates fixed "
        "per-epoch overhead from the per-image rate",
    )
    p.set_defaults(func=_cmd_probe)

    p = sub.add_parser("eval", help="validate one or more checkpoints")
    p.add_argument("weights", nargs="+")
    p.add_argument("--data", default=None, help="default: the arm each checkpoint was trained on")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("predict", help="run a checkpoint over frames and save the results")
    p.add_argument("weights")
    p.add_argument("source", help="image, directory, or glob; raw .tiff is rendered first")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-save", action="store_true")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("export", help="export a checkpoint for deployment")
    p.add_argument("weights")
    p.add_argument("--format", default="onnx")
    p.add_argument("--imgsz", type=int, default=640)
    p.set_defaults(func=_cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
