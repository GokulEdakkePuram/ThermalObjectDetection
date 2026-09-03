"""Validation and cross-run comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import resolve_device
from .paths import REPORTS_DIR, RUNS_DIR, configure_ultralytics

DEFAULT_DATA = "configs/data/flir_agc.yaml"


def run_config(weights: str) -> dict[str, Any]:
    """Read back the resolved config a checkpoint was trained under."""
    path = Path(weights).parent.parent / "thermaldet_config.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def training_data(weights: str) -> str | None:
    """Which preprocessing arm a checkpoint was trained on.

    Evaluating a checkpoint against a different arm than it trained on is a
    domain-shift experiment, not a validation -- an interesting one, but not
    what someone typing `thermaldet eval <weights>` is asking for. So the
    default follows the run rather than a constant.
    """
    return run_config(weights).get("data")


def evaluate(
    weights: str,
    data: str | None = None,
    imgsz: int = 640,
    split: str = "val",
    device: str = "auto",
    batch: int = 16,
) -> dict[str, Any]:
    """Run validation for one checkpoint and return a flat metrics dict."""
    from ultralytics import YOLO

    configure_ultralytics()
    data = data or training_data(weights) or DEFAULT_DATA

    model = YOLO(weights)
    metrics = model.val(
        data=data,
        imgsz=imgsz,
        split=split,
        device=resolve_device(device),
        batch=batch,
        project=str(RUNS_DIR / "val"),
    )

    results = {k: float(v) for k, v in metrics.results_dict.items() if isinstance(v, (int, float))}
    per_class = {
        model.names[int(cid)]: float(ap)
        for cid, ap in zip(metrics.ap_class_index, metrics.box.ap50, strict=False)
    }
    return {
        "weights": str(weights),
        "run": Path(weights).parent.parent.name,
        "arm": Path(data).stem.replace("flir_", ""),
        "data": data,
        "imgsz": imgsz,
        "split": split,
        "metrics": results,
        "ap50_per_class": per_class,
    }


def write_comparison(results: list[dict[str, Any]], out_dir: Path | None = None) -> Path:
    """Render a markdown comparison table across evaluated checkpoints."""
    out_dir = out_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Results",
        "",
        "| run | arm | split | mAP50-95 | mAP50 | precision | recall |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        m = r["metrics"]
        lines.append(
            f"| {r['run']} | {r['arm']} | {r['split']} "
            f"| {m.get('metrics/mAP50-95(B)', 0):.4f} "
            f"| {m.get('metrics/mAP50(B)', 0):.4f} "
            f"| {m.get('metrics/precision(B)', 0):.4f} "
            f"| {m.get('metrics/recall(B)', 0):.4f} |"
        )

    # Per-class AP50 next to the mean, always. mAP weights `bicycle` (3,087
    # boxes) exactly as heavily as `car` (36,209), so a moved headline number
    # can come entirely from the rare class getting slightly luckier.
    classes = sorted({name for r in results for name in r["ap50_per_class"]})
    if classes:
        lines += [
            "",
            "## AP50 per class",
            "",
            "| run | " + " | ".join(classes) + " |",
            "| --- | " + " | ".join("---:" for _ in classes) + " |",
        ]
        for r in results:
            cells = " | ".join(f"{r['ap50_per_class'].get(c, 0):.4f}" for c in classes)
            lines.append(f"| {r['run']} | {cells} |")

    report = out_dir / "results.md"
    report.write_text("\n".join(lines) + "\n")
    return report
