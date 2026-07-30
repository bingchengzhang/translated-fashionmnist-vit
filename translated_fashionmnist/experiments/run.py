"""Run the controlled study and write a complete, self-contained record."""

from __future__ import annotations

import argparse
import csv
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torchvision

from ..utils import save_json, write_csv
from .config import GROUPS, configuration_ids_for_groups
from .plots import generate_visualizations
from .protocol import ProtocolConfig, run_configuration


HISTORY_FIELDS = (
    "epoch",
    "learning_rate",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["all", *GROUPS],
        default=["all"],
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-dir",
        default="outputs/comparison",
        help="Run directory. The committed results/ record is never overwritten by default.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-train-samples", type=int, default=0)
    parser.add_argument("--limit-val-samples", type=int, default=0)
    parser.add_argument("--limit-test-samples", type=int, default=0)
    return parser


def consolidate_training_history(
    output_dir: str | Path,
    configuration_ids: list[str],
    epochs: int,
) -> list[dict[str, object]]:
    """Merge per-fit histories into the record consumed by the report."""
    root = Path(output_dir)
    rows: list[dict[str, object]] = []
    for config_id in configuration_ids:
        for train_mode in ("A", "B"):
            history_path = (
                root / "runs" / config_id / f"train_{train_mode}" / "history.csv"
            )
            if not history_path.is_file():
                raise FileNotFoundError(f"Missing training history: {history_path}")
            with history_path.open(newline="", encoding="utf-8") as handle:
                records = list(csv.DictReader(handle))
            if not records or not set(HISTORY_FIELDS).issubset(records[0]):
                raise ValueError(f"Incomplete training history: {history_path}")
            observed_epochs = [int(record["epoch"]) for record in records]
            if observed_epochs != list(range(1, epochs + 1)):
                raise ValueError(
                    f"Expected epochs 1..{epochs} in {history_path}, got {observed_epochs}."
                )
            rows.extend(
                {
                    "config_id": config_id,
                    "train_mode": train_mode,
                    **{field: record[field] for field in HISTORY_FIELDS},
                }
                for record in records
            )
    write_csv(rows, root / "training_history.csv")
    return rows


def main() -> None:
    args = create_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = ProtocolConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
        deterministic=args.deterministic,
        download=args.download,
        limit_train_samples=args.limit_train_samples,
        limit_val_samples=args.limit_val_samples,
        limit_test_samples=args.limit_test_samples,
    )
    selected = configuration_ids_for_groups(args.groups)
    all_rows: list[dict] = []
    for config_id in selected:
        all_rows.extend(
            run_configuration(
                config_id,
                protocol,
                resume=args.resume,
            )
        )

    all_rows.sort(key=lambda row: (selected.index(row["config_id"]), row["setting"]))
    results_csv = output_dir / "metrics.csv"
    write_csv(all_rows, results_csv)
    history_rows = consolidate_training_history(
        output_dir,
        selected,
        protocol.epochs,
    )
    manifest = {
        "groups": args.groups,
        "configurations": selected,
        "fit_count": len(history_rows) // protocol.epochs,
        "evaluation_count": len(all_rows),
        "protocol": asdict(protocol),
        "method": {
            "loss": "cross-entropy",
            "optimizer": "AdamW",
            "scheduler": "cosine annealing",
            "checkpoint": "highest validation accuracy",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
            "determinism": (
                "deterministic algorithms enabled"
                if protocol.deterministic
                else "seeded; deterministic algorithms disabled"
            ),
        },
    }
    save_json(manifest, output_dir / "manifest.json")

    generate_visualizations(
        results_csv=results_csv,
        figures_dir=output_dir / "figures",
    )
    print(f"Results: {results_csv.resolve()}")
    print(f"Training history: {(output_dir / 'training_history.csv').resolve()}")
    print(f"Figures: {(output_dir / 'figures').resolve()}")


if __name__ == "__main__":
    main()
