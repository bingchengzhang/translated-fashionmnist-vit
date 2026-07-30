"""Run the three optional comparison groups and generate report figures."""

from __future__ import annotations

import argparse
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from utils import save_json, write_csv

from .configs import configuration_ids_for_groups
from .protocol import ProtocolConfig, run_configuration
from .plot_results import generate_visualizations


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["all", "model", "patch_size", "patch_embedding"],
        default=["all"],
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-dir",
        default="results/comparisons",
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
    manifest = {
        "author": "bc",
        "groups": args.groups,
        "configurations": selected,
        "protocol": asdict(protocol),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    save_json(manifest, output_dir / "manifest.json")

    package_dir = Path(__file__).resolve().parent
    generate_visualizations(
        results_csv=results_csv,
        baseline_csv=package_dir / "references" / "teammate_vit.csv",
        assets_dir=output_dir / "figures",
        summary_markdown=output_dir / "summary.md",
    )
    print(f"Results: {results_csv.resolve()}")
    print(f"Figures: {(output_dir / 'figures').resolve()}")


if __name__ == "__main__":
    main()
