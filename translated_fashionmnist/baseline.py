"""Run the four A/B evaluation settings using two trained ViT models."""

from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path

import torch

from .training import (
    add_training_arguments,
    create_test_loader,
    evaluate,
    run_training,
)
from .common import save_json, write_csv


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, include_train_mode=False)
    parser.set_defaults(output_dir="outputs/four_settings")
    return parser


def plot_summary(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [
        f"{row['setting']}\n{row['train_mode']} -> {row['test_mode']}"
        for row in rows
    ]
    values = [100 * float(row["test_accuracy"]) for row in rows]
    colors = ["#2878B5", "#C82423", "#5B8E3E", "#8E5EA2"]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(labels, values, color=colors)
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title("Position-controlled FashionMNIST: four settings")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = create_parser().parse_args()
    root_output = Path(args.output_dir)
    root_output.mkdir(parents=True, exist_ok=True)
    measured: dict[tuple[str, str], dict[str, object]] = {}

    # Train exactly two models. Each model is evaluated on both official test
    # distributions, producing the four requested settings without duplicate training.
    for train_mode in ("A", "B"):
        train_args = copy.deepcopy(args)
        train_args.train_mode = train_mode
        train_args.val_mode = train_mode
        train_args.output_dir = str(root_output / f"train_{train_mode}")
        model, device, criterion, training_result = run_training(train_args)

        for test_mode in ("A", "B"):
            test_loader = create_test_loader(train_args, test_mode)
            metrics = evaluate(model, test_loader, criterion, device)
            measured[(train_mode, test_mode)] = {
                "train_mode": train_mode,
                "test_mode": test_mode,
                "test_loss": metrics["loss"],
                "test_accuracy": metrics["accuracy"],
                "best_val_accuracy": training_result["best_val_accuracy"],
                "best_epoch": training_result["best_epoch"],
                "checkpoint": training_result["checkpoint"],
            }
            print(
                f"Measured train {train_mode} -> test {test_mode} | "
                f"loss {metrics['loss']:.4f}, accuracy {100 * metrics['accuracy']:.2f}%"
            )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Keep the setting numbers identical to the assignment handout.
    ordered_settings = [
        (1, "A", "A"),
        (2, "B", "B"),
        (3, "A", "B"),
        (4, "B", "A"),
    ]
    rows: list[dict[str, object]] = []
    for setting, train_mode, test_mode in ordered_settings:
        rows.append(
            {
                "setting": setting,
                **measured[(train_mode, test_mode)],
            }
        )

    write_csv(rows, root_output / "summary.csv")
    save_json(rows, root_output / "summary.json")
    plot_summary(rows, root_output / "summary.png")
    print(f"Results saved to {root_output.resolve()}")


if __name__ == "__main__":
    main()
