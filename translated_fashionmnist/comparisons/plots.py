"""Generate the three core comparison figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from .configs import EXPERIMENTS, GROUPS


SETTING_LABELS = {
    1: "A -> A",
    2: "B -> B",
    3: "A -> B",
    4: "B -> A",
}
PALETTE = ("#356D9E", "#D8792B", "#4B9366", "#8A63A8")


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "axes.edgecolor": "#46515E",
            "axes.labelcolor": "#27313D",
            "axes.titlecolor": "#172231",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.color": "#27313D",
            "ytick.color": "#27313D",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_results(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["setting"] = int(row["setting"])
        row["test_accuracy"] = float(row["test_accuracy"])
    return rows


def _accuracy(rows: list[dict], config_id: str, setting: int) -> float:
    for row in rows:
        if row["config_id"] == config_id and row["setting"] == setting:
            return 100 * row["test_accuracy"]
    raise KeyError(f"Missing result for {config_id}, setting {setting}")


def _plot_group(
    rows: list[dict],
    config_ids: tuple[str, ...],
    title: str,
    output: Path,
) -> None:
    settings = (1, 2, 3, 4)
    x = list(range(len(settings)))
    width = 0.8 / len(config_ids)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for index, config_id in enumerate(config_ids):
        offset = (index - (len(config_ids) - 1) / 2) * width
        values = [_accuracy(rows, config_id, setting) for setting in settings]
        bars = axis.bar(
            [position + offset for position in x],
            values,
            width=width,
            label=EXPERIMENTS[config_id].display_name,
            color=PALETTE[index],
        )
        axis.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    axis.set_xticks(x, [SETTING_LABELS[setting] for setting in settings])
    axis.set_ylabel("Test accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def generate_visualizations(
    results_csv: str | Path,
    figures_dir: str | Path,
) -> None:
    """Generate one figure for each formal comparison group."""
    _configure_style()
    rows = read_results(results_csv)
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    groups = (
        ("model", "Architecture comparison", "model_comparison.png"),
        ("patch_size", "ViT patch-size comparison", "patch_size_comparison.png"),
        (
            "patch_embedding",
            "Patch-embedding comparison",
            "patch_embedding_comparison.png",
        ),
    )
    for group_name, title, filename in groups:
        config_ids = GROUPS[group_name]
        if all(any(row["config_id"] == item for row in rows) for item in config_ids):
            _plot_group(rows, config_ids, title, figures / filename)
