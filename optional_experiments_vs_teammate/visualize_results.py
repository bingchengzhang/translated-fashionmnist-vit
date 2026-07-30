"""Generate report-ready figures and a Markdown result summary."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from .configurations import EXPERIMENTS, GROUPS


SETTING_LABELS = {
    1: "1\nA -> A",
    2: "2\nB -> B",
    3: "3\nA -> B",
    4: "4\nB -> A",
}


def read_results(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    numeric_int = {"setting", "best_epoch", "parameter_count"}
    numeric_float = {
        "test_loss",
        "test_accuracy",
        "best_val_accuracy",
        "train_elapsed_seconds",
    }
    for row in rows:
        for key in numeric_int:
            row[key] = int(row[key])
        for key in numeric_float:
            row[key] = float(row[key])
    return rows


def _accuracy(rows: list[dict], config_id: str, setting: int) -> float:
    for row in rows:
        if row["config_id"] == config_id and row["setting"] == setting:
            return 100 * row["test_accuracy"]
    raise KeyError(f"Missing result for {config_id}, setting {setting}")


def _training_minutes(rows: list[dict], config_id: str) -> float:
    elapsed_by_mode = {
        row["train_mode"]: row["train_elapsed_seconds"]
        for row in rows
        if row["config_id"] == config_id
    }
    return sum(elapsed_by_mode.values()) / 60


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


def _plot_generalization(rows: list[dict], output: Path) -> None:
    config_ids = GROUPS["model"]
    labels = [EXPERIMENTS[config_id].display_name for config_id in config_ids]
    matched = [_accuracy(rows, config_id, 2) for config_id in config_ids]
    shifted = [_accuracy(rows, config_id, 4) for config_id in config_ids]
    drops = [before - after for before, after in zip(matched, shifted)]

    x = list(range(len(config_ids)))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    width = 0.36
    axes[0].bar([value - width / 2 for value in x], matched, width, label="B -> B")
    axes[0].bar([value + width / 2 for value in x], shifted, width, label="B -> A")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("Fixed-position training under distribution shift")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    bars = axes[1].bar(labels, drops, color="#C44E52")
    axes[1].bar_label(bars, fmt="%.1f pp", padding=3)
    axes[1].set_ylabel("Accuracy drop (percentage points)")
    axes[1].set_title("B -> B minus B -> A")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _plot_training_time(rows: list[dict], output: Path) -> None:
    config_ids = [
        config_id
        for config_id in EXPERIMENTS
        if any(row["config_id"] == config_id for row in rows)
    ]
    labels = [EXPERIMENTS[config_id].display_name for config_id in config_ids]
    totals = [_training_minutes(rows, config_id) for config_id in config_ids]

    figure, axis = plt.subplots(figsize=(10, 5.2))
    bars = axis.bar(labels, totals, color="#4C72B0")
    axis.bar_label(bars, fmt="%.1f min", fontsize=8, padding=3)
    axis.set_ylabel("Total training time for A and B (minutes)")
    axis.set_title("Training cost under the common protocol")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=18)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _read_teammate_baseline(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["setting"] = int(row["setting"])
        row["reported_accuracy"] = float(row["reported_accuracy"])
    return rows


def _plot_teammate_comparison(
    rows: list[dict],
    baseline_path: Path,
    output: Path,
) -> None:
    teammate = _read_teammate_baseline(baseline_path)
    settings = (1, 2, 3, 4)
    ours = [_accuracy(rows, "vit_p16_conv", setting) for setting in settings]
    reported = [
        next(row["reported_accuracy"] for row in teammate if row["setting"] == setting)
        for setting in settings
    ]
    x = list(range(len(settings)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    bars_ours = axis.bar(
        [value - width / 2 for value in x],
        ours,
        width,
        label="bc protocol (validation-selected)",
    )
    bars_teammate = axis.bar(
        [value + width / 2 for value in x],
        reported,
        width,
        label="teammate repository (reported best)",
    )
    axis.bar_label(bars_ours, fmt="%.1f", fontsize=8, padding=2)
    axis.bar_label(bars_teammate, fmt="%.1f", fontsize=8, padding=2)
    axis.set_xticks(x, [SETTING_LABELS[setting] for setting in settings])
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title("ViT patch-16 reference comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _write_markdown_summary(rows: list[dict], output: Path) -> None:
    lines = [
        "# 可选实验结果",
        "",
        "所有模型使用相同的数据划分、训练轮数、优化器和随机种子。"
        "最佳模型由验证集选择，FashionMNIST 官方测试集仅用于最终评价。",
        "",
        "| 配置 | 参数量 | A→A | B→B | A→B | B→A |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config_id, definition in EXPERIMENTS.items():
        available = [row for row in rows if row["config_id"] == config_id]
        if not available:
            continue
        parameter_count = available[0]["parameter_count"]
        values = [_accuracy(rows, config_id, setting) for setting in (1, 2, 3, 4)]
        lines.append(
            f"| {definition.display_name} | {parameter_count:,} | "
            + " | ".join(f"{value:.2f}%" for value in values)
            + " |"
        )

    lines.extend(
        [
            "",
            "## 训练耗时",
            "",
            "| 配置 | A、B 两个模型总耗时 |",
            "|---|---:|",
        ]
    )
    for config_id, definition in EXPERIMENTS.items():
        if not any(row["config_id"] == config_id for row in rows):
            continue
        lines.append(
            f"| {definition.display_name} | "
            f"{_training_minutes(rows, config_id):.2f} min |"
        )

    lines.extend(["", "## 位置泛化", ""])
    for config_id in GROUPS["model"]:
        if not any(row["config_id"] == config_id for row in rows):
            continue
        drop = _accuracy(rows, config_id, 2) - _accuracy(rows, config_id, 4)
        lines.append(
            f"- {EXPERIMENTS[config_id].display_name}: "
            f"B→B 到 B→A 下降 {drop:.2f} 个百分点。"
        )
    lines.extend(
        [
            "",
            "同学仓库数据在图中标记为 reported best；该仓库每个 epoch "
            "使用测试集记录并选择最佳结果，因此只作为外部参考，不与本实验作严格统计等价比较。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_visualizations(
    results_csv: str | Path,
    baseline_csv: str | Path,
    assets_dir: str | Path,
    summary_markdown: str | Path,
) -> None:
    rows = read_results(results_csv)
    assets = Path(assets_dir)
    assets.mkdir(parents=True, exist_ok=True)

    if all(any(row["config_id"] == item for row in rows) for item in GROUPS["model"]):
        _plot_group(
            rows,
            GROUPS["model"],
            "Model comparison under four position settings",
            assets / "model_comparison.png",
        )
        _plot_generalization(rows, assets / "position_generalization.png")
    if all(any(row["config_id"] == item for row in rows) for item in GROUPS["patch_size"]):
        _plot_group(
            rows,
            GROUPS["patch_size"],
            "ViT patch-size comparison",
            assets / "patch_size_comparison.png",
        )
    if all(
        any(row["config_id"] == item for row in rows)
        for item in GROUPS["patch_embedding"]
    ):
        _plot_group(
            rows,
            GROUPS["patch_embedding"],
            "Patch embedding comparison",
            assets / "patch_embedding_comparison.png",
        )
    if any(row["config_id"] == "vit_p16_conv" for row in rows):
        _plot_teammate_comparison(
            rows,
            Path(baseline_csv),
            assets / "teammate_baseline_comparison.png",
        )
    _plot_training_time(rows, assets / "training_time_comparison.png")
    _write_markdown_summary(rows, Path(summary_markdown))
